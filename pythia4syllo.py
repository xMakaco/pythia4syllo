"""
Use the pythia4sillo to interact with Pythia.
Usage:
    pythia4syllo.py [--model_size=<str> --modality=[chat|dataset]]
    pythia4syllo.py (-h | --help)
    pythia4syllo.py --version

Options:
    -h --help                       Show this screen.
    -v --version                    Show version.
    -s --model_size=<str>           The size of pythia to use [default: 70M].
                                    Possible sizes are: "70M", "160M", "410M", "1.0B", "1.4B", "2.8B", "6.9B", "12B".
    -m --modality=[chat|dataset]    chat inputed prompt or prompt to be taken from a dataset [default: chat].
"""


""" Import Libraries """
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast  # Pythia via Hugging Face
import torch  # Needed to work with tensors
from datetime import datetime  # Date and time, used to create a unique ID for each interaction in the chat
from docopt import docopt  # Used to use argparser and to manage the version of the program
from transformers import set_seed  # Ensure reproducibility
import pandas as pd  # Load and create DataFrames to store results
import os, csv, sys  # Libraries to create the log file
from tqdm import tqdm  # A simple progress bar



""" Docopt is used to provide a little documentation for the program """
args = docopt(__doc__, version='Pythia for Syllogisms, ver 0.1')
model_size = args["--model_size"]   # Default is "70M"
modality = args["--modality"]       # Default is "chat"


# Pythia comes in 8 sizes, in standard and deduped version (https://github.com/EleutherAI/pythia)
# We are using the standard one
# If the chosen model size is not valid, print usage and exit
possible_sizes = ["70M", "160M", "410M", "1.0B", "1.4B", "2.8B", "6.9B", "12B"]
if model_size not in possible_sizes or modality not in ["chat", "dataset"]:
    print(docopt(__doc__, "-h"))
    # exit program


def load_model_and_tokenizer(model_size):
    """
    Choose and load model
    input: model_size
    output: model, tokenizer, device
    """
    set_seed(0)  # Ensure reproducibility

    # Load the model
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}",           # The model size
        cache_dir=f"./cache/pythia-{model_size}")    # Where the model cache is downloaded
    # Load the tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(
        f"EleutherAI/pythia-{model_size}",
        cache_dir=f"./cache/pythia-{model_size}") 

    # Check for CUDA (faster), else run on CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{5*'#'} YOU ARE USING THE {device.upper()} {5*'#'}\n")
    return model, tokenizer, device


def ask_question(prompt):
    """
    Interact with the model
    input: prompt
    output: answer (cut, without prompt)
    """
    # TODO: add input and output types

    # If possible, load both the model and the tokenizer to CUDA to speed up the process
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device)

    # Generate answer with the specified hyperparameters
    tokens = model.generate(
        **inputs,
        #max_length=200,          # SET TO DEFAULT
        max_new_tokens=25,        #  The number of new tokens, i.e. the lenght of the answer of the model
        temperature=0.1,          #  Randomness, see https://huggingface.co/blog/how-to-generate#:~:text=A%20trick%20is,look%20as%20follows.
        top_p=0.6,                #  See https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling
        top_k=5,                  #  See https://huggingface.co/blog/how-to-generate#top-k-sampling
        repetition_penalty=1.0,   #  See https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TFRepetitionPenaltyLogitsProcessor.repetition_penalty
        do_sample=True,           #  See https://stackoverflow.com/a/71281111/21343868
        #no_repeat_ngram_size=2,
        #num_beams=2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        )

    # Decode answer
    # We take the first element of tokens as there each token ID is stored
    # Its other element is the associated device (e.g., "cpu")
    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    answer = answer.replace("\n", " ")  # All in one line to store in a tsv
    answer = answer.replace("\t", " ")  # Remove eventual tabs
    return answer


def save_log(text):
    """
    This function enables to save the parameters of the interaction in a log file
    The expected input of "text" is a list of strings
    Each element of the list will be placed in its column
    Each column is separated by a tab charachter ('\t', four spaces)
    """
    # TODO: add input and output types

    file_path_minilog = f"log_{model_size}_{modality}.tsv"

    # The list of parameters we want to log
    # The list of parameters we want to log
    log_components = ["prompt", "response"]

    # If the file does not exist
    if not os.path.exists(file_path_minilog):
        # Create the file
        with open(file_path_minilog, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            # Define the first row of the tsv (its header)
            writer.writerow([x for x in log_components])
            print('"log.tsv" not found, created')

        # Append the text in the file.
        with open(file_path_minilog, 'a', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(text)
        #print("Log updated")


def chat_with_model():
    """
    Interact with the model with manual input
    """

    # Store the time
    now = datetime.now().strftime("%d%m%Y_%H%M%S")
    print(f"{5*'#'} GPT-NeoX Chat Interface with Pythia {model_size}. Type 'exit' to stop. {now}  {5*'#'}\n")
    prompt_num = 1

    # Store the number of the interaction
    while True:
        chat_input = input("Input your prompt - ('exit' to quit): ")
        # 'exit' to quit
        if chat_input.lower() == 'exit':
            print("Closing...")
            break
        # Generate answer
        response = ask_question(chat_input)
        reduced_r = response[len(chat_input):]  # Don't print the chat input again
        print(f'Model {model_size}, answer to "{chat_input}": \n {reduced_r}')
        print(f"{3*'#'} End of interaction {prompt_num} {3*'#'}\n")

        # Store answer
        log = [chat_input, reduced_r]
        save_log(text=log, mod="short", model_size=model_size, modality=modality)
        prompt_num += 1


def run_model_on_data(syllogism_path, trigger):
    """
    Load a data.tsv and run the model for each row
    input: path to the syllogism_data.tsv
    output: a log.tsv file
    """

    # Check that the file exists
    if not os.path.exists(syllogism_path):
        print("Can't find the 'syllogism_data.tsv' file")
        sys.exit()
    
    # Load the data into pandas
    data = pd.read_csv(syllogism_path, delimiter="\t")

    # Create an empty DataFrame with these columns
    log_columns = ['mood_figure', 'premise_1', 'premise_2', 'trigger', 'conclusion_1', 'conclusion_2', 'model_response']
    log_dataframe = pd.DataFrame(columns=log_columns)

    # For each row in the syllogism file
    # tqdm is used to have a progress bar
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        mood_figure = row.mood_figure
        p1 = row.premise_1 + ". "  # Add a period and a space
        p2 = row.premise_2 + ". "
        c1 = row.conclusion
        c2 = row.conclusion_2

        full = p1 + p2 + trigger  # Combine the full prompt
        model_response = ask_question(full)
        model_response = model_response[len(full):]  # Cut the answer of the model (remove the prompt)
        #print(full + model_response)

        # Assign each variable to the right column to create the final log
        new_result = {
            'mood_figure': mood_figure,
            'premise_1': p1,
            'premise_2': p2,
            'trigger' : trigger,
            'conclusion_1': c1,
            'conclusion_2': c2,
            'model_response': model_response
        }

        # Combine the new answer in a Dataframe
        new_row = pd.DataFrame([new_result])
        # Concatenate the new answer to the log DataFrame
        log_dataframe = pd.concat([log_dataframe, new_row], ignore_index=True)
    
    # Create the log.tsv from the log DataFrame
    file_name = f"log_{model_size}_{modality}.tsv"
    log_dataframe.to_csv(file_name, sep='\t', index=False)
    print("Done!")




""" HERE THE MODEL IS ACTUALLY LOADED AND THE CHOSEN MODALITY RUN """

model, tokenizer, device = load_model_and_tokenizer(model_size)
syllogism_path = "./syllomaker_program/syllogism_data.tsv"

if modality == "chat":  # Default
    chat_with_model()
elif modality == "dataset":
    run_model_on_data(syllogism_path, trigger="It follows that: ")
    