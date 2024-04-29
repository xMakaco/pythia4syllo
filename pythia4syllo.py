"""
Use the pythia4syllo to interact with Pythia.
Usage:
    pythia4syllo.py [--model_size=<str> --modality=[chat|dataset] --syllogism_data_path=<path> --device=<str>]
    pythia4syllo.py (-h | --help)
    pythia4syllo.py --version

Options:
    -h --help                           Show this screen.
    -v --version                        Show version.
    -s --model_size=<str>               The size of pythia to use [default: 70M].
                                        Possible sizes are: "70M", "160M", "410M", "1.0B", "1.4B", "2.8B", "6.9B", "12B".
    -m --modality=[chat|dataset]        Either enter a chat-style interaction or load a dataset [default: chat].
    -p --syllogism_data_path=<path>     The path to the syllogism_data.tsv. [default: syllomaker_master/toy_syllogism_data.tsv]
    -d --device=<str>                   Choose the device on which to run the model [default: cpu]
"""


""" Import Libraries """
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast  # Pythia via Hugging Face
import torch  # Needed to work with tensors
from docopt import docopt  # Used to use argparser and to manage the version of the program
import transformers  # Ensure reproducibility
import pandas as pd  # Load and create DataFrames to store results
import os  # Libraries to create the log file
from tqdm import tqdm  # A simple progress bar



""" Docopt is used to provide a little documentation for the program """
args = docopt(__doc__, version = 'Pythia for Syllogisms, ver 0.3')
model_size = args["--model_size"]   # Default is "70M"
modality = args["--modality"]       # Default is "chat"

# If modality is dataset, load the syllogisms
if modality == "dataset":
    if args["--syllogism_data_path"] is None:
        raise Exception("Please specify dataset filepath with -d\nSee usage with -h")
    syllogism_data_path = args["--syllogism_data_path"]
    # Check that the file exists
    if not os.path.exists(syllogism_data_path):
        raise Exception("Can't find the 'syllogism_data.tsv' file")
    print(f"# {syllogism_data_path} found #")


# Pythia comes in 8 sizes, in standard and deduped version (https://github.com/EleutherAI/pythia)
# We are using the standard one
# If the chosen model size is not valid, print usage and exit
#possible_sizes = ["70M", "160M", "410M", "1.0B", "1.4B", "2.8B", "6.9B", "12B"]
#if model_size not in possible_sizes or modality not in ["chat", "dataset"]:
#    raise Exception("modality has to be either 'chat' or 'dataset'")


def load_model_and_tokenizer(model_size):
    """
    Choose and load model
    input: model_size
    output: model, tokenizer, device
    """
    #transformers.set_seed(0)  # Ensure reproducibility
    transformers.logging.set_verbosity_error()  # Silence warning

    print(f"# Loading model pythia-{model_size}... #")

    # Load the model
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}",           # The model size
        cache_dir=f"./cache/pythia-{model_size}")    # Where the model cache is downloaded
    # Load the tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(
        f"EleutherAI/pythia-{model_size}",
        cache_dir=f"./cache/pythia-{model_size}") 

    # Check for CUDA (faster), else run on CPU
    if torch.cuda.is_available():
        device = "cuda:0" if args["--device"] == 'cuda' else 'cpu'
    else: device = "cpu"
    print(f"# The model pythia-{model_size} is loaded on the device: {device} #")
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
        max_new_tokens=75,        # The number of new tokens, i.e. the lenght of the answer of the model
        temperature=0.5,          # Randomness, see https://huggingface.co/blog/how-to-generate#:~:text=A%20trick%20is,look%20as%20follows.
        top_p=0.6,                # See https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling
        top_k=5,                  # See https://huggingface.co/blog/how-to-generate#top-k-sampling
        repetition_penalty=1.0,   # See https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TFRepetitionPenaltyLogitsProcessor.repetition_penalty
        do_sample=True,           # See https://stackoverflow.com/a/71281111/21343868
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


def chat_with_model():
    """
    Interact with the model with manual input
    """

    # Prepare the log file
    chat_log_file = "./chat_log.tsv"
    chat_log_df = pd.DataFrame(columns=['model_size', 'prompt', 'model_response'])

    # Enter the chat
    print(f"# GPT-NeoX Chat Interface with pythia-{model_size}. Type 'exit' to quit #")
    while True:
        chat_input = input("Prompt: ")
        if chat_input.lower() == 'exit': # 'exit' to quit
            print("# Your chat log is stored in 'chat_log.tsv' #")
            print("# Goodbye... #")
            break

        # Generate answer
        response = ask_question(chat_input)
        reduced_r = response[len(chat_input):]  # Don't print the chat input again
        print(f'Model response: {reduced_r}')

        # Store answer
        current_chat = {
            'model_size': model_size,
            'prompt': chat_input,
            'model_response': reduced_r
        }
        
        # Update the chat_log.tsv
        new_row_chat = pd.DataFrame([current_chat])  # Combine the new answer in a Dataframe
        chat_log_df = pd.concat([chat_log_df, new_row_chat], ignore_index=True)  # Concatenate the new answer to the log DataFrame
        chat_log_df.to_csv(chat_log_file, sep='\t', index=False)  # Create the log.tsv from the log DataFrame



def run_model_on_data():
    """
    Load a data.tsv and run the model for each row
    input: path to the syllogism_data.tsv
    output: a log.tsv file
    """
    
    # Load the data into pandas
    syllogism_data = pd.read_csv(syllogism_data_path, delimiter="\t")
    new = True

    # Load the log file if it exists
    if os.path.exists(f"log_{model_size}_{modality}.tsv"):
        data_log_df = pd.read_csv(f"log_{model_size}_{modality}.tsv", delimiter='\t')
        new = False
    else:
        # Create the file where to store log if it doesn't exist
        data_log_columns = ['type', 'context', 'premise_1', 'premise_2', 'conclusion', 'conclusion_2', 'trigger', 'model_response']
        data_log_df = pd.DataFrame(columns=data_log_columns)
        
    # Ask for context and trigger
    context = input("(Optional) Context: ")
    trigger = input("Trigger: ")

    # For each row in the syllogism file
    # tqdm is used to have a progress bar
    print("# Generating answers... #")
    for index, row in tqdm(syllogism_data.iterrows(), total=syllogism_data.shape[0]):
        type = row.type
        p1 = row.premise_1;     p2 = row.premise_2
        c1 = row.conclusion;    c2 = row.conclusion_2

        # Combine the full prompt
        full = context + p1 + p2 + trigger  
        model_response = ask_question(full)
        model_response = model_response[len(full):]  # Cut the answer of the model (remove the prompt)

        # Assign each variable to the right column to create the final log
        new_result = {
            'type': type,
            'context': context,
            'premise_1': p1, 'premise_2': p2,
            'conclusion': c1, 'conclusion_2': c2,
            'trigger' : trigger,
            'model_response': model_response
        }

        # Combine the new answer in a Dataframe
        new_row = pd.DataFrame([new_result])
        # Concatenate the new answer to the log DataFrame
        data_log_df = pd.concat([data_log_df, new_row], ignore_index=True)
    
    data_log_df.to_csv(f"log_{model_size}_{modality}.tsv", sep='\t', index=False)
    if new == True:
        print(f"# 'log_{model_size}_{modality}.tsv' created! #")
    else:
        print(f"# 'log_{model_size}_{modality}.tsv' updated! #")
    return



""" HERE THE MODEL IS ACTUALLY LOADED AND THE CHOSEN MODALITY RUN """
model, tokenizer, device = load_model_and_tokenizer(model_size)
if modality == "chat": chat_with_model()
elif modality == "dataset": run_model_on_data()
    