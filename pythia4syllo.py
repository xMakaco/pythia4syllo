"""Use the pythia4sillo to interact with Pythia.
Usage:
    pythia4syllo.py --model_size=[70M|160M|410M|1.0B|1.4B|2.8B|6.9B|12B]  --modality=[user|dataset]
    pythia4syllo.py (-h | --help)
    (-h | --help) --version

Options:
    -h --help                                               Show this screen.
    --version                                               Show version.
    --model_size=[70M|160M|410M|1.0B|1.4B|2.8B|6.9B|12B]    The size of pythia to use. [default: 70M]
    --modality=[user|dataset]                               User inputed prompt or prompt to be taken from a dataset. [default: user]
"""


""" Import Libraries """
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast  # Pythia via Hugging Face
import torch  # Needed to work with tensors
from datetime import datetime  # Date and time, used to create a unique ID for each interaction
import os, csv  # Libraries to create the log file
from docopt import docopt  # Used to use argparser and to manage the version of the program
import sys  # To exit the program and print error when using the program wrong



if __name__ == '__main__':

    """ Docopt is used to provide a little documentation for the program """
    args = docopt(__doc__, version='Pythia for Syllogisms, ver 0.1')
    model_size = args["--model_size"]
    modality = args["--modality"]
    # TODO: Add other arguments to call different functions


    """ Choose and load model """
    # Pythia comes in 8 sizes, in standard and deduped version
    # https://github.com/EleutherAI/pythia
    # We are using the standard one
    possible_sizes = ["70M", "160M", "410M", "1.0B", "1.4B", "2.8B", "6.9B", "12B"]
    if model_size not in possible_sizes:
        print("Chosen model_size is not good. Retry")
        print("Possible sizes:", possible_sizes)
        sys.exit()

    # Optional: choose step of training. "All Pythia models trained for 143000"
    # Specifying no step means loading the fully trained model
    # It's possible to choose steps in 1k batches (so 1000, 2000, 3000 ... 142000, 143000)
    step = "" # Add a "/" at the beginning of the number, like "/1000"

    # Load the model
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}{step}",            # The model size
        cache_dir=f"./cache/pythia-{model_size}{step}",     # Where the model is downloaded
    )

    # Load the tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(
        f"EleutherAI/pythia-{model_size}{step}",
        cache_dir=f"./cache/pythia-{model_size}{step}",  
    )

    # Check for CUDA (faster), else run on CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n{5*'#'} YOU ARE USING THE {device.upper()} {5*'#'}")


    def ask_question(prompt):
        """ Interact with the model """
        # TODO: add input and output types
        # TO-DISCUSS: choose top_p, top_k and repetition_penalty

        # If possible, load both the model and the tokenizer to CUDA to speed up the process
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model.to(device)

        # Generate answer with the specified hyperparameters
        tokens = model.generate(
            **inputs,
            #max_length=200,           #  Maximum length of the output. (output = user_input + answer). Use max_new_tokens instead?
            max_new_tokens=50,        #  The number of new tokens, i.e. the lenght of the answer of the model
            temperature=0.1,          #  Randomness, see https://huggingface.co/blog/how-to-generate#:~:text=A%20trick%20is,look%20as%20follows.
            top_p=0.6,                #  See https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling
            top_k=5,                  #  See https://huggingface.co/blog/how-to-generate#top-k-sampling
            repetition_penalty=1.0,   #  See https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TFRepetitionPenaltyLogitsProcessor.repetition_penalty
            do_sample=True,           #  See https://stackoverflow.com/a/71281111/21343868
            no_repeat_ngram_size=2,
            num_beams=2,
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

        file_path = f"/log_{model_size}.tsv"

        # The list of parameters we want to log
        log_components = ["interaction_id", "premises", "expected_answer", "trigger", "other", "model_response"]

        # If the file does not exist
        if not os.path.exists(file_path):
            # Create the file
            with open(file_path, 'w', newline='') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
                # Define the first row of the tsv (its header)
                writer.writerow([x for x in log_components])
                print('"log.tsv" not found, created')

            # Append the text in the file.
            with open(file_path, 'a', newline='') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
                writer.writerow(text)
            #print("Log updated")


    def chat_with_model():
        """ Interact with the model with manual input """

        # Store the time
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"{5*'#'} GPT-NeoX Chat Interface. Type 'exit' to stop. {now}  {5*'#'}\n")

        # Store the number of the interaction
        prompt_num = 1

        while True:
            print(f"{3*'#'} Interaction number {prompt_num} {3*'#'}")
            user_input = input("Input your prompt - ('exit' to quit): ")

            # 'exit' to quit
            if user_input.lower() == 'exit':
                print("Closing...")
                break

            # Generate answer
            response = ask_question(user_input)
            reduced_r = response[len(user_input):]  # Don't print the user input again
            print(f'Model {model_size}, answer to "{user_input}": \n {reduced_r}')
            print(f"{3*'#'} End of interaction {prompt_num} {3*'#'}\n")
            prompt_num += 1

    def chat_and_save_log():
        """ Interact with the model and automatically save log """
        # TODO: Prepare the code to load the datasets of syllogisms

        # For now, manually input your prompt in "premises"
        premises = "All grapes are interferences. All interferences are jackets. "
        trigger = ["Therefore", "It follows that", "We conclude that", "What follows", "The conlcusion is", "We can conclude that", "A simple inference is"]
        other = [":", "...", ",", "?", "", ", in short: ", ", in short, "]
        expected_answer = "All grapes are jackets."

        # Unique id for each interaction (example: "23/04/2023 12:33:54" becomes "23042023_123354")
        date_time = datetime.now().strftime("%d%m%Y_%H%M%S")
        model_specs = f"pythia-{model_size}"
        id = 1

        for t in trigger:
            for o in other:
                print(f"Prompt: {premises + t + o}")
                model_response = ask_question(premises + t + o)  # Join together the pieces
                model_response = model_response[len(premises):]  # Remove the premises from the output
                log_components = [f"{date_time}_{id}", f'"{premises}"', f'"{expected_answer}"', t, o, f'"{model_response}"']
                save_log(log_components)
                id += 1
        print("Done!")


    """ Call the function associated to the modality called """
    if modality == "user":  # Default
        chat_with_model()
    elif modality == "dataset":
        chat_and_save_log()
    