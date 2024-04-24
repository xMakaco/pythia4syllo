""" Import Libraries """
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast  # Pythia via Hugging Face
import torch  # Needed to work with tensors
from datetime import datetime  # Date and time, used to create a unique ID for each interaction
import os, csv  # Libraries to create the log file


""" Choose and load model """
# Pythia comes in 8 sizes, in standard and deduped version
# https://github.com/EleutherAI/pythia
# We are using the standard one
model_size = "70M"  # sizes = ["70M", "160M", "410M", "1.0B", "1.4B", "2.8B", "6.9B", "12B"]

# Optional: choose step of training. "All Pythia models trained for 143000"
# Specifying no step means loading the fully trained model
# It's possible to choose steps in 1k batches (so 1000, 2000, 3000 ... 142000, 143000)
step = "" # Add a "/" at the beginning of the number, like "/1000"

# Load the model
model = GPTNeoXForCausalLM.from_pretrained(
    f"EleutherAI/pythia-{model_size}{step}",
    cache_dir=f"./pythia-{model_size}{step}",
)

# Load the tokenizer
tokenizer = GPTNeoXTokenizerFast.from_pretrained(
    f"EleutherAI/pythia-{model_size}{step}",
    cache_dir=f"./pythia-{model_size}{step}",
)

# Check for CUDA (faster), else run on CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"\n{5*'#'} YOU ARE USING THE {device.upper()} {5*'#'}")


def ask_question(
    prompt,                  #  Our prompt to interact with the model
    max_length=100,          #  Maximum length (in tokens) of the INPUT
    temperature=0.1,         #  Randomness, see https://huggingface.co/blog/how-to-generate#:~:text=A%20trick%20is,look%20as%20follows.
    top_p=0.65,               #  See https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling
    top_k=5,                #  See https://huggingface.co/blog/how-to-generate#top-k-sampling
    repetition_penalty=1.0   #  See https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TFRepetitionPenaltyLogitsProcessor.repetition_penalty
    ):
    """ Interact with the model """
    # TODO: add input and output types
    # TO-DISCUSS: choose top_p, top_k and repetition_penalty

    # If possible, load both the model and the tokenizer to CUDA to speed up the process
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device)

    # Generate answer with the specified hyperparameters
    # tokens is a torch.tensor
    tokens = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,  # See https://stackoverflow.com/a/71281111/21343868
        top_p=top_p,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
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
        print(f'Model {model_size}, answer to "{user_input}": \n {response}')
        print(f"{3*'#'} End of interaction {prompt_num} {3*'#'}\n")
        prompt_num += 1


chat_with_model()