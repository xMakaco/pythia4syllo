# Pythia4syllo

Pythia4syllo is a Python program designed to facilitate interaction with the Pythia model in a chat-styled manner or to probe it with a dataset of syllogisms.

## Usage

An example of how to use Pythia4syllo is as follows:

```bash
python3 pythia4syllo.py --model_size 12B --modality chat --device cuda
```

You can provide the following arguments:

- `--model_size=<str>`: Specifies the Pythia model to load. For example, you can specify "12B" to load a 12-billion parameter model.
- `--modality=[chat|dataset]`: Choose between interacting with the model in a chat-styled manner or loading a dataset for probing.
- `--syllogism_data_path=<path>` (optional): Specify the path to the syllogism data created in the style of Syllomaker.
- `--device=[cuda|cpu]`: Choose the device where to load the model, either CUDA for GPU acceleration or CPU.

## Logging

Each interaction, whether in chat mode or when loading a dataset, is stored in a log file with the current model and modality in use. For example, if you are using a 12B model with a dataset, the log file will be named "log_12B_dataset.tsv".

By separating each interaction into log files, analysis and tracking of model performance become more manageable.


## Main Functions explanation

### `load_model_and_tokenizer(model_size)`

Choose and load the Pythia model and tokenizer.

- **Input:** `model_size` - The size of the Pythia model.
- **Output:** `model`, `tokenizer`, `device` - The loaded model, tokenizer, and the device where the model is loaded.

### `ask_question(prompt)`

Actually interrograte the model. Here the model hyperparameters can be found.

- **Input:** `prompt` - The prompt to interact with the model.
- **Output:** `answer` - The generated response from the model.

### `chat_with_model()`

Interact with the Pythia model using manual input.

- **Functionality:** Engages in a chat-style interaction with the loaded model, allowing users to input prompts and receive responses.
- **Output:** Stores the chat log in 'chat_log.tsv'.

### `run_model_on_data()`

Load a dataset and run the Pythia model for each row.

- **Functionality:** Loads a dataset of syllogisms and runs the Pythia model for each row, generating responses.
- **User Inputs:**
  - `context` - Optional, space where to add any kind of contextual knowledge about the task at hand, the definition of syllogisms, etc.
  - `trigger` - The phrase that prompts the model to give a logical answer, i.e. "It follows that: ", "Therefore, ", etc.
- **Output:** Stores the results in a log file.

## Additional Libraries

The required libraries can be installed with: 

```
pip install -r requirements.txt
```

Here, a general overlook on them:

- `transformers`: Used to load the Pythia model.
- `torch`: Needed to work with tensors.
- `docopt`: Used for argument parsing and managing the version of the program.
- `pandas`: Used to load and create DataFrames to store results.
- `os`: Libraries to manage files and create the log file.
- `tqdm`: Provides a simple progress bar for iterating over the dataset.
