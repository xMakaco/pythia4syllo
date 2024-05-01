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

---

Feel free to enhance Pythia4syllo's capabilities and contribute to its development. If you encounter any issues or have suggestions, please create an issue on GitHub or submit a pull request.
