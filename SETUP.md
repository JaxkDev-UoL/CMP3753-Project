# Setup Instructions

Follow the clear steps below to set up the project.

# 1. Download the Datasets
### 1.1. Obtain the required dataset for the fine-tune script you will be running.
    
This will most likely be either:

- `MCV` generated from https://github.com/JaxkDev-UOL/CMP3753-Project-MCV
- `ABCD` pr-processed from https://github.com/JaxkDev-UOL/CMP3753-Project-ABCD

### 1.2. Move dataset(s) into `datasets/` directory

The resulting folder structure should look like this:
```
./datasets/abcd/abcd_********.jsonl
./datasets/mcv/***_minecraft_villager_dataset_*********-llama.jsonl
```


# 2. Download the Large Language Model
### 2.1. Request access to the Llama-3.1 gated model on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/).

***- This can take a few hours to a few days, you cannot proceed without this access. -***

### 2.2. Download the required Llama-3.1 Model from [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main?clone=true) using these specific instructions.
> This will most likely be the **Llama-3.1-8B-Instruct** model, earlier fine-tuning scripts may use **Llama-3.1-8B**, check which you need!

> _**!!! WARNING !!!**_
>
> You will need to create a `READ` personal access token, this can be done by following this link https://huggingface.co/settings/tokens/new?canReadGatedRepos=true&tokenType=read
>
> (SSH alternative is also available for cloning [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct?clone=true))

To download the model into the correct place execute [these commands](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct?clone=true) in order:

1. `git lfs install`
2. `git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct models/Llama-3.1-8B-Instruct`
    > When prompted for a password, use the access token created.

### 2.3. Verify the structure of the `models/` directory:
```
./models/Llama-3.1-8B-Instruct/model-*-of-*.safetensors
./models/Llama-3.1-8B-Instruct/tokenizer*.json
./models/Llama-3.1-8B/model-*-of-*.safetensors
./models/Llama-3.1-8B/tokenizer*.json
```

Ensure all directory structures match the given paths for the project to function correctly.

# 3. Setup Python and Dependencies

### 3.1. Hardware/Software requirements
- Your PC **MUST** have a GPU with at least 32GB Total Memory. (DOES NOT INCLUDE PC'S RAM)
- Your PC **MUST** be CUDA-Capable - https://developer.nvidia.com/cuda-zone (generally any Nvidia GPU (RTX series) will suffice)
- Your PC **MUST** have at least 64GB of free space.
- Your PC **MUST** have either a Linux or Windows OS running (Unknown AMD/32bit support).

### 3.2. Python
Install Python 3.11 from [Pythons Offical Website](https://www.python.org/downloads/release/python-3119/) - Note the last binary installer was provided with `3.11.9`, advanced users can compile `*>=3.11.10 < 3.12` from provided source code.

### 3.3 Virtual environment
To setup your virtual environment ensure your python installation is available as `python` and `pip` (optionally suffixed with `3` eg `python3` `pip3`)

1. Setup virtual environment using

    `python -m venv venv`

2. Activate the virtual environment:

    - Windows CMD: `C://FULL_PATH/TO/venv/Scripts/activate.bat`
    - Windows PS: `C://FULL_PATH/TO/venv/Scripts/Activate.ps1`
    - Linux: `source ./venv/bin/activate`

    You should now have a prefix in your command line `(venv)` - Refer to the official [python docs](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activate-a-virtual-environment) if any issues arise.

3. Install PyTorch via `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` - https://pytorch.org/get-started/locally/#start-locally

4. Install all other requirements via `pip install -r requirements.txt`

