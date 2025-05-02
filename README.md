# CMP3753-Project - Action Tokens in LLM's

This repository stores the main brain of my project:
- Finetuning Llama models
- Running inference on finetuned models.

Required dependencies such as datasets and models can be found below:

Datasets:
- `MCV` @ [CMP3753-Project-MCV](https://github.com/JaxkDev-UOL/CMP3753-Project-MCV)
- `ABCD` @ [CMP3753-Project-ABCD](https://github.com/JaxkDev-UOL/CMP3753-Project-ABCD)

Models:
- `Llama 3.1 8B Instruct` @ [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/)
- `Llama 3.1 8B` @ [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B/)

<br /><br />

# Setup [COMPULSORY]

See [SETUP.md](SETUP.md) for detailed setup instructions.

<br /><br />

# Fine Tuning [COMPULSORY]

A nice and easy one-liner after you've gone through all the effort of setup :)

`(venv) python finetune.py`

> `(venv)` indicates the virtual environment has been activated, see [SETUP.md#virtual-environment](SETUP.md#33-virtual-environment) for a revision on how to do that.

To run *Archived* finetuning scripts, copy them out of the `archived_finetune_scripts` directory and into the root directory and then do the same:

`(venv) python old_finetune.py`

<br /><br />

# Running Inference [Optional]

Another one-liner for running the latest finetuned model *(Note: the finetuning **MUST** be completed first)*

`(venv) python inference.py`

> Note: this inference.py is specific to the latest finetune.py found in root of the project, different finetune scripts cause slightly different parameters and especially in earlier version (0-20) the inference script had to be changed to run each finetuned model.
>
> If you are desperate to run these old models, see `archived_inference_scripts` which may contain the needed parameters for loading certain older models.
>
> **-- NO SUPPORT FOR OLD MODELS --**

<br /><br />

# License
    CMP3753 Project - Action Tokens in LLM's
    Copyright (C) 2025 Jack Honour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.