#!/bin/bash
python3 -m venv env
source env/bin/activate

pip install sentencepiece
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q -U datasets scipy ipywidgets matplotlib einops
pip install accelerate
pip install -q wandb -U
pip install -r requirements.txt