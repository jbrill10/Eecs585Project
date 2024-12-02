# EECS585 Project

## Setup
A couple of library is required for running the evaluation and finetuning pipelines:
- Cuda
- Pytorch
Mine ran with Python 3.10 and CUDA 12.0

Additional libraries can be installed in Mistral-7B file. There is a section at the top that install required dependencies. After this, running the separate notebooks should be very straightforward.

## Third Party
Additional third party login are required for the notebooks.
For benchmarking, we need `huggingface` and  `deepeval` login.
For finetuning, we need `huggingface`, `deepeval`, and `wandb` login.
