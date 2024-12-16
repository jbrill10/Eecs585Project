# EECS585 Project

## Setup
Install dependencies by running ./install.sh

## Running Finetuning
Use the model.py file to run finetuning experiments. You can choose which datasets to finetune over by modifying the for loop in main(). The weights from the training run will be saved into a folder whose name will depend on the base model name and the name of the source you used when calling finetune_new_model_on_dataset(). The folder will use the following naming convention: 

weights directory = "meta-llama/" + model_name + "llama-finetune/ablation/" + source_name

It is recommended to pipe your console output to a folder so you can refer to it if something goes wrong during training.

Example: python3 model.py > training_output.txt

You can find all the data sampling methods in dataset.py, including both statistical and ablation-guided sampling. Here you can also find methods for accessing information in the dataset and logging function profiling information.

## Running Benchmarking
The code for running benchmarking can be found in the benchmark.ipynb notebook. In order to perform benchmarking, first upload model weights to your google drive so Colab is able to access them. Then run all the code blocks to set everything up. After that, follow the examples given at the bottom of the notebook and paste in the name of the file where you have copied the model weights. Once you run the block, using the provided benchmark functions, you will see output underneath the block. This can be written to a file, or simply copied down. 


## Third Party
Additional third party login are required for the notebooks.
For benchmarking, we need `huggingface` and  `deepeval` login.
For finetuning, we need `huggingface`, `deepeval`, and `wandb` login.
