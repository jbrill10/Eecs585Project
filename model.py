from datasets import load_dataset
from dataset import PlatypusDataset, track_performance
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import wandb, os
import transformers
from datetime import datetime
from accelerate import PartialState
import matplotlib.pyplot as plt
access_token = "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"
torch.cuda.empty_cache()

wandb.login()
wandb_project = "llama-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
    
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

max_length = 2048 # This was an appropriate max length for my dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result
    
base_model_id = "meta-llama/Llama-3.1-8B"
# base_model_id = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map={"": PartialState().process_index}, token=access_token, )
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    use_fast=False, # needed for now, should be fixed soon
    trust_remote_code=True,
    token=access_token
)
tokenizer.pad_token = tokenizer.eos_token

accelerator = Accelerator()

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()

# plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

def formatting_func(example):
    text = f"### Question: {example['instruction']}\n ### Answer: {example['output']}"
    return text




@track_performance
def finetune_new_model_on_dataset(tokenized_train_dataset, tokenized_val_dataset, base_model_id, base_model, accelerator, source_name):    
    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(model, config)
    # print_trainable_parameters(model)
    
    # Apply the accelerator. You can comment this out to remove the accelerator.
    model = accelerator.prepare_model(model)

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    project = "llama-finetune/ablation/"
    run_name = base_model_id + "-" + project + source_name
    output_dir = "./" + run_name
    
    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=500,
            learning_rate=2.5e-5,
            logging_steps=50,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=25,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=50,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to="wandb",           # Comment this out if you don't want to use weights & baises
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

def main():
    dataset = PlatypusDataset()
    
    data_sources = ['reclor','MATH/PRM-800K','airoboros']

    for source in data_sources:
        print(f"source: {source}")
        train_dataset = dataset.get_train_data_without_source(source)
        val_dataset = dataset.get_val_data_without_source(source)
        # tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
        # tokenized_val_dataset = val_dataset.map(generate_and_tokenize_prompt2)
        tokenized_train_dataset = [generate_and_tokenize_prompt2(point).to("cuda") for point in train_dataset]
        tokenized_val_dataset = [generate_and_tokenize_prompt2(point).to("cuda") for point in val_dataset]
        print(f"Training set size: {len(tokenized_train_dataset)}")
        print(f"Validation set size: {len(tokenized_val_dataset)}")
        # plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)
        finetune_new_model_on_dataset(tokenized_train_dataset, tokenized_val_dataset, base_model_id, model, accelerator, source)

if __name__ == "__main__":
    main()

