DATA = {{'loss': 3.0316, 'grad_norm': 1.7991057634353638, 'learning_rate': 0.0009995850966724752, 'epoch': 0.0},
{'loss': 2.5527, 'grad_norm': 7.56449556350708, 'learning_rate': 0.0009991701933449507, 'epoch': 0.0} }

# Takes 77H to run...

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Configuration
MODEL_NAME = "models/Llama-3.1-8B-Instruct"  # Path to local model
DATASET_TRAIN_PATH = "./datasets/abcd/abcd_v1.1_processed_train.json"  # Path to local dataset
DATASET_TEST_PATH = "./datasets/abcd/abcd_v1.1_processed_test.json"  # Path to local dataset
OUTPUT_DIR = "./finetuned/1_llama_3_1_8b_Instruct_abcd"  # Path to save the fine-tuned model
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-3

# Load dataset
dataset = load_dataset("json", data_files={"train": DATASET_TRAIN_PATH, "test": DATASET_TEST_PATH})

# Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Add custom special tokens
special_tokens = ['[[action]]', '[[/action]]']
tokenizer.add_tokens(special_tokens)

# Tokenization function
def tokenize_function(example):
    encoding = tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=256,
        padding_side="right",
    )
    encoding["labels"] = encoding["input_ids"]  
    return encoding

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=4)

# Check for NaN or None in tokenized data
def check_for_invalid_tokens(tokenized_data):
    for split_name, data in tokenized_data.items():
        for i in range(5):
            for column in data.column_names:
                column_data = data[column][i]
                if isinstance(column_data, list):
                    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in column_data):
                        print(f"Invalid value found in column {column} for row {i}")

# Check tokenized data for issues
#check_for_invalid_tokens(tokenized_datasets)

# Model Setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True
)

# Load model with BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")#, quantization_config=bnb_config)
model = model.to("cuda")

# Apply LoRA
lora_config = LoraConfig(
    r=4,  
    lora_alpha=16,  
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Resize model token embeddings to account for the new special tokens
model.resize_token_embeddings(len(tokenizer))

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir="./logs",
    logging_steps=1,
    save_steps=500,
    save_total_limit=2,
    max_grad_norm=5.0, 
    gradient_accumulation_steps=1,
    run_name="finetune_llama_3_1_8b_instruct_abcd",
    #report_to="tensorboard",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Start training
trainer.train()

# Save fine-tuned model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning complete. Model saved at", OUTPUT_DIR)
