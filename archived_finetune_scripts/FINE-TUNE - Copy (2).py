import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Configuration
MODEL_NAME = "models/Llama-3.1-8B-Instruct"  # Path to local model
DATASET_TRAIN_PATH = "./datasets/abcd/abcd_v1.1_processed_train.json"  # Path to local dataset
DATASET_TEST_PATH = "./datasets/abcd/abcd_v1.1_processed_test.json"  # Path to local dataset
OUTPUT_DIR = "./finetuned/2_llama_3_1_8b_instruct_abcd"  # Path to save the fine-tuned model
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 1e-4

# Load dataset
dataset = load_dataset("json", data_files={"train": DATASET_TRAIN_PATH, "test": DATASET_TEST_PATH})

# Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Add custom special tokens
special_tokens = ['[[action]]', '[[/action]]']
tokenizer.add_tokens(special_tokens, special_tokens=True)

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
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype="float16", 
    bnb_4bit_use_double_quant=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, quantization_config=bnb_config)
model.to(device)

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=4,  # Decrease rank if pos
    lora_alpha=16,  # Experiment with different alpha
    lora_dropout=0.1,  # Experiment with dropout
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
    logging_steps=10,
    save_steps=200,
    save_total_limit=200,
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
trainer.train()#resume_from_checkpoint=True)

# Save fine-tuned model
model.save_pretrained(OUTPUT_DIR + "/model/")
tokenizer.save_pretrained(OUTPUT_DIR + "/model/")

# merge models and save to full_model
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR + "/full_model/")
tokenizer.save_pretrained(OUTPUT_DIR + "/full_model/")

print("Fine-tuning complete. Model saved at", OUTPUT_DIR)




# 1_llama_3_1_8b_Instruct_abcd (no quantisation / 4bit quantisation)
# {'loss': 1.2984, 'grad_norm': 1.0197371244430542, 'learning_rate': 8.298066550493735e-08, 'epoch': 3.0}
# 100%|███████████████████████████████████| 12051/12051 [7:32:50<00:00,  3.06s/it]C:\Users\Jack\Documents\UniProject\llama\llama_env\Lib\site-packages\peft\utils\save_and_load.py:260: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
#   warnings.warn(
# {'train_runtime': 27172.2581, 'train_samples_per_second': 0.887, 'train_steps_per_second': 0.444, 'train_loss': 0.9869333031899443, 'epoch': 3.0}
# 100%|███████████████████████████████████| 12051/12051 [7:32:52<00:00,  2.25s/it]
# C:\Users\Jack\Documents\UniProject\llama\llama_env\Lib\site-packages\peft\utils\save_and_load.py:260: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
#   warnings.warn(
# Fine-tuning complete. Model saved at ./finetuned/1_llama_3_1_8b_Instruct_abcd


# 2_llama_3_1_8b_Instruct_abcd (4bit quantisation)
# {'loss': 1.401, 'grad_norm': 2.704669237136841, 'learning_rate': 8.298066550493736e-09, 'epoch': 3.0}
# 100%|███████████████████████████████████| 12051/12051 [1:05:21<00:00,  3.14it/s]C:\Users\Jack\Documents\UniProject\llama\llama_env\Lib\site-packages\peft\utils\save_and_load.py:260: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
#   warnings.warn(
# {'train_runtime': 3923.7139, 'train_samples_per_second': 6.143, 'train_steps_per_second': 3.071, 'train_loss': 1.4961791990822395, 'epoch': 3.0}
# 100%|███████████████████████████████████| 12051/12051 [1:05:23<00:00,  3.07it/s]
# C:\Users\Jack\Documents\UniProject\llama\llama_env\Lib\site-packages\peft\utils\save_and_load.py:260: UserWarning: Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
#   warnings.warn(
# Fine-tuning complete. Model saved at ./finetuned/2_llama_3_1_8b_Instruct_abcd