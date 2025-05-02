import numpy as np
import torch
import psutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback

# Configuration
MODEL_NAME = "models/Llama-3.1-8B-Instruct"  # Path to local model
DATASET_TRAIN_PATH = "./datasets/abcd/abcd_v1.1_processed_train.json"  # Path to local dataset
DATASET_TEST_PATH = "./datasets/abcd/abcd_v1.1_processed_test.json"  # Path to local dataset
OUTPUT_DIR = "./finetuned/4_llama_3_1_8b_instruct_abcd"  # Path to save the fine-tuned model
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
LOGGING_STEPS = 15

# TensorBoard Logger
writer = SummaryWriter(log_dir="./logs/" + OUTPUT_DIR.split("/")[-1])

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
check_for_invalid_tokens(tokenized_datasets)


# Custom function to compute Perplexity
def compute_perplexity(loss):
    return np.exp(loss) if loss is not None else None

# Enhanced logging function
def log_metrics(log, step, model, validation_loss=None, validation_accuracy=None):
    # Loss and Perplexity Logging
    loss = log.get("loss")
    lr = log.get("learning_rate")
    epoch = log.get("epoch")

    if loss is not None:
        writer.add_scalar("Loss/train", loss, step)
        writer.add_scalar("Perplexity/train", compute_perplexity(loss), step)
    if validation_loss is not None:
        writer.add_scalar("Loss/validation", validation_loss, step)
        writer.add_scalar("Perplexity/validation", compute_perplexity(validation_loss), step)
    
    # Learning rate and Epoch Logging
    if lr is not None:
        writer.add_scalar("Learning_Rate/train", lr, step)
    if epoch is not None:
        writer.add_scalar("Epoch/train", epoch, step)
    
    # Validation Accuracy (if available)
    if validation_accuracy is not None:
        writer.add_scalar("Accuracy/validation", validation_accuracy, step)

    # Gradient Norm Logging (Track gradient issues or instability)
    total_grad_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item()**2
            count += 1
    if count > 0:
        writer.add_scalar("Grad_Norm", total_grad_norm**0.5, step)

    # Weight Norm Logging (Monitor weight changes)
    total_weight_norm = sum(p.data.norm(2).item()**2 for p in model.parameters())**0.5
    writer.add_scalar("Weight_Norm", total_weight_norm, step)

    
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                writer.add_scalar("Gradient_Exploding/NaN", 1, step)
                print(f"NaN detected in gradients at step {step}")
    
    
    if loss is not None and validation_loss is not None:
        writer.add_scalar("Loss/Train_vs_Validation", loss - validation_loss, step)

    writer.flush()



class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step <= 0 or step % LOGGING_STEPS != 0 or logs is None:
            return
        
        train_loss = logs.get("loss")

        # Log metrics
        log_metrics(logs, step, model)

    def on_epoch_end(self, args, state, control, **kwargs):
        validation_loss = trainer.evaluate()["eval_loss"]
        #print(f"Epoch {state.epoch}: Validation Loss = {validation_loss}")
        log_metrics({"eval_loss": validation_loss}, state.global_step, model, validation_loss)


# Model Setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype="float16", 
    bnb_4bit_use_double_quant=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)#, quantization_config=bnb_config)
model.to(device)

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16, 
    lora_alpha=16,  # Higherfor better scaling
    lora_dropout=0.2,  # Higher to prevent overfitting
    target_modules=["q_proj", "k_proj", "v_proj"],
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
    #logging_dir="./logs",
    logging_steps=LOGGING_STEPS,
    save_steps=100,
    save_total_limit=2,
    #max_grad_norm=5.0, 
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


# Add the callback to the trainer
trainer.add_callback(LoggingCallback())

trainer.train()#resume_from_checkpoint=True)

# Save fine-tuned model
model.save_pretrained(OUTPUT_DIR + "/model/")
tokenizer.save_pretrained(OUTPUT_DIR + "/model/")

# merge models and save to full_model
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR + "/full_model/")
tokenizer.save_pretrained(OUTPUT_DIR + "/full_model/")

writer.close()

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