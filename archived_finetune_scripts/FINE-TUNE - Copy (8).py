import os
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback, AdamW, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

# Configuration
ID = "8"
DATASET_ID = "2"
MODEL_NAME = "models/Llama-3.1-8B-Instruct"  # Path to local model
DATASET_PATH = f"./datasets/mcv/500_minecraft_villager_dataset_formatted_{DATASET_ID}-llama.txt"  # Path to local dataset
TOKENS_PATH = f"./datasets/mcv/500_minecraft_villager_dataset_tokens_{DATASET_ID}-llama.json"  # Path to the tokens json
OUTPUT_DIR = f"./finetuned/{ID}_llama_3_1_8b_instruct-TMPO"  # Path to save the fine-tuned model
EPOCHS = 1  # Increased number of epochs for better fine-tuning
BATCH_SIZE = 4  # Adjusted batch size to fit the GPU (RTX 4070 Ti Super)
LEARNING_RATE = 5e-5
LOGGING_STEPS = 20
WARMUP_STEPS = 200  # Warmup steps for better learning rate adjustment
GRADIENT_ACCUMULATION_STEPS = 2  # Use gradient accumulation to simulate a larger batch size

# Initialize TensorBoard Logger
def setup_tensorboard_log():
    return SummaryWriter(log_dir="./logs/" + OUTPUT_DIR.split("/")[-1])

# Load dataset
def load_data():
    # Load the dataset from the text file
    return load_dataset("text", data_files=DATASET_PATH, split="train")

# Load special tokens
def load_special_tokens():
    with open(TOKENS_PATH, 'r') as f:
        return json.load(f)

# Initialize tokenizer and add special tokens
def setup_tokenizer(special_tokens):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    return tokenizer

# Tokenization function
def tokenize_function(example, tokenizer):
    # Get the raw conversation text (entire block of conversation)
    conversation = example['text']  # Entire conversation as one string

    # Tokenize the entire conversation
    encoding = tokenizer(conversation, truncation=True, padding="max_length", max_length=256, return_tensors="pt")

    # The same text is used as labels for the output (as this is a conversational model)
    encoding["labels"] = encoding["input_ids"].clone()

    print(encoding)
    print(type(encoding['input_ids']))
    print(encoding['input_ids'].shape)

    # Convert to list of integers rather than PyTorch tensor
    encoding = {k: v.squeeze(0).tolist() for k, v in encoding.items()}

    return encoding

# Tokenize the entire dataset
def tokenize_dataset(dataset, tokenizer, batch_size=4, num_proc=16):
    return dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        # batched=True,
        # batch_size=batch_size,
        # num_proc=num_proc,
    )

# Compute Perplexity
def compute_perplexity(loss):
    return np.exp(loss) if loss is not None else None

# Enhanced logging function for training
def log_metrics(writer, log, step, model, validation_loss=None, validation_accuracy=None):
    loss = log.get("loss")
    lr = log.get("learning_rate")
    epoch = log.get("epoch")

    if loss is not None:
        writer.add_scalar("Loss/train", loss, step)
        writer.add_scalar("Perplexity/train", compute_perplexity(loss), step)
    if validation_loss is not None:
        writer.add_scalar("Loss/validation", validation_loss, step)
        writer.add_scalar("Perplexity/validation", compute_perplexity(validation_loss), step)
    
    if lr is not None:
        writer.add_scalar("Learning_Rate/train", lr, step)
    if epoch is not None:
        writer.add_scalar("Epoch/train", epoch, step)
    
    if validation_accuracy is not None:
        writer.add_scalar("Accuracy/validation", validation_accuracy, step)

    # Log gradient and weight norms
    log_gradient_and_weight_norms(writer, step, model)

    writer.flush()

# Log gradient and weight norms
def log_gradient_and_weight_norms(writer, step, model):
    total_grad_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item()**2
            count += 1
    if count > 0:
        writer.add_scalar("Grad_Norm", total_grad_norm**0.5, step)

    total_weight_norm = sum(p.data.norm(2).item()**2 for p in model.parameters())**0.5
    writer.add_scalar("Weight_Norm", total_weight_norm, step)

# Custom Logging Callback
class LoggingCallback(TrainerCallback):
    def __init__(self, writer, model, trainer):
        self.writer = writer
        self.model = model
        self.trainer = trainer
        # parent init
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = state.global_step
        if step <= 0 or step % LOGGING_STEPS != 0 or logs is None:
            return

        log_metrics(self.writer, logs, step, self.model)

    def on_epoch_end(self, args, state, control, **kwargs):
        validation_loss = self.trainer.evaluate()["eval_loss"]
        log_metrics(self.writer, {"eval_loss": validation_loss}, state.global_step, self.model, validation_loss)

# Setup model and apply LoRA for fine-tuning
def setup_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to(device)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    model = get_peft_model(model, peft_config)
    return model, device

# Main function for training
def train_model():
    torch.cuda.empty_cache()
    writer = setup_tensorboard_log()
    dataset = load_data()
    special_tokens = load_special_tokens()
    tokenizer = setup_tokenizer(special_tokens)

    tokenized_datasets = tokenize_dataset(dataset, tokenizer, batch_size=BATCH_SIZE, num_proc=8)

    total_tokens = 0
    # count total tokens in the dataset (input_ids) using least memory
    for example in tokenized_datasets:
       total_tokens += len(example["input_ids"])

    print(f"Total tokens in the dataset: {total_tokens}")

    model, device = setup_model()
    model.resize_token_embeddings(len(tokenizer))

    # Split the dataset into train and test
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)  # 80% train, 20% test
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=len(train_dataset) * EPOCHS,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,  # Warm-up steps for better training stability
        lr_scheduler_type="linear",  # Using cosine decay for learning rate
        logging_steps=LOGGING_STEPS,
        save_steps=100,
        save_total_limit=1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Simulate larger batch sizes
        report_to="none",
        fp16=True,  # Use mixed precision for training efficiency
        no_cuda=False,
        #weight_decay=0.01,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        optimizers=(optimizer, scheduler), 
    )

    trainer.add_callback(LoggingCallback(writer, model, trainer))

    # Resume training if checkpoint exists
    dirs = os.listdir(OUTPUT_DIR)
    resume_from_checkpoint = any("checkpoint" in d for d in dirs)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model = model.merge_and_unload()

    # Save fine-tuned model
    model.save_pretrained(OUTPUT_DIR + "/full_model/")
    tokenizer.save_pretrained(OUTPUT_DIR + "/full_model/")

    writer.close()

    print("Fine-tuning complete. Model saved at", OUTPUT_DIR)

# Entry point
if __name__ == '__main__':
    train_model()
