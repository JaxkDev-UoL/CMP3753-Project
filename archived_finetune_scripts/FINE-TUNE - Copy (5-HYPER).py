import json
import os
import torch
import optuna
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
from optuna.pruners import MedianPruner  # Import pruner

# Base Config
MODEL_NAME = "models/Llama-3.1-8B-Instruct"  # Local model path
DATASET_TRAIN_PATH = "./datasets/abcd/abcd_v1.1_processed_train.json"
DATASET_TEST_PATH = "./datasets/abcd/abcd_v1.1_processed_test.json"
OUTPUT_DIR = "./finetuned/5_llama_3_1_8b_instruct_abcd"  # Base output dir
EPOCHS = 2
LOGGING_STEPS = 15

# Load dataset
dataset = load_dataset("json", data_files={"train": DATASET_TRAIN_PATH, "test": DATASET_TEST_PATH})

# Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens(["Action:"], special_tokens=True)

# Tokenization function
def tokenize_function(example):
    encoding = tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=256, padding_side="right"
    )
    encoding["labels"] = encoding["input_ids"]  # Use input as labels for loss computation
    return encoding

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=4)

# BitsAndBytesConfig for efficient training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype="float16", 
    bnb_4bit_use_double_quant=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(device)

# LoRA Configuration
lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.15, 
    target_modules=["q_proj", "k_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.resize_token_embeddings(len(tokenizer))

# Function to get the latest checkpoint
def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))  # Sort by step number
    return os.path.join(output_dir, checkpoints[-1])

# Optuna Objective Function
def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

    # Set unique output directory for each trial
    trial_output_dir = f"{OUTPUT_DIR}/trial_{trial.number}"
    os.makedirs(trial_output_dir, exist_ok=True)

    # Check for latest checkpoint
    latest_checkpoint = get_latest_checkpoint(trial_output_dir)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=trial_output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=EPOCHS,
        learning_rate=learning_rate,
        logging_steps=LOGGING_STEPS,
        save_steps=100,
        save_total_limit=2,
        gradient_accumulation_steps=1,
        run_name=f"trial_{trial.number}",
        report_to="tensorboard",
        logging_dir=f"./logs/{OUTPUT_DIR.split('/')[-1]}_trial_{trial.number}",
        #load_best_model_at_end=True,  # Optionally load best model at the end
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Resume training if a checkpoint exists
    if latest_checkpoint:
        print(f"Resuming trial {trial.number} from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print(f"Starting fresh training for trial {trial.number}...")
        trainer.train()

    # Evaluate model performance
    eval_results = trainer.evaluate()
    loss = eval_results["eval_loss"]

    # Report intermediate results to allow pruning
    trial.report(loss, step=LOGGING_STEPS)  # Report the loss at every LOGGING_STEPS interval

    # Prune the trial if needed
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()  # Prune trial if it doesn't meet expectations

    # Save model for this trial
    model.save_pretrained(trial_output_dir + "/model/")
    tokenizer.save_pretrained(trial_output_dir + "/model/")

    # Merge and save full model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(trial_output_dir + "/full_model/")
    tokenizer.save_pretrained(trial_output_dir + "/full_model/tokenizer/")

    return loss

# Optuna Pruner: Use MedianPruner for pruning trials with worse-than-median results
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=200, interval_steps=1)

# Run Optuna Hyperparameter Search with pruning enabled
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=10)  # Run 10 trials

# Save Best Hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")
with open(f"{OUTPUT_DIR}/best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)
