import datetime
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
MODEL_PATH = "models/Llama-3.1-8B-Instruct"
DATASET_PATH = "datasets/mcv/1k_minecraft_villager_dataset_formatted_6-llama.jsonl"
OUTPUT_DIR = "./finetuned/41_results"
ACTION_TOKENS = ["<<speak>>", "<<wave>>", "<<offer>>", "<<grumble>>", "<<ignore>>", "<<give>>"]
ITEM_TOKENS = ["<item>", "<price>", "<currency>"]

def load_conversations(path: str) -> Dataset:
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            conv_chain = json.loads(line)
            conversations.append({"conversation": conv_chain})
    return Dataset.from_list(conversations).train_test_split(test_size=0.1)

# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add tokens as regular tokens (not special tokens)
num_new_tokens = tokenizer.add_tokens(ACTION_TOKENS + ITEM_TOKENS)
model.resize_token_embeddings(len(tokenizer))

# Initialize new embeddings with small random values
if num_new_tokens > 0:
    with torch.no_grad():
        embedding_size = model.get_input_embeddings().weight.shape[-1]
        new_embeddings = torch.randn(num_new_tokens, embedding_size) * 0.02
        model.get_input_embeddings().weight[-num_new_tokens:] = new_embeddings.to(
            model.get_input_embeddings().weight.dtype
        )
        # Tie output embeddings if needed
        if model.get_output_embeddings() is not None:
            model.get_output_embeddings().weight[-num_new_tokens:] = new_embeddings.to(
                model.get_output_embeddings().weight.dtype
            )

model = prepare_model_for_kbit_training(model)

# Optimized LoRA Configuration
peft_config = LoraConfig(
    r=24,
    lora_alpha=48,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Dataset processing
dataset = load_conversations(DATASET_PATH)

def format_conversation(example):
    messages = []
    for turn in example["conversation"]:
        # Clean action tokens from accidental spaces
        user_content = turn["user"].replace("<< ", "<<").replace(" >>", ">>")
        assistant_content = turn["assistant"].replace("<< ", "<<").replace(" >>", ">>")
        
        messages.extend([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ])
    
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    }

# Process datasets
train_dataset = dataset["train"].map(format_conversation, remove_columns=["conversation"])
eval_dataset = dataset["test"].map(format_conversation, remove_columns=["conversation"])

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=40,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=12,
    learning_rate=3e-4,
    bf16=True,
    max_grad_norm=0.2,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    weight_decay=0.05,
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none",
)

# Custom Data Collator
class DialogueDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        # Mask padding tokens for loss calculation
        batch["labels"] = torch.where(
            batch["input_ids"] == self.tokenizer.pad_token_id,
            -100,
            batch["input_ids"]
        )
        return batch

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    dataset_text_field="text",
    data_collator=DialogueDataCollator(
        tokenizer=tokenizer,
        mlm=False
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    tokenizer=tokenizer,
)

# Training
print(f"\n\n------ Training Started ------\nTime: {datetime.datetime.now()}\n")
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)