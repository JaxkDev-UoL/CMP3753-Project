import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import json
from datasets import Dataset

# Configuration
MODEL_PATH = "models/Llama-3.1-8B-Instruct"
DATASET_PATH = "datasets/mcv/100_minecraft_villager_dataset_formatted_6-llama.jsonl"
OUTPUT_DIR = "./finetuned/47_results"
SPECIAL_TOKENS = ["<<speak>>", "<<wave>>", "<<offer>>", "<<grumble>>", "<<ignore>>", "<<give>>"]

def load_conversations(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

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
tokenizer.add_special_tokens({
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>" 
})
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

# Add special tokens and resize
tokenizer.add_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer))

with torch.no_grad():
    new_embeddings = model.get_input_embeddings().weight.data
    old_emb_size = new_embeddings.shape[0] - len(SPECIAL_TOKENS)
    new_embeddings[old_emb_size:] = new_embeddings[:old_emb_size].mean(dim=0)

    new_embeddings2 = model.get_output_embeddings().weight.data
    old_emb_size2 = new_embeddings2.shape[0] - len(SPECIAL_TOKENS)
    new_embeddings2[old_emb_size2:] = new_embeddings2[:old_emb_size2].mean(dim=0)

model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load and format dataset
raw_data = load_conversations(DATASET_PATH)

def format_conversation(conv_chain):
    formatted = []
    for turn in conv_chain:
        # Clean user input
        user = turn["user"].replace("  ", " ").strip()
        # Clean assistant response
        assistant = turn["assistant"].split("<|endoftext|>")[0].strip()
        
        formatted.append(
            f"<|user|>\n{user}<|endoftext|>\n"
            f"<|assistant|>\n{assistant}<|endoftext|>\n"
        )
    return "".join(formatted)

# Create properly structured dataset
formatted_texts = [format_conversation(conv) for conv in raw_data]
train_dataset = Dataset.from_dict({"text": formatted_texts})

# Fixed Data Collator
class ConversationCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Tokenize batch with proper padding
        batch = self.tokenizer(
            [f["text"] for f in features],
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["input_ids"] == self.tokenizer.pad_token_id] = -100

        return batch

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=32,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    #optim="adamw_torch",
    save_steps=100,
    logging_steps=5,
    learning_rate=5e-4,
    #fp16=False,
    bf16=True,
    #max_grad_norm=0.5,
    warmup_ratio=0.10,
    lr_scheduler_type="cosine",
    report_to="none",
    weight_decay=0.01,
    gradient_checkpointing=True,  # Enable gradient checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Fix warning
    include_tokens_per_second=True,  # Include tokens per second in logs
    include_num_input_tokens_seen=True,  # Include number of input tokens seen in logs-
    
    remove_unused_columns=False,  # Critical fix
    # eval_steps=20,
    # eval_strategy="steps",
    # eval_accumulation_steps=4,
)

# Initialize trainer with updated settings
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=4096,
    dataset_text_field="text",
    data_collator=ConversationCollator(
        tokenizer=tokenizer,
        mlm=False
    ),
    packing=False,
    tokenizer=tokenizer,  # Pass tokenizer to the trainer
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)