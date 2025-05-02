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
DATASET_PATH = "datasets/mcv/1k_minecraft_villager_dataset_formatted_6-llama.jsonl"
OUTPUT_DIR = "./finetuned/44_results"
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
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add special tokens and resize
tokenizer.add_tokens(SPECIAL_TOKENS)
model.resize_token_embeddings(len(tokenizer))

# Initialize embeddings
with torch.no_grad():
    new_embeddings = model.get_input_embeddings().weight.data
    old_emb_size = new_embeddings.shape[0] - len(SPECIAL_TOKENS)
    new_embeddings[old_emb_size:] = new_embeddings[:old_emb_size].mean(dim=0)

model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load and format dataset
raw_data = load_conversations(DATASET_PATH)

def format_conversation(conv_chain):
    text = ""
    for turn in conv_chain:
        text += f"<|user|>{turn['user']}<|endoftext|>\n<|assistant|>{turn['assistant']}<|endoftext|>\n"
    return text

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
        return batch

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=64,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=5e-4,
    bf16=True,
    max_grad_norm=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    save_steps=500,
    logging_steps=50,
    remove_unused_columns=False,  # Critical fix
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    dataset_text_field="text",
    data_collator=ConversationCollator(
        tokenizer=tokenizer,
        mlm=False
    ),
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)