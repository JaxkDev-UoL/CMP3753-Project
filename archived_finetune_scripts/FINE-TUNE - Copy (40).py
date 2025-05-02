import datetime
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
MODEL_PATH = "models/Llama-3.1-8B-Instruct"
DATASET_PATH = "datasets/mcv/1k_minecraft_villager_dataset_formatted_6-llama.jsonl"
OUTPUT_DIR = "./finetuned/40_results"
SPECIAL_TOKENS = [
    "<<speak>>", "<<wave>>", "<<offer>>", "<<grumble>>", "<<ignore>>", "<<give>>",
    "<item>", "<price>", "<currency>"
]  # Added XML-style tags

def load_conversations(path: str) -> Dataset:
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            conv_chain = json.loads(line)
            conversations.append({"conversation": conv_chain})
    return Dataset.from_list(conversations)

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

# Add special tokens with proper initialization
num_new_tokens = tokenizer.add_tokens(SPECIAL_TOKENS)#, special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

# Initialize new embeddings with small random values
if num_new_tokens > 0:
    with torch.no_grad():
        embedding_size = model.get_input_embeddings().weight.shape[-1]
        new_embeddings = torch.randn(num_new_tokens, embedding_size) * 0.02
        model.get_input_embeddings().weight[-num_new_tokens:] = new_embeddings.to(
            model.get_input_embeddings().weight.dtype
        )

model = prepare_model_for_kbit_training(model)

# LoRA Configuration
peft_config = LoraConfig(
    r=32,  # Increased from 16 for better adaptation
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"#,
    #target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Load and format dataset
dataset = load_conversations(DATASET_PATH)

# Replace the existing dataset mapping and collator with:

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

class TextDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Extract text strings
        texts = [feature["text"] for feature in features]
        
        # Tokenize batch
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        
        # Create labels
        batch["labels"] = batch["input_ids"].clone()
        return batch

dataset = dataset.map(
    format_conversation,
    remove_columns=["conversation"]
)

# Training Arguments with improved settings
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=20,  # Reduced from 64 to prevent overfitting
    per_device_train_batch_size=4,  # Reduced for stability
    gradient_accumulation_steps=16,
    learning_rate=2e-5,  # Lower learning rate for better stability
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    report_to="none",
    weight_decay=0.01,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    remove_unused_columns=False  # Important for preserving data structure
)

# Initialize trainer with updated collator
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    dataset_text_field="text",
    data_collator=TextDataCollator(
        tokenizer=tokenizer,
        mlm=False
    ),
    packing=False,
    tokenizer=tokenizer,
)

print(f"\n\n------ Training Started ------ \nTime: {datetime.datetime.now()}\n")

# Train and save
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)