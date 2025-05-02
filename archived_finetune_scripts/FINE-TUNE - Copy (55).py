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

# 49 = based off 26.

# Configuration
MODEL_PATH = "models/Llama-3.1-8B-Instruct"
DATASET_PATH = "datasets/mcv/1k_minecraft_villager_dataset_formatted_6-llama.jsonl"
OUTPUT_DIR = "./finetuned/55_results"
SPECIAL_TOKENS = ["<<speak>>", "<<wave>>", "<<offer>>", "<<grumble>>", "<<ignore>>", "<<give>>"]

def load_conversations(path: str) -> Dataset:
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Each line contains one complete conversation chain
            conv_chain = json.loads(line)
            conversations.append({
                "conversation": conv_chain  # Preserve full conversation structure
            })
    return Dataset.from_list(conversations)

# QLoRA Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model with trust_remote_code
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer and configure padding
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add special tokens and configure BOS/EOS
tokenizer.add_tokens(SPECIAL_TOKENS)
tokenizer.add_bos_token = True  # Add BOS token automatically
tokenizer.add_eos_token = True  # Add EOS token automatically
model.resize_token_embeddings(len(tokenizer))

# And add after model resize:
# Reinitialize the new embeddings properly
with torch.no_grad():
    new_embeddings = model.get_input_embeddings().weight.data
    old_emb_size = new_embeddings.shape[0] - len(SPECIAL_TOKENS)
    new_embeddings[old_emb_size:] = new_embeddings[:old_emb_size].mean(dim=0)

    new_embeddings2 = model.get_output_embeddings().weight.data
    old_emb_size2 = new_embeddings2.shape[0] - len(SPECIAL_TOKENS)
    new_embeddings2[old_emb_size2:] = new_embeddings2[:old_emb_size2].mean(dim=0)

model = prepare_model_for_kbit_training(model)

# Official Llama 3.1 chat template (should include BOS/EOS via tokenizer settings)
# tokenizer.chat_template = """{% for message in messages %}
# <|start_header_id|>{{ message['role'] }}<|end_header_id|>

# {{ message['content'] }}<|eot_id|>{% endfor %}
# """

# LoRA Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.10,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load dataset with complete conversations
dataset = load_conversations(DATASET_PATH)

# Formatting function for full conversation chains
def format_conversation(example):
    messages = []
    for turn in example["conversation"]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    }

# Apply formatting while preserving conversation structure
dataset = dataset.map(
    format_conversation,
    remove_columns=["conversation"]
)

# Modified Data Collator
class ConversationDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        
        
        input_ids = []
        labels = []
        for seq in batch["input_ids"]:
            input_ids.append(seq.clone().detach())  
            labels.append(seq.clone().detach())  
            
        # Pad sequences to equal length
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )
        
        return batch
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

# Training Arguments with gradient checkpointing
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=12,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    #optim="adamw_torch",
    save_steps=100,
    logging_steps=2,
    learning_rate=1e-4,
    fp16=False,
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
)

# Initialize trainer with updated settings
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=4096,
    dataset_text_field="text",
    data_collator=ConversationDataCollator(
        tokenizer=tokenizer,
        mlm=False
    ),
    packing=False,
    tokenizer=tokenizer,  # Pass tokenizer to the trainer
)

print(f"\n\n ------ Training ------ \nTime: {datetime.datetime.now()}\n")

# Train and save
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)



# NOTED RESPONSE (INCLUDES TRUE NEW TOKEN, ID=128257=<<wave>>)
# system

# Cutting Knowledge Date: December 2023
# Today Date: 07-04-2025

# user

# <<speak>> Hello, got anything?assistant

# <<wave>> Hey, how's the world treating you? my friend, how are you?
