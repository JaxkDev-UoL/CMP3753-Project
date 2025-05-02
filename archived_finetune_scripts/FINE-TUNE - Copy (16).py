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
from torch.nn.utils import clip_grad_norm_

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configuration
MODEL_PATH = "models/Llama-3.1-8B"
DATASET_PATH = "datasets/mcv/10k_minecraft_villager_dataset_formatted_6-llama.jsonl"
OUTPUT_DIR = "./finetuned/16_results"
SPECIAL_TOKENS = ["<<speak>>"]#, "<<wave>>", "<<offer>>", "<<grumble>>", "<<ignore>>", "<<give>>"]

def load_conversations(path: str) -> Dataset:
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            conv_chain = json.loads(line)
            conversations.append({"conversation": conv_chain})
    return Dataset.from_list(conversations)


if __name__ == "__main__":

    # QLoRA Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
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

    # Add special tokens as regular tokens
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        print(f"Added {len(new_tokens)} new tokens: {new_tokens}")

    # Verify tokenization
    test_text = "<<speak>> Hello"
    print("\nTokenization test:")
    print("Original:", test_text)
    print("Tokenized:", tokenizer.tokenize(test_text))
    print("Token IDs:", tokenizer.encode(test_text))

    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    # Chat template
    tokenizer.chat_template = """{% for message in messages %}
    <|start_header_id|>{{ message['role'] }}<|end_header_id|>

    {{ message['content'] }}<|eot_id|>{% endfor %}
    """

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.10,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Dataset processing
    def format_conversation(example):
        messages = []
        for turn in example["conversation"]:
            # Add space AFTER special tokens
            messages.append({
                "role": "user", 
                "content": "<<speak>>" #turn['user'].strip()
            })
            messages.append({
                "role": "assistant",
                "content": "<<speak>>" #turn['assistant'].strip()
            })
        
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        }

    dataset = load_conversations(DATASET_PATH).map(format_conversation, remove_columns=["conversation"])

    # Training setup
    # Modified Training Arguments with stable settings
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Reduced from 4
        gradient_accumulation_steps=8,  # Reduced from 8
        optim="adamw_torch",
        save_steps=50,
        logging_steps=2,
        learning_rate=1e-5,  # Reduced learning rate
        bf16=True,
        max_grad_norm=1.0,  # Increased gradient norm
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},  # Changed to True
        dataloader_num_workers=0,  # Force disable multiprocessing
        remove_unused_columns=True,
    )

    # Windows-specific multiprocessing fix
    torch.multiprocessing.freeze_support()
    
    print(f"\n\n ------ Training ------ \nTime: {datetime.datetime.now()}\n")
    
    
    # Modified SFTTrainer configuration
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=64,
        dataset_text_field="text",
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Helps with GPU memory alignment
        ),
        packing=False,  # Disable packing to isolate issues
        #neftune_noise_alpha=5,  # Adds noise to prevent NaN
        tokenizer=tokenizer,  # Explicit tokenizer passing
        dataset_num_proc=1,  # Disable dataset preprocessing parallelism
    )

    torch.autograd.set_detect_anomaly(True)
    # Inside custom training loop (if using one)
    clip_grad_norm_(model.parameters(), 1.0)
    
    # Train and save
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)