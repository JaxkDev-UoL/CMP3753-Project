import json
import os
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)

# Configuration
MODEL_PATH = "models/Llama-3.1-8B-Instruct"
DATASET_PATH = "datasets/abcd/abcd_v1.1_processed.jsonl"
BASE_OUTPUT_DIR = "./finetuned/67_results_abcd"
SPECIAL_TOKENS = json.load(open('datasets/abcd/abcd_v1.1_tokens.json', 'r'))

def load_conversations(path: str) -> Dataset:
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            conv_chain = json.loads(line)
            conversations.append({"conversation": conv_chain})
    return Dataset.from_list(conversations[:500])

def format_conversation(examples, tokenizer): 
    all_texts = []
    for conv in examples["conversation"]:
        messages = []
        for turn in conv:
            messages.append({"role": "user", "content": turn["user"].strip()})
            messages.append({"role": "assistant", "content": turn["assistant"].strip()})
        all_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    
    tokenized = tokenizer(
        all_texts,
        padding=False,
        truncation=True,
        max_length=4096,
        return_tensors=None,
        add_special_tokens=True
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True
    ).cpu()

    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(SPECIAL_TOKENS)
    
    # Manual embedding initialization on CPU
    with torch.no_grad():
        old_emb_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        
        input_embeddings = model.get_input_embeddings().weight
        output_embeddings = model.get_output_embeddings().weight
        
        # Calculate statistics from existing embeddings
        mean = input_embeddings[:old_emb_size].mean(dim=0)
        std = input_embeddings[:old_emb_size].std(dim=0)
        
        # Initialize new tokens
        num_special = len(SPECIAL_TOKENS)
        input_embeddings[-num_special:] = torch.normal(
            mean.repeat(num_special, 1),
            std.repeat(num_special, 1)
        )
        output_embeddings[-num_special:] = torch.normal(
            mean.repeat(num_special, 1),
            std.repeat(num_special, 1)
        )
    
    return model.to('cuda' if torch.cuda.is_available() else 'cpu'), tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()

    # Dataset preparation with splits
    if not all(os.path.exists(f"{BASE_OUTPUT_DIR}/{split}") for split in ["train", "val", "test"]):
        full_dataset = load_conversations(DATASET_PATH)
        
        # First split: 80% train, 20% temp
        train_test = full_dataset.train_test_split(test_size=0.2, seed=42)
        # Second split: 10% val, 10% test from temp
        val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
        
        splits = {
            'train': train_test['train'],
            'val': val_test['train'],
            'test': val_test['test']
        }

        def format_and_save(split_name, dataset):
            formatted = dataset.map(
                lambda ex: format_conversation(ex, tokenizer),
                batched=True,
                batch_size=32,
                remove_columns=["conversation"],
                num_proc=1
            )
            formatted.save_to_disk(f"{BASE_OUTPUT_DIR}/{split_name}")

        for name, split in splits.items():
            format_and_save(name, split)
        print("Created and saved dataset splits")
        train_dataset = splits['train']
        eval_dataset = splits['val']
        test_dataset = splits['test']
    else:
        train_dataset = Dataset.load_from_disk(f"{BASE_OUTPUT_DIR}/train")
        eval_dataset = Dataset.load_from_disk(f"{BASE_OUTPUT_DIR}/val")
        test_dataset = Dataset.load_from_disk(f"{BASE_OUTPUT_DIR}/test")
        print("Loaded existing dataset splits")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=BASE_OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=25,
        logging_steps=5,
        save_steps=100,
        learning_rate=5e-5,
        weight_decay=0.1,
        bf16=True,
        #load_best_model_at_end=True,
        #metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        optim="adafactor",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",
        max_grad_norm=0.5,
        include_num_input_tokens_seen=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    
    # Final evaluation on test set
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest set results: {test_results}")
    
    trainer.save_model(BASE_OUTPUT_DIR)
    tokenizer.save_pretrained(BASE_OUTPUT_DIR)
    print("\n--- Training Complete ---")