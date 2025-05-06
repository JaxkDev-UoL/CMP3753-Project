import json
import os
import torch
from datasets import Dataset
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
BASE_OUTPUT_DIR = "./finetuned/63_results_abcd"
SPECIAL_TOKENS = json.load(open('datasets/abcd/abcd_v1.1_tokens.json', 'r'))

# ========== PHASE CONFIGURATION ==========
current_phase = int(input("Enter current phase: "))  # <<< CHANGE THIS
PHASE_LAYERS = {
    1: list(range(24, 32)),
    2: list(range(16, 24)),
    3: list(range(8, 16)),
    4: list(range(0, 8)),
    5: "full"  # Full model for final phase (test vram limits.)
}

PHASE_PARAMS = {
    1: {'batch_size': 2, 'accum': 4,  'lr': 2e-5, 'epochs': 1, 'wd': 0.01,  'max_grad': 0.5},
    2: {'batch_size': 2, 'accum': 4, 'lr': 1e-5, 'epochs': 2, 'wd': 0.005, 'max_grad': 0.4},
    3: {'batch_size': 2, 'accum': 4, 'lr': 5e-6, 'epochs': 2, 'wd': 0.002, 'max_grad': 0.3},
    4: {'batch_size': 1, 'accum': 8, 'lr': 2e-6, 'epochs': 4, 'wd': 0.001, 'max_grad': 0.2},
    5: {'batch_size': 1, 'accum': 16, 'lr': 1e-6, 'epochs': 1, 'wd': 0.0001,'max_grad': 0.1}
}
# =========================================

def load_conversations(path: str) -> Dataset:
    conversations = []
    max = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Each line contains one complete conversation [{},{},{},...]
            conv_chain = json.loads(line)
            conversations.append({
                "conversation": conv_chain
            })
            if len(conv_chain) > max:
                max = len(conv_chain)
    print(f"Max conversation length: {max}")

    return Dataset.from_list(conversations[0:1000])

def format_conversation(examples, tokenizer): 
    all_texts = []
    for conv in examples["conversation"]:
        messages = []
        for turn in conv:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        all_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    
    # Batch tokenization
    tokenized = tokenizer(
        all_texts,
        padding=False,
        truncation=True,
        max_length=2048,
        return_tensors=None,
        add_special_tokens=True
    )

    #print(tokenized["input_ids"][0][-5:])
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

def load_phase_model(phase: int):
    """Smart model loading with phase progression"""
    if phase == 1:
        # Load base model without device map first
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=None,  # Remove device_map
            trust_remote_code=True,
            low_cpu_mem_usage=False
        )
        #model = torch.compile(model)  # cant compile on windows yet
        model = model.cpu()  # Start on CPU for safe initialization
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Add special tokens
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

    else:
        # Load checkpoints with device map
        prev_phase_dir = f"{BASE_OUTPUT_DIR}_phase{phase-1}"
        model = AutoModelForCausalLM.from_pretrained(
            prev_phase_dir,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=False  # Required for device_map to work, otherwise false
        )
        tokenizer = AutoTokenizer.from_pretrained(prev_phase_dir)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model, tokenizer

if __name__ == "__main__":
    # Load model and tokenizer for the current phase
    model, tokenizer = load_phase_model(current_phase)

    # Freeze all layers initially except embedding layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    for param in model.get_output_embeddings().parameters():
        param.requires_grad = True

    # Layer unfreezing logic (which layers to train)
    if PHASE_LAYERS[current_phase] == "full":
        for param in model.parameters():
            param.requires_grad = True
    else:
        for layer_idx in PHASE_LAYERS[current_phase]:
            for param in model.model.layers[layer_idx].parameters():
                param.requires_grad = True

    if torch.cuda.is_available():
        model = model.cuda()
        torch.cuda.empty_cache()
    else:
        model = model.cpu()

    for param in model.parameters():
        if param.is_meta:
            param.to_empty(device='cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(f"{BASE_OUTPUT_DIR}_phase1/dataset"):
        dataset = Dataset.load_from_disk(f"{BASE_OUTPUT_DIR}_phase1/dataset")
        print(f"Dataset loaded from {BASE_OUTPUT_DIR}_phase1/dataset")
    else:
        # Define formatting with tokenizer access, kept having issues with num_proc > 1 on windows.
        def format_conv_with_tokenizer(examples):
            return format_conversation(examples, tokenizer)

        # Apply mapping
        dataset = load_conversations(DATASET_PATH).map(
            format_conv_with_tokenizer,
            batched=True,
            batch_size=32,
            remove_columns=["conversation"],
            num_proc=1
        )

        dataset.save_to_disk(f"{BASE_OUTPUT_DIR}_phase1/dataset")
        print(f"Dataset saved to {BASE_OUTPUT_DIR}_phase1/dataset")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Training setup
    params = PHASE_PARAMS[current_phase]
    
    training_args = TrainingArguments(
        output_dir=f"{BASE_OUTPUT_DIR}_phase{current_phase}",
        num_train_epochs=params['epochs'],
        per_device_train_batch_size=params['batch_size'],
        gradient_accumulation_steps=params['accum'],
        learning_rate=params['lr'],
        weight_decay=params['wd'],
        bf16=True,
        gradient_checkpointing=True,
        optim="adafactor",
        logging_steps=2,
        save_steps=500,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=params['max_grad'],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05 if current_phase == 1 else 0.02,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        group_by_length=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Training execution
    print(f"\n--- Phase {current_phase} Training ---")
    print(f"Training Layers: {PHASE_LAYERS[current_phase] if current_phase != 5 else 'All'}")
    print(f"Checkpoint Source: {'Base Model' if current_phase == 1 else f'Phase {current_phase-1}'}")
    print(f"Batch Size: {params['batch_size']} * {params['accum']} = {params['batch_size']*params['accum']}")

    trainer.train()
    trainer.save_model(f"{BASE_OUTPUT_DIR}_phase{current_phase}")
    tokenizer.save_pretrained(f"{BASE_OUTPUT_DIR}_phase{current_phase}")

    print(f"\n--- Phase {current_phase} Complete ---")
    print(f"Next: Set current_phase = {current_phase+1} and rerun\n")