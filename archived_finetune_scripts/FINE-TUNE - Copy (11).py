import os
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from logging_utils import LoggingCallback

# Configuration
ID = "11"
DATASET_ID = "6"
MODEL_NAME = "models/Llama-3.1-8B-Instruct"  # Path to local model
DATASET_PATH = f"./datasets/mcv/1k_minecraft_villager_dataset_formatted_{DATASET_ID}-llama.jsonl"  # Path to local dataset
TOKENS_PATH = f"./datasets/mcv/1k_minecraft_villager_dataset_tokens_{DATASET_ID}-llama.json"  # Path to the tokens json
OUTPUT_DIR = f"./finetuned/{ID}_llama_3_1_8b_instruct"  # Path to save the fine-tuned model
EPOCHS = 6  # Increased number of epochs for better fine-tuning
BATCH_SIZE = 2  # Adjusted batch size to fit the GPU (RTX 4070 Ti Super)
LEARNING_RATE = 5e-5
LOGGING_STEPS = 10

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    return tokenizer

def format_conversation(conversation):
    formatted_text = "<|begin_of_text|>\n"
    for turn in conversation:
        user_text = turn['user']
        assistant_text = turn['assistant']
        formatted_text += (
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{user_text}\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{assistant_text}\n"
            f"<|eot_id|>\n"
        )
    return formatted_text

# Tokenization function
def tokenize_function(example, tokenizer):
    # Get the raw conversation text (entire block of conversation)
    # print(example)
    conversation = json.loads(example['text'])  # Entire conversation as one string

    conversation = format_conversation(conversation)
    
    tokenized = tokenizer(
        conversation,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt',  # Ensure PyTorch tensors are returned
        add_special_tokens=False,  # Special tokens are already added
    )
    # Add labels for causal language modeling
    tokenized['labels'] = tokenized['input_ids'].clone()
    tokenized['input_ids'] = tokenized['input_ids'][:, :-1]  # Remove last token
    tokenized['labels'] = tokenized['labels'][:, 1:]  # Remove first token

    tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

    # Verify labels alignment
    for i in range(len(tokenized['input_ids']) - 1):
        assert tokenized['labels'][i] == tokenized['input_ids'][i + 1], "Labels are not correctly aligned with input_ids"

    if tokenized['input_ids'][-1] != tokenized['input_ids'][-2] and tokenized['input_ids'][-2] != tokenized['input_ids'][-3]:
        print("WARNING: Last 3 tokens are not the same, Is max_length too small??")

    return tokenized

# Tokenize the entire dataset
def tokenize_dataset(dataset, tokenizer, batch_size=4, num_proc=16):
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=False,  # Process one example at a time
        num_proc=num_proc,
        remove_columns=dataset.column_names,  # Remove original columns to save memory
    )

    return tokenized_dataset

# Setup model and apply LoRA for fine-tuning
def setup_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.to(device)

    # Freeze all layers except the first few (e.g., the first 6 layers)
    for name, param in model.named_parameters():
        if 'model.layers.' in name:
            layer_num = int(name.split('.')[2])  # Extract layer number
            if layer_num >= 12:  # Freeze layers after the 6th layer
                param.requires_grad = False
                print(f"Freezing layer {layer_num}")

    return model, device

# Main function for training
def train_model():
    torch.cuda.empty_cache()
    writer = setup_tensorboard_log()
    dataset = load_data()
    special_tokens = load_special_tokens()
    tokenizer = setup_tokenizer(special_tokens)

    tokenized_datasets = tokenize_dataset(dataset, tokenizer, batch_size=BATCH_SIZE, num_proc=8)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    total_tokens = 0
    # count total tokens in the dataset (input_ids) using least memory
    for example in tokenized_datasets:
       total_tokens += len(example["input_ids"])

    print(f"Total tokens in the dataset: {total_tokens}")

    model, device = setup_model()
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    print(tokenized_datasets['input_ids'])

    # Check for invalid token IDs
    # invalid_token_ids = [idx for idx in tokenized_datasets['input_ids'].flatten() if idx >= tokenizer.vocab_size]
    # if invalid_token_ids:
    #     print(f"Invalid token IDs found: {len(invalid_token_ids)}")


    #print(tokenized_datasets[0]['input_ids'].shape)  # Should be (sequence_length,)
    #print(tokenized_datasets[0]['attention_mask'].shape)  # Should be (sequence_length,)
    #print(tokenized_datasets[0]['labels'].shape)  # Should be (sequence_length,)

    # # # Verify loss computation
    # sample = tokenized_datasets[0]
    # inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in sample.items() if k in ['input_ids', 'attention_mask', 'labels']}
    # outputs = model(**inputs)
    # print(outputs.loss)  # Should print a loss value

    # Split the dataset into train and test
    train_test_split = tokenized_datasets.train_test_split(test_size=0.1)  # 90% train, 10% test
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']


    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if 'model.layers.0.' in n],
            "lr": 5e-4,  # Higher learning rate for lower layers
        },
        {
            "params": [p for n, p in model.named_parameters() if 'model.layers.1.' in n],
            "lr": 5e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if 'model.layers.2.' in n],
            "lr": 5e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if 'model.layers.3.' in n],
            "lr": 5e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if 'model.layers.4.' in n],
            "lr": 5e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if 'model.layers.5.' in n],
            "lr": 5e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if 'model.layers.' in n and int(n.split('.')[2]) >= 6],
            "lr": 1e-5,  # Lower learning rate for higher layers
        },
        {
            "params": [p for n, p in model.named_parameters() if 'lm_head' in n or 'embed_tokens' in n],
            "lr": 5e-5,  # Higher learning rate for embeddings and output layer
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=100,
        save_total_limit=10,
        report_to="none",
        no_cuda=False,
        disable_tqdm=False,

        eval_strategy="steps",
        eval_steps=200,
        warmup_steps=200,
        weight_decay=0.01,
        bf16=True,
        max_grad_norm=1.0,  # Enable gradient clipping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        #optimizers=(optimizer, None), #, schedule), 
    )

    trainer.add_callback(LoggingCallback(writer, model, trainer))

    # Resume training if checkpoint exists
    dirs = os.listdir(OUTPUT_DIR)
    resume_from_checkpoint = any("checkpoint" in d for d in dirs)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save fine-tuned model
    model.save_pretrained(OUTPUT_DIR + "/model/")
    tokenizer.save_pretrained(OUTPUT_DIR + "/model/")

    writer.close()

    print("Fine-tuning complete. Model saved at", OUTPUT_DIR)

# Entry point
if __name__ == '__main__':
    train_model()
