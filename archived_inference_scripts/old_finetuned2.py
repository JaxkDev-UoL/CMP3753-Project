import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path model directory
model_path = "finetuned/5_llama_3_1_8b_instruct_abcd/full_model"  


tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

tokenizer.pad_token = tokenizer.eos_token
#tokenizer.eos_token = None  # Ensure it's not triggering early stop

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to(device)

model.resize_token_embeddings(len(tokenizer))

print(f"Model embedding size: {model.config.vocab_size}")


# Move model to evaluation  
model.eval()

def generate_response(prompt, max_length=512, temperature=0.2, top_k=150, top_p=0.90):

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    print(f"Input tokens: {inputs['input_ids']}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            #max_length=max_length,
            #temperature=temperature,
            #top_k=top_k,
            #top_p=top_p,
            do_sample=False,
            #logits_processor=logits_processor
        )

    return tokenizer.decode(output[0], skip_special_tokens=False)

#tokenizer.pad_token = tokenizer.eos_token

if not tokenizer.decode(tokenizer.encode("<<wave>>")[1]) == "<<wave>>":  # True
   print("Token doesnt match")
   #raise ValueError("Tokenizer is not correctly configured for action token. Please ensure that the tokenizer is correctly loaded.")
else:
   print(f"<<wave>> = {tokenizer.encode('<<wave>>')[1]}")


prompt = """<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>
You have a friendly personality.

<|start_header_id|>user<|end_header_id|>
<<speak>> How's life? buddy!

<|start_header_id|>assistant<|end_header_id|>
<<speak>> Hello!!

<|start_header_id|>user<|end_header_id|>
<<speak>> What's the cost for that piece?

<|start_header_id|>assistant<|end_header_id|>
<<offer>> <item> Potion of Healing <price> 31 <currency> Coins

<|start_header_id|>assistant<|end_header_id|>
<<speak>> A special deal just for you! Potion of Healing is 31 Coins.

<|start_header_id|>user<|end_header_id|>
<<speak>> Alright, I'll take it.

<|start_header_id|>user<|end_header_id|>
<<give>> <item> 31 Coins

<|start_header_id|>assistant<|end_header_id|>
"""

prompt = """<|begin_of_text|>

<|start_header_id|>user<|end_header_id|>
Hello!<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
Hello! How can I help you today?<|eot_id|>

<|start_header_id|>user<|end_header_id|>
I recently signed up for a subscription but it looks like you guys charged me twice for it.<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
response = generate_response(prompt)
print("Prompt:", prompt)
print("\nGenerated Response:\n", response)

