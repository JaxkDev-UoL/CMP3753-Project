import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "finetuned/13_llama_3_1_8b_instruct/model"

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
#model.resize_token_embeddings(len(tokenizer))

model.eval()
print("Model loaded successfully.")

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

print(f"Model embedding size: {model.config.vocab_size}")

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
            do_sample=True,
            #logits_processor=logits_processor
        )

    print(output)

    return tokenizer.decode(output[0], skip_special_tokens=False)

#tokenizer.pad_token = tokenizer.eos_token

if not tokenizer.decode(tokenizer.encode("<<wave>>")[1]) == "<<wave>>":  # True
   raise ValueError("Tokenizer is not correctly configured for action token. Please ensure that the tokenizer is correctly loaded.")
else:
   print(f"<<wave>> = {tokenizer.encode('<<wave>>')[1]}")


prompt = """<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
<<wave>>
<|start_header_id|>assistant<|end_header_id|>
"""

response = generate_response(prompt)
print("Prompt:", prompt)
print("\n---------------\nGenerated Response:\n", response)
