import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path model directory
model_path = "finetuned/9_llama_3_1_8b_instruct/full_model"  

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Base model path (the model you originally finetuned)
base_model_path = "models/Llama-3.1-8B-Instruct"
checkpoint_path = "finetuned/9_llama_3_1_8b_instruct/checkpoint-280"

# Load base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("finetuned/8_llama_3_1_8b_instruct/full_model")
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16).to(device)
model.resize_token_embeddings(len(tokenizer))

# Load LoRA adapter and merge
model = PeftModel.from_pretrained(model, checkpoint_path)
model = model.merge_and_unload()  # Merge LoRA into the model for pure inference

model.eval()
print("Model loaded successfully with LoRA checkpoint.")


#tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.eos_token = None  # Ensure it's not triggering early stop

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
            do_sample=True,
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
You have a generous personality.

<|start_header_id|>user<|end_header_id|>
<<speak>> Ah, it's been too long! mate!<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
<<grumble>>

<|start_header_id|>assistant<|end_header_id|>
<<speak>> Nice to meet you mate!<|eot_id|>

<|start_header_id|>user<|end_header_id|>
<<speak>> What's available for trade?<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
<<offer>> <item> Podzol <price> 25 <currency> Coins

<|start_header_id|>assistant<|end_header_id|>
<<speak>> """

response = generate_response(prompt)
print("Prompt:", prompt)
print("\nGenerated Response:\n", response)

test = """
# Do the token replacement
    params = model.state_dict()
    #print(params.keys())
    param_name = 'transformer.wte.weight' if 'transformer.wte.weight' in params else 'bert.embeddings.word_embeddings.weight'
    if data_args.new_token_method in {'avg_emb_add'}:
      num_inits = 1 if not data_args.do_wikitext_type_addition else 100
      original_embs = params[param_name][:-num_inits,:]
      mean = torch.mean(original_embs, dim=0)
      samples = params[param_name][-num_inits:,:]
      sample_norms = torch.norm(samples, dim=1)

      #params[param_name][-num_inits:,:] = (samples * torch.norm(mean) / sample_mean / torch.sqrt(torch.tensor(float(mean.size()[0])))) + mean
      params[param_name][-num_inits:,:] = (samples.t() / sample_norms).t() * torch.norm(mean) * 1e-8 + mean

      #params[param_name][-num_inits:,:] = (params[param_name][-num_inits:,:] * torch.norm(mean) / torch.mean(params[param_name][-num_inits:,:], dim=1) / torch.sqrt(params.size()[1])) + mean
      #params[param_name][-num_inits:,:] = params[param_name][-num_inits:,:] * torch.norm(mean) + mean #/ torch.mean(params[param_name][-num_inits:,:], dim=1) / torch.sqrt(params.size()[1])) + mean
      model.load_state_dict(params)
"""