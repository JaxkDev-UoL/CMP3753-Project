import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

MODEL_PATH = "./finetuned/26_results"
BASE_MODEL = "models/Llama-3.1-8B-Instruct"

def load_model():
    # Load tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config matching training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        #bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model and resize embeddings
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.config.use_cache = False
    return model.to('cuda'), tokenizer

def generate_response(prompt, max_new_tokens=150):
    model, tokenizer = load_model()
    #model.eval()
    
    # Structure matches training format EXACTLY
    messages = [
        {"role": "user", "content": f"<<speak>> {prompt}"},
        #{"role": "assistant", "content": ""}  # Empty content to trigger response
    ]
    
    # Apply same template as training
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        #truncation=True,
        #add_generation_prompt=True,

        date_string=datetime.datetime.now().strftime("%d-%m-%Y"),
    )
    #formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"  # Add assistant header for generation

    # save formatted_prompt to file for debugging
    with open("formatted_prompt.txt", "w") as f:
        f.write(formatted_prompt)
    
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=128,
        padding=False,#"max_length",
        truncation=True
    ).to(model.device)

    # save torch tensor inputs to file for debugging
    torch.save(inputs, "inputs.pt")

    # Conservative generation parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.75,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=6,
        )

    torch.save(outputs[0], "outputs.pt")
    print("outputs = ", outputs)

    # Decode and clean using training patterns
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    #with open("full_response.txt", "w") as f:
    #    f.write(full_response)

    print("full_response = ", full_response)
    
    # Extract only the assistant's response after last <|eot_id|>
    response_start = full_response.rfind("<assistant>\n\n") + len("<assistant>\n\n")
    clean_response = full_response[response_start:]
    clean_response = clean_response.split("<|eot_id|>")[0].strip()
    
    return clean_response

if __name__ == "__main__":
    test_prompt = "Hello, got anything?"
    print(f"\n=== Response to '{test_prompt}' ===")
    #print(generate_response(test_prompt))
    generate_response(test_prompt)

# NOTED RESPONSE:

# outputs =  tensor([[128000, 128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,
#            2696,     25,   6790,    220,   2366,     18,    198,  15724,   2696,
#              25,    220,   2589,     12,   2371,     12,   2366,     20,    271,
#          128009, 128006,    882, 128007,    271, 128256,  22691,     11,   2751,
#            4205,     30, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
#          128009, 128009, 128006,  78191, 128007,    271, 128258,    366,   1224,
#              29,   7573,    763,  23196,    366,   6692,     29,    220,   1544,
#             366,  16353,     29,  62876,    198, 128257,  28653,   1070,     11,
#            4333,      0,    358,   3077,   2751,   7573,    763,  23196,    369,
#            1120,    220,   1544,  62876,      0, 128009]], device='cuda:0')
# full_response =  system

# Cutting Knowledge Date: December 2023
# Today Date: 07-04-2025

# user

# <<speak>> Hello, got anything?assistant

# <<offer>> <item> Gold Ingot <price> 27 <currency> Coins
# <<wave>> Hey there, friend! I've got Gold Ingot for just 27 Coins!