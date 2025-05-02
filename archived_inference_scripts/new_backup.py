import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

id = 0
id = input("Enter the ID of the finetuned model you want to run: ")

MODEL_PATH = "./finetuned/"+id+"_results"
BASE_MODEL = "models/Llama-3.1-8B-Instruct"

def load_model():
    # Load tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config matching training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
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
    model.eval()
    
    # Structure matches training format EXACTLY
    # messages = [
    #     {"role": "user", "content": f"<<speak>> {prompt}"},
    #     #{"role": "assistant", "content": ""}  # Empty content to trigger response
    # ]
    
    # Apply same template as training
    # formatted_prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     #truncation=True,
    #     #add_generation_prompt=True,

    #     date_string=datetime.datetime.now().strftime("%d-%m-%Y"),
    # )
    #formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"  # Add assistant header for generation
    formatted_prompt = "<|user|>\n<<speak>> Hello, what have you got for me today?<|endoftext|>\n<|assistant|>\n"

    # save formatted_prompt to file for debugging
    with open("formatted_prompt.txt", "w") as f:
        f.write(formatted_prompt)
    
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True
    ).to(model.device)

    # save torch tensor inputs to file for debugging
    torch.save(inputs, "inputs.pt")

    # Conservative generation parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,#max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria = [
                lambda input_ids, scores: input_ids[0][-1] == tokenizer.convert_tokens_to_ids("<|assistant|>")
            ]
        )

    torch.save(outputs[0], "outputs.pt")
    print("outputs = ", outputs)

    # Decode and clean using training patterns
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
    print("full_response = ", full_response)

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()

    #with open("full_response.txt", "w") as f:
    #    f.write(full_response)

    print("full_response = ", full_response)
    
    # Extract only the assistant's response after last <|eot_id|>
    response_start = full_response.rfind("<assistant>\n\n") + len("<assistant>\n\n")
    clean_response = full_response[response_start:]
    clean_response = clean_response.split("<|eot_id|>")[0].strip()
    
    return clean_response

if __name__ == "__main__":
    test_prompt = "Hello, what have you got for me today?"
    print(f"\n=== Response to '{test_prompt}' ===")
    #print(generate_response(test_prompt))
    generate_response(test_prompt)

