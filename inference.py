import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

MODEL_PATH = "./finetuned/62_results_abcd"
BASE_MODEL = "models/Llama-3.1-8B-Instruct"

def load_model():
    # Load tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # TODO: Check what model is being loaded as very late ones don't use LORA/QLORA/B&B
    # Quantization config matching training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        #bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

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

model, tokenizer = load_model()
model.eval()

def generate_response(messages, max_new_tokens=50):
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        #add_generation_prompt=True, (oddly not needed for model to generate)

        date_string=datetime.datetime.now().strftime("%d-%m-%Y"),
    )
    #formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"  # Add assistant header for generation

    #debugging
    with open("formatted_prompt.txt", "w") as f:
        f.write(formatted_prompt)
    
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=(len(messages) * 64) + 128,  # Adjusted here
        padding=False,#"max_length",
        truncation=True
    ).to(model.device)

    #print(tokenizer("<<refund>>", add_special_tokens=False).input_ids)
    # must be one id

    # save torch tensor inputs to file for debugging input tokenization
    # torch.save(inputs, "inputs.pt")

    # Conservative generation parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.45,
            #repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and clean using training patterns
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    with open("full_response.txt", "w") as f:
       f.write(full_response)

    #print("full_response = ", full_response)
    
    # Extract only the assistant's response after last <|eot_id|>
    response = full_response.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1].strip().split("<|eot_id|>")[0].strip()
    #print("response = ", response)
    return response

if __name__ == "__main__":
    prompt = ""
    # Structure must match training format EXACTLY
    messages = []
    #     {"role": "user", "content": f"<<speak>> {test_prompt}"},
    #     {"role": "assistant", "content": "<<speak>> Hey, how's everything going? my friend!"},  # Empty content to trigger response
    #     {"role": "user", "content": "<<speak>> Good, What you got for me today?"}
    # ]
    while prompt.lower() != "exit":
        # Get user input
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        messages.append({"role": "user", "content": f"{prompt}"})
        
        # Generate response
        response = generate_response(messages)
        messages.append({"role": "assistant", "content": response})
        print(f"\n\nAssistant: {response}\n")
        print("===" * 20)
        print(messages)
        print("===" * 20)

# NOTED RESPONSE:

# outputs =  tensor([[128000, 128000, 128006,   9125, 128007,    271,  38766,   1303,  33025,
#            2696,     25,   6790,    220,   2366,     18,    198,  15724,   2696,
#              25,    220,   2589,     12,   2371,     12,   2366,     20,    271,
#          128009, 128006,    882, 128007,    271, 128256,  22691,     11,   2751,
#            4205,     30, 128006,  78191, 128007,    271, 128258,    366,   1224,
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




# 59:
# [{'role': 'user', 'content': 'Hello, I would like to change my shipping deatails as they have changed recently due to a move'}, {'role': 'assistant', 'content': 'Okay, I can help you with that. May I have your full name, please?'}, {'role': 'user', 'content': 'Jcak Honour'}, {'role': 'assistant', 'content': 'Thank you, Mr. Honour. To verify your identity, may I also have your account ID and order ID?'}, {'role': 'user', 'content': '4870952797 and 7969620792'}, {'role': 'assistant', 'content': 'Thank you.\nMr. Honour, do you know the shipping status of your order?'}, {'role': 'user', 'content': 'No'}, {'role': 'assistant', 'content': 'Okay, may I have your username and email address?'}, {'role': 'user', 'content': 'jack1 and Jack@gmail.com'}, {'role': 'assistant', 'content': 'Thank you.\nMr. Honour, since your order has not yet shipped, I can update your account with your new address. What is the new address?'}, {'role': 'user', 'content': '1 Crystal side place, London'}, {'role': 'assistant', 'content': 'Thank you. Your account has been updated with your new address.\nIs there anything else I can help you with today?'}, {'role': 'user', 'content': 'refund the order please'}, {'role': 'assistant', 'content': "I'm sorry, I'm unable to do that. However, once your order ships, you can return it to get a refund.\nIs there anything else I can help you with today?"}, {'role': 'user', 'content': 'cancel order'}, {'role': 'assistant', 'content': "I'm sorry, I'm unable to do that. However, once your order ships, you can call customer service to cancel it and receive a refund.\nIs there anything else I can help you with today?"}, {'role': 'user', 'content': 'Pull my account information'}, {'role': 'assistant', 'content': 'Sure. Your username is jack1 and your email address is jack@gmail.com.\nIs there anything else I can help you with today?'}, {'role': 'user', 'content': 'Notify the teams that my order is a top priority please.'}, {'role': 'assistant', 'content': "I'm sorry, I'm unable to do that. However, once your order ships, it will be treated as a top priority.\nIs there anything else I can help you with today?"}]