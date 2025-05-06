import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./finetuned/63_results_abcd"
BASE_MODEL = "models/Llama-3.1-8B-Instruct"

def load_model():
    # Load tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    return model.to('cuda'), tokenizer

model, tokenizer = load_model()
model.eval()

def get_model():
    return BASE_MODEL.split("/")[-1] +"_" + MODEL_PATH.split("/")[-1].split("_")[0]

def generate_response(messages, max_new_tokens=50):
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,

        date_string=datetime.datetime.now().strftime("%d-%m-%Y"), # Used in default system prompt.
    )
    
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=(len(messages) * 64) + 128,  # Adjusted max length here to avoid truncation over time
        padding=False,
        truncation=True
    ).to(model.device)

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

    with open("debug_conversation.txt", "w") as f:
        f.write("Model: " + get_model())
        f.write("Timestamp: " + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        f.write("\n\n\n\n")
        f.write("Prompt: " + formatted_prompt)
        f.write("\n\n\n\n")
        f.write("Response: " + full_response)
        f.write("\n\n\n\n")
        f.write("Special Tokens count: " + str(sum(map(lambda x: x > 128255, outputs[0].tolist())))) #output id's
        f.write("\n\n\n\n")
        f.write("Output ID's: " + str(outputs[0].tolist())) #output id's

    
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

    print("=== Welcome to the Llama-3.1-8B-Instruct inference ===")
    print("=== Type 'exit' to quit ===")
    print("=== Type 'clear' or 'reset' to clear conversation history ===")


    while prompt.lower() != "exit":
        # Get user input
        prompt = input("You: ")
        if prompt.lower() == "exit":
            print("Goodbye...")
            break
        if prompt.lower() == "clear" or prompt.lower() == "reset":
            messages = []
            print("Cleared conversation history.")
            continue

        messages.append({"role": "user", "content": f"{prompt}"})
        
        # Generate response
        response = generate_response(messages)
        messages.append({"role": "assistant", "content": response})
        print(f"\n\nAssistant: {response}\n")
        print("===" * 20)
        print(messages)
        print("===" * 20)
