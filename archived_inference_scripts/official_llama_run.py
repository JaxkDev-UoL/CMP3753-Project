# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# Official llama inference code:

import json
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, TextStreamer
import torch


class Llama:
    """
    Wrapper for local LLaMA model loading and text generation.
    """
    def __init__(self, model_dir: str, max_seq_len: int):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on device: {self.device}")

        self.config = LlamaConfig.from_pretrained(model_dir)
        f = open(model_dir + "generation_config.json", "r")
        self.generator_config = json.load(f)
        f.close()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        #print(self.tokenizer.encode("Hello World!"))
        #exit()

        # Ensure pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            print("No pad_token_id found in tokenizer. Setting pad_token_id to eos_token_id.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.generator_config['pad_token_id'] = self.tokenizer.pad_token_id
        #print(f"Tokenizer loaded: {self.tokenizer} (pad_token_id: {self.tokenizer.pad_token_id})")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,  # Use FP16 for better GPU performance
            device_map="auto",
            config=self.config,
        )
        self.max_seq_len = max_seq_len

    @classmethod
    def build(cls, model_dir: str, max_seq_len: int):
        """
        Factory method to initialize the Llama wrapper.
        """
        return cls(model_dir, max_seq_len)

    def chat_completion(self, dialogs: List[List[dict]], max_gen_len: Optional[int], temperature: float, top_p: float):
        """
        Generate responses for dialogs.
        """
        results = []
        for dialog in dialogs:
            prompt = self._build_prompt(dialog)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len).to(self.device)

            attention_mask = inputs["attention_mask"]

            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_gen_len or self.max_seq_len,
                attention_mask=attention_mask,
                **self.generator_config,
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"generation": {"role": "assistant", "content": generated_text}})
        return results

    def _build_prompt(self, dialog: List[dict]):
        """
        Convert dialog history into a single prompt for the model.
        """
        prompt = ""
        for msg in dialog:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"[INST] {content} [/INST]\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        return prompt.strip()

def main():
    """
    Main function to load the model and generate responses for predefined dialogs.
    """
    model_dir = "models/Llama-3.1-8b-Instruct/"  # Replace with the local path to the tokenizer
    #model_dir = "finetuned/1_llama_3_1_8b_instruct_abcd/full_model/"  # Replace with the local path to the tokenizer
    max_seq_len = 512

    # Initialize the generator
    generator = Llama.build(
        model_dir=model_dir,
        max_seq_len=max_seq_len
    )

    # Initialize the dialog history with the system instruction only once
    dialog_history = [
        {"role": "system", "content": "Give a direct answer ONLY, keep extended chat to minimum and remain polite."}
    ]

    #print("Example of a multi-turn conversation:")

    while True:
        # Accept user input
        user_input = input("User: ")

        # Add the user input to the dialog history
        dialog_history.append({"role": "user", "content": user_input})

        # Generate response from the assistant (model)
        result = generator.chat_completion(
            [dialog_history],  # Pass the entire dialog history with system included once
            max_gen_len=100*(len(dialog_history)-1),
            temperature=0.05,
            top_p=0.95,
        )

        print("Result:")
        print(result)
        print("-------------------------------")

        # Get the response content from the model's output
        assistant_response = result[0]["generation"]["content"].split("\n")[-1]

        # Add assistant response to the dialog history
        dialog_history.append({"role": "assistant", "content": assistant_response})

        # Print assistant response
        print(f"\nLLama: {assistant_response}\n")

        # Break out of the loop if the user types "exit"
        if user_input.lower() == "exit":
            break




if __name__ == "__main__":
    main()
