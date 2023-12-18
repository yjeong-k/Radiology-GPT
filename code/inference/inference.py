""" 
This is the modification of huggingface-projects/llama-2-7b-chat
from https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/app.py

Note: This module did not use pipeline for the exploitation of system prompt modification.
"""


import argparse
import re
import torch
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TextStreamer
from peft import PeftModel

## Constant
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default = "meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--ft_path", type=str)

    return parser.parse_args()

def load():
    args = parse_args()
    if not torch.cuda.is_available():
        print("You should use GPU for inference")
    else:
        bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float32
    )
        ## load_model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config = bnb_config,
            device_map = "auto"
        )
        model = PeftModel.from_pretrained(model, args.ft_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return model, tokenizer


def inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.75,
    top_p: float = 0.9,
    top_k: float = 100,
    repetition_penalty: float = 1.2,
):
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
    input_ids = input_ids.to(model.device)

    streamer = TextStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    output_tokens = model.generate(**generate_kwargs)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens = True)
    pattern = re.compile(r"\[INST\].*?\[/INST\]", re.DOTALL)
    output_text = re.sub(pattern, "", output_text)
    return output_text

def main():
    model, tokenizer = load()
    chat_history = []
    
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "exit":
            break
        response = inference(model, tokenizer, prompt, chat_history, SYSTEM_PROMPT)
        chat_history.append((prompt, response))

if __name__=="__main__":
    main()