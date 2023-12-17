# Get answers from PMC-Llama baseline model for comparison
# Input : a CSV file with a column containing answer-generating prompts
# Output : a CSV file with a newly added column containing each corresponding answer for the prompts

from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output CSV file")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_path)

    tokenizer = LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
    model = LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')

    for index, row in df.iterrows():
        prompt = row['prompt']

        input = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        with torch.no_grad():
            response = model.generate(inputs = input["input_ids"], max_new_tokens=512)

            # find the answer
            # if a prompt doesn't contain "Response:" at the end, you MUST add it
            decoded = tokenizer.decode(response[0])
            answer = decoded.split('Response:')[1].strip()
            answer = answer.strip('"').strip("'")
        
        df.at[index, 'PMC-Llama_answer'] = answer

    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
