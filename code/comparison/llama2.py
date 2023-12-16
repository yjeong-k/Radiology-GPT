# Get answers from non-finetuned llama2 baseline model for comparison
# Input : a CSV file with a column containing answer-generating prompts
# Output : a CSV file with a newly added column containing each corresponding answer for the prompts

from transformers import pipeline
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

    model_name = "meta-llama/Llama-2-7b-chat-hf"

    llama2_pl = pipeline("text-generation", model=model_name, tokenizer=model_name)

    for index, row in df.iterrows():
        prompt = row['prompt']

        # ask llama2
        response = llama2_pl(prompt)

        # find the answer
        # if a prompt doesn't contain "Response:" at the end, you MUST add it
        answer = response[0]['generated_text'].split('Response:')[1].strip()
        answer = answer.strip('"').strip("'")

        df.at[index, 'llama2_answer'] = answer

    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
