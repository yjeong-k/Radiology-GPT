# Get answers from medAlpaca baseline model for comparison
# Input : a CSV file with a column containing answer-generating prompts
# Output : a CSV file with a newly added column containing each corresponding answer for the prompts

from transformers import pipeline, AutoTokenizer
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

    # Custom tokenizer, since we have long input sequence(prompt)!
    tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b")
    # Set the maximum input sequence length
    max_input_length = 1024  # originally it is 512
    tokenizer.model_max_length = max_input_length

    medAlpaca_pl = pipeline("text-generation", model="medalpaca/medalpaca-7b", tokenizer=tokenizer, max_new_tokens=1024)

    for index, row in df.iterrows():
        prompt = row['prompt']

        # ask medAlpaca
        response = medAlpaca_pl(prompt)

        # find the answer
        # if a prompt doesn't contain "Response:" at the end, you MUST add it
        answer = response[0]['generated_text'].split('Response:')[1].strip()
        answer = answer.strip('"').strip("'")

        df.at[index, 'medAlpaca_answer'] = answer

    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
