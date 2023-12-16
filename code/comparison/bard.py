# Get answers from Bard baseline model for comparison
# Input : a CSV file with a column containing answer-generating prompts
# Output : a CSV file with a newly added column conatining each corresponding answer for the prompts

from bardapi import BardCookies
import pandas as pd
import argparse
import bard_secrets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output CSV file")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_path)

    bard = BardCookies(cookie_dict=bard_secrets.COOKIE_DICT)
    # COKKIE_DICT would look like the following
    # COOKIE_DICT = {
    #     "__Secure-1PSID": "yours",
    #     "__Secure-1PSIDTS": "yours"
    # }

    for index, row in df.iterrows():
        prompt = row['prompt']

        # ask Bard
        answer = bard.get_answer(prompt)['content']
        answer = answer.strip('"').strip("'")

        df.at[index, 'bard_answer'] = answer

    df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()

