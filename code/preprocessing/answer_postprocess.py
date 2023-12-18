import argparse
import re

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_json(args.input_path, lines=True)

    df['filter'] = df[1].map(lambda x: False if type(x)==list else True)
    df = df[df['filter']]

    df["report"] = (
        df[0]
        .map(lambda x: x["messages"][0]["content"])
        .map(
            lambda x: re.findall(
                r"\[Radiology Report Begin\]\n(.*)\n\[Radiology Report End\]",
                x,
                flags=re.DOTALL | re.MULTILINE,
            )[0]
        )
    )
    df["instruction"] = (
        df[0]
        .map(lambda x: x["messages"][0]["content"])
        .map(
            lambda x: re.findall(
                r"\[Question Begin\]\n(.*)\n\[Question End\]",
                x,
                flags=re.DOTALL | re.MULTILINE,
            )[0]
        )
    )
    df["answer"] = df[1].map(lambda x: x["choices"][0]["message"]["content"])

    df = df[["report", "instruction", "answer"]]

    df.to_csv(args.save_path)


if __name__ == "__main__":
    main()