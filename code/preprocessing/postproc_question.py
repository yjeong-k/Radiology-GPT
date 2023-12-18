import argparse
import re

import pandas as pd
import tiktoken

prompt_summarization = """You are an intelligent clinical language model. Your task is to analyze a radiology report and provide an appropriate answer to the given question, solely based on the report.

Patient's radiology report and a question that you're supposed to answer based on the report is provided below.

[Radiology Report Begin]
{report}
[Radiology Report End]

[Question Begin]
{question}
[Question End]

Generate an appropriate answer to the question, using the contents of the radiology report.

When creating your answer, you must meet the conditions below:
1. Your answer must be accurate
2. Your answer should be as concise as possible.
3. Generate answers that not only medical professionals but also non-medical professionals can understand.
4. If it is difficult to give an exact answer to a question, do not provide an answer and explain why it is difficult to answer.
5. Don't simply repeat questions or regenerate questions.

Response:"""

prompt_expansion = """You are an intelligent clinical language model. Your task is to analyze a radiology report and provide an appropriate answer to the given question.

Patient's radiology report and a question that you're supposed to answer are provided below.

[Radiology Report Begin]
{report}
[Radiology Report End]

[Question Begin]
{question}
[Question End]

Generate an appropriate answer to the question, using the radiology report.

When creating your answer, you must meet the conditions below:
1. Your answer must be accurate.
2. If your answer contains medical terms, explain their meaning for non-medical professionals.
3. Your answer should be a maximum of 3 sentences.
4. If it is difficult to give an exact answer to a question, do not provide an answer and explain why it is difficult to answer.
5. Don't simply repeat questions or regenerate questions.

Response:"""


prompt = """You are an intelligent clinical language model. Your task is to analyze a radiology report and provide an appropriate answer to the given question.

Patient's radiology report and a question that you're supposed to answer are provided below.

[Radiology Report Begin]
{report}
[Radiology Report End]

[Question Begin]
{question}
[Question End]

Generate an appropriate answer to the question, using the radiology report.

When creating your answer, you must meet the conditions below:
1. Your answer must be accurate.
2. Generate answers that not only medical professionals but also non-medical professionals can understand.
3. Your answer should be a maximum of 3 sentences.
4. If it is difficult to give an exact answer to a question, do not provide an answer and explain why it is difficult to answer.
5. Don't simply repeat questions or regenerate questions.

Response:"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--save_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_json(args.input_path, lines=True)

    df['filter'] = df[1].map(lambda x: False if type(x) == list else True)
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
    df["task"] = (
        df[0]
        .map(lambda x: x["messages"][0]["content"])
        .map(
            lambda x: re.findall(
                r"\"(.*?)\"",
                x,
                flags=re.DOTALL | re.MULTILINE,
            )[0]
        )
    )

    df["question"] = df[1].map(lambda x: x["choices"][0]["message"]["content"])

    """df["input"] = df.apply(
        lambda x: prompt.format(
            report=x["report"],
            question=x["question"],
        ),
        axis=1,
    )"""

    df["input"] = df.apply(
        lambda x: (
            prompt_summarization.format(
                report=x["report"],
                question=x["question"],
            )
            if x["task"] == "Summarization"
            else prompt_expansion.format(
                report=x["report"],
                question=x["question"],
            )
            if x["task"] == "Abbreviation Expansion"
            else prompt.format(
                report=x["report"],
                question=x["question"],
            )
        ),
        axis=1,
    )

    df["input"].map(
        lambda x: {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": x,
                }
            ],
            # Max of GPT-3.5-Turbo
            "max_tokens": len(tiktoken.get_encoding("cl100k_base").encode(x)) + 400,
            "temperature": 1,
        },
    ).to_json(args.save_path, orient="records", lines=True)


if __name__ == "__main__":
    main()