# Get answers from bioGPT-Large-finetuned-chatdoctor baseline model for comparison
# Input : a CSV file with a column containing answer-generating prompts
# Output : a CSV file with a newly added column containing each corresponding answer for the prompts
# Code is based on https://huggingface.co/Narrativaai/BioGPT-Large-finetuned-chatdoctor

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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

    model_id = "Narrativaai/BioGPT-Large-finetuned-chatdoctor"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
    model = AutoModelForCausalLM.from_pretrained(model_id)

    for index, row in df.iterrows():
        prompt = row['prompt']

        # ask bioGPT-chatDoctor
        # if a prompt doesn't contain "Response:" at the end, you MUST add it
        answer = ask_bioGPT_chatDoctor(prompt, tokenizer, model)

        # find the answer
        answer = answer.strip().strip('"').strip("'")

        df.at[index, 'bioGPT-chatDoctor_answer'] = answer

    df.to_csv(args.save_path, index=False)


def ask_bioGPT_chatDoctor(
        prompt,
        tokenizer,
        model,
        num_beams=2,
        **kwargs
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    generation_config = GenerationConfig(
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return output.split("Response:")[1]


if __name__ == "__main__":
    main()
