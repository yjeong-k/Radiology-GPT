"""
Source: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
NOTE: Note/Inst gen usually bounded on TPM, not RPM
thus, simply ignore rpm cases
Also, the file size can be loaded into mem
"""

import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import random
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata

# imports
import aiohttp  # for making API calls concurrently
import pandas as pd

@dataclass
class APICaller:
    requests_filepath: str
    save_filepath: str
    request_url: str
    api_key: str
    max_requests_per_minute: float
    max_tokens_per_minute: float
    token_encoding_name: str
    max_attempts: int
    logging_level: int

    def __post_init__(self):
        self.sema = asyncio.Semaphore(1)
        self.rate_limit_sleep = False
        self.api_endpoint = "chat/completions"
        if "azure" in self.request_url:
            self.request_headers = {"api-key": self.api_key}
        else:
            self.request_headers = {"Authorization": f"Bearer {self.api_key}"}
        self.task_id_generator = task_id_generator_function()

    async def run(self):
        df = pd.read_json(self.requests_filepath, lines=True)
        data = df[["model", "messages", "max_tokens", "temperature"]].to_dict(orient="records")
        tasks = [self.await_and_call(i) for i in data]
        await asyncio.gather(*tasks)

    async def await_and_call(self, request_json):
        """
        request: {"note":note text, "idx": task_idx}
        """
        async with self.sema:
            if self.rate_limit_sleep:
                await asyncio.sleep(10)
                self.rate_limit_sleep = False
            await asyncio.sleep(
                2048 * 60 / self.max_tokens_per_minute
            )  # Max 4000, but actually about ~2000
        task_id = next(self.task_id_generator)

        await self.call_api(task_id, request_json)

    async def call_api(
        self,
        task_id: int,
        request_json: dict,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{task_id}")
        for _ in range(self.max_attempts):
            error = None
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=self.request_url,
                        headers=self.request_headers,
                        json=request_json,
                    ) as response:
                        response = await response.json(content_type=None)
                if "error" in response:
                    logging.warning(
                        f"Request {task_id} failed with error {response['error']}"
                    )
                    error = response
                    if "Rate limit" in response["error"].get("message", ""):
                        self.rate_limit_sleep = True
                else:
                    break
            except (
                Exception
            ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
                logging.warning(f"Request {task_id} failed with Exception {e}")
                error = e
        if error:
            logging.error(
                f"Request {request_json} failed after all attempts. Saving errors"
            )
            append_to_jsonl([request_json, [str(error)]], self.save_filepath)
            return None
        else:
            append_to_jsonl([request_json, response], self.save_filepath)
            logging.debug(f"Request {task_id} saved to {self.save_filepath}")
            return response


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script
async def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--save_path")
    parser.add_argument("--request_url", default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--api_key")
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.input_path.replace(".jsonl", "_results.jsonl")

    caller = APICaller(
        requests_filepath=args.input_path,
        save_filepath=args.save_path,
        request_url=args.request_url,
        api_key=args.api_key,
        max_requests_per_minute=float(args.max_requests_per_minute),
        max_tokens_per_minute=float(args.max_tokens_per_minute),
        token_encoding_name=args.token_encoding_name,
        max_attempts=int(args.max_attempts),
        logging_level=int(args.logging_level),
    )
    await caller.run()


if __name__ == "__main__":
    asyncio.run(main())

##python script.py --input_path /path/to/requests.csv --save_path /path/to/output.csv --api_key YOUR_API_KEY