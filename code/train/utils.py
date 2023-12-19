""" This code is modification of asclepius: https://github.com/starmpcc/Asclepius/blob/main/src/utils.py
"""

from dataclasses import dataclass, field
from typing import Optional

import io
import json
import transformers
from transformers import Trainer



def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict

def modify_special_tokens(tokenizer):
    tokenizer.add_special_tokens(
        {
            "pad_token": "<s>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
    )

    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 0
    tokenizer.pad_token_id = 1

    return tokenizer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    remove_unused_columns: bool = field(
        default=False,
    )
    dataloader_num_workers: int = field(
        default=16,
    )