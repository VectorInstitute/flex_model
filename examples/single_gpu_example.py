import argparse
from typing import Dict

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from flex_model.core import FlexModel, HookFunction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir", type=str, default="/ssd005/projects/llm/llama-2-13b-hf"
    )
    parser.add_argument(
        "--tokenizer_dir", type=str, default="/ssd005/projects/llm/llama-2-13b-hf"
    )
    args = parser.parse_args()
    return args


def main(args):
    """Single forward pass of llama-2-13b-hf model retrieving a single activation.

    This script runs on a single GPU only.
    """
    # Load llama-2-13b-hf model
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True,
    )

    ## NEW ##
    # Define output to dump activations to
    activation_dict: Dist[str, Tensor] = {}

    # Wrap model in FlexModel
    model = FlexModel(model, activation_dict)

    # Create a hook function
    hook_function = HookFunction(
        module_name="model.layers.30",
        expected_shape=(None, None, None),  # Not sharded, can pass None per dim
        editing_function=None,  # Just doing retrieval
    )

    # Register hook function with the model
    model.register_hook_function(hook_function)
    ## NEW ##

    # Tokenize a prompt
    inputs = tokenizer("Where is the best spot for lunch?", return_tensors="pt")[
        "input_ids"
    ]

    # Run through model to generate logits and activations
    logits = model(inputs)

    print(activation_dict)


if __name__ == "__main__":
    args = parse_args()
    main(args)
