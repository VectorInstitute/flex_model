import argparse
from typing import Dict

import torch
from accelerate import Accelerator
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
    """Forward pass of FSDP-wrapped llama-2-13b-hf model retrieving activations.

    This script must be run via Huggingface Accelerate FSDP. Retrieves
    activations over all DP-workers by gathering them in the batch dimension.
    """
    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]
    # Load llama-2-13b-hf model and prepare it for FSDP
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model = accelerator.prepare(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 128

    # Define output to dump activations to
    activation_dict: Dist[str, Tensor] = {}

    # Wrap model in FlexModel
    model = FlexModel(
        model, activation_dict, data_parallel_size=accelerator.num_processes,
    )

    # Create a hook function
    module_name = "_fsdp_wrapped_module.model.layers.30._fsdp_wrapped_module.mlp"
    hook_function = HookFunction(
        module_name=module_name,
        expected_shape=(None, None, None),
        editing_function=None,
    )

    # Register hook function with the model
    model.register_hook_function(hook_function)

    # Tokenize a prompt
    inputs = tokenizer(prompts, padding="max_length", return_tensors="pt",)["input_ids"]

    # Split the batch across dp workers
    dp_worker_inputs = inputs.chunk(accelerator.num_processes, dim=0,)[
        accelerator.process_index
    ].to(accelerator.device)

    # Run through model to generate logits and activations
    logits = model(dp_worker_inputs)

    # Activations are only dumped to main process
    if accelerator.is_main_process:
        print(f"Activation shape: {activation_dict[module_name].shape}")
        print(activation_dict[module_name])


if __name__ == "__main__":
    args = parse_args()
    main(args)
