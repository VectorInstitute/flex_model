"""Runs Llama-2-13B on 2 GPUs using Fairscale's implementation of Megatron-LM
layers. This script demonstrates basic usage of `FlexModel` with a generic
`HookFunction`.

Running:

torchrun --nnodes 1 --nproc_per_node 2 megatron_example.py

"""
import argparse
from typing import Dict, List

import torch
from llama import Llama
from torch import Tensor

from flex_model.core import FlexModel, HookFunction
from flex_model.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="debug")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="/model-weights/Llama-2-13b"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="/model-weights/Llama-2-13b/tokenizer.model",
    )
    args = parser.parse_args()
    return args


def main(args):
    """Forward pass through llama-2-13b which uses megatron for TP, PP, and DP."""
    setup_logger(args.log_level)

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]
    # Load llama-2 using megatron layers
    generator = Llama.build(
        ckpt_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_dir,
        max_seq_len=512,
        max_batch_size=32,
    )
    model = generator.model

    # Define tokenizer function
    def tokenize_fn(prompts):
        input_tokens = [
            generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts
        ]
        bsz = len(input_tokens)
        total_len = max(len(t) for t in input_tokens)
        pad_id = 0
        tokens = torch.full(
            (bsz, total_len), pad_id, dtype=torch.long, device="cuda"
        )
        for k, t in enumerate(input_tokens):
            tokens[k, : len(t)] = torch.tensor(
                t, dtype=torch.long, device="cuda"
            )
        return tokens

    # Define output to dump activations to
    activation_dict: Dict[str, List[Tensor]] = {}

    # Wrap model in FlexModel (llama-2-13b requires tensor parallel size 2)
    model = FlexModel(
        model,
        activation_dict,
        tensor_parallel_size=2,
    )

    # Define some editing function
    def do_something(current_module, inputs, save_ctx, modules):
        save_ctx.my_tensor = inputs
        print(f"Saving tensor: {hash(inputs)}")
        return inputs * 20

    def do_another_thing(current_module, inputs, save_ctx, modules):
        past_tensor = save_ctx.my_tensor
        print(f"Retrieved tensor: {hash(past_tensor)}")
        return inputs + past_tensor

    # Define hook functions
    my_hook_function_1 = HookFunction(
        module_name="layers.20.feed_forward.w3",
        expected_shape=(None, None, 13824),
        editing_function=do_something,
    )
    my_hook_function_2 = HookFunction(
        module_name="layers.28.feed_forward.w3",
        expected_shape=(None, None, 13824),
        editing_function=do_another_thing,
    )

    # Register hook function with the model
    model.register_forward_hook(my_hook_function_1)
    model.register_forward_hook(my_hook_function_2)

    # Tokenize a prompt
    inputs = tokenize_fn(prompts)

    # Run through model to generate logits and activations
    _outputs = model(inputs, start_pos=0)

    # Activations are only dumped to main process. Activations per-module key
    # are accumulated in a list.
    if torch.distributed.get_rank() == 0:
        activation_1 = activation_dict["layers.20.feed_forward.w3"][0]
        print(f"Activation shape: {activation_1.shape}")
        print(activation_1)

        assert activation_1.shape[0] == 4
        assert activation_1.shape[-1] == 13824

        activation_2 = activation_dict["layers.28.feed_forward.w3"][0]
        print(f"Activation shape: {activation_2.shape}")
        print(activation_2)

        assert activation_2.shape[0] == 4
        assert activation_2.shape[-1] == 13824


if __name__ == "__main__":
    args = parse_args()
    main(args)
