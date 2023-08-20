import argparse
from typing import Dict

import torch
from torch import Tensor
from llama import Llama

from flex_model.core import FlexModel, HookFunction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="/ssd005/projects/llm/llama-2-13b")
    parser.add_argument("--tokenizer_dir", type=str, default="/ssd005/projects/llm/llama-2-13b/tokenizer.model")
    args = parser.parse_args()
    return args


def main(args):
    """Forward pass through llama-2-13b which uses megatron for TP, PP, and DP.
    """
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
    tokenizer = generator.tokenizer

    # Define tokenizer function
    def tokenize_fn(prompts):
        input_tokens = [generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        bsz = len(input_tokens)
        total_len = max(len(t) for t in input_tokens)
        pad_id = 0
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(input_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        return tokens

    # Define output to dump activations to
    activation_dict: Dist[str, Tensor] = {}

    # Wrap model in FlexModel (llama-2-13b requires tensor parallel size 2)
    model = FlexModel(
        model,
        activation_dict,
        tensor_parallel_size=2,
    )

    # Create a hook function
    module_name = "layers.28.feed_forward.w3"
    hook_function = HookFunction(
        module_name=module_name,
        expected_shape=(None, None, 13824),
        editing_function=None,
    )

    # Register hook function with the model
    model.register_hook_function(hook_function)

    # Tokenize a prompt
    inputs = tokenize_fn(prompts)

    # Run through model to generate logits and activations
    logits = model(inputs, start_pos=0)

    # Activations are only dumped to main process
    if torch.distributed.get_rank() == 0:
        print(f"Activation shape: {activation_dict[module_name].shape}")
        print(activation_dict[module_name])


if __name__ == "__main__":
    args = parse_args()
    main(args)
