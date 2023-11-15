"""Runs Llama-2-13B on 2 GPUs using PyTorch's FSDP wrapper. This script
demonstrates basic usage of the `FlexModel` wrapper with a generic
`HookFunction`.

Running:

torchrun --nodes 1 --nproc_per_node 2 fsdp_example.py
"""
import argparse
import functools
import os
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from flex_model.core import FlexModel, HookFunction
from flex_model.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="debug")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="/model-weights/Llama-2-13b-hf"
    )
    parser.add_argument(
        "--tokenizer_dir", type=str, default="/model-weights/Llama-2-13b-hf"
    )
    args = parser.parse_args()
    return args


def get_llama2_tokenizer(tokenizer_dir):
    tokenizer = LlamaTokenizerFast.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
    )
    tokenizer.model_max_length = 512

    # Llama-2 has no PAD token, substitute the EOS token.
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def make_llama2_fsdp(checkpoint_dir):
    # Load llama-2 model and prepare it for FSDP (CPU RAM-efficient)
    if dist.get_rank() == 0:
        base_model = LlamaForCausalLM.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        param_init_fn = None
    else:
        with torch.device("meta"):
            base_model = LlamaForCausalLM.from_pretrained(
                checkpoint_dir,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            )

        def _param_init_fn(module: nn.Module):
            module = module.to_empty(
                device=torch.cuda.current_device(), recurse=False
            )
            return module

        param_init_fn = _param_init_fn

    # Initialize fsdp options.
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # Shard model parameters, optimizer, grads over all GPUs.
    sharding_strategy = ShardingStrategy.FULL_SHARD

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_root_forward_inputs=True,
    )

    # Don't offload to CPU.
    cpu_offload = CPUOffload(offload_params=False)

    transformer_auto_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # Wrap model.
    model = FSDP(
        base_model,
        process_group=None,  # default pg.
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        auto_wrap_policy=transformer_auto_wrapper_policy,
        backward_prefetch=backward_prefetch,
        mixed_precision=mixed_precision,
        ignored_modules=None,
        param_init_fn=param_init_fn,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        forward_prefetch=True,
        limit_all_gathers=True,
        use_orig_params=False,
    )

    return model


def init_dist():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def main(args):
    """Forward pass of FSDP-wrapped llama-2-13b-hf model retrieving activations.

    This script must be run via Huggingface Accelerate FSDP. Retrieves
    activations over all DP-workers by gathering them in the batch dimension.
    """
    setup_logger("debug")

    init_dist()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    model = make_llama2_fsdp(args.checkpoint_dir)

    # Load tokenizer
    tokenizer = get_llama2_tokenizer(args.tokenizer_dir)

    # Define output to dump activations to
    activation_dict: Dict[str, List[Tensor]] = {}

    # Wrap model in FlexModel
    model = FlexModel(
        model,
        activation_dict,
        data_parallel_size=world_size,
    )

    # Create a hook function
    module_name = (
        "_fsdp_wrapped_module.model.layers.30._fsdp_wrapped_module.mlp"
    )
    hook_function = HookFunction(
        module_name=module_name,
        expected_shape=(None, None, None),
        editing_function=None,
    )

    # Register hook function with the model
    model.register_forward_hook(hook_function)

    # Tokenize a prompt
    inputs = tokenizer(prompts, padding="max_length", return_tensors="pt")[
        "input_ids"
    ]

    # Split the batch across dp workers
    dp_worker_inputs = inputs.chunk(world_size, dim=0)[rank]

    # Run through model to generate logits and activations
    _outputs = model(dp_worker_inputs)

    # Activations are only dumped to main process
    if rank == 0:
        activation = activation_dict[module_name][0]
        print(f"Activation shape: {activation.shape}")
        print(activation)

        assert activation.shape[0] == 4
        assert activation.shape[-1] == 5120


if __name__ == "__main__":
    args = parse_args()
    main(args)
