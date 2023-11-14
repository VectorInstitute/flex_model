from __future__ import annotations

import argparse
import os
from argparse import Namespace
from functools import partial
from typing import Any, Callable

import einops
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from flex_model.core import FlexModel, HookFunction


def setup() -> None:
    """Instantiate process group."""
    dist.init_process_group("nccl")


def cleanup() -> None:
    """Destroy process group."""
    dist.destroy_process_group()


def args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    return parser.parse_args()


def setup_model(model_path: str, rank: int) -> tuple[nn.Module, LlamaConfig]:
    """Instantiate model, tokenizer, and config.

    Args:
    ----
        model_path: A path to the model being instantiated
        rank: The worker rank

    Returns:
    -------
        A tuple of length two containing the model and the config.
    """
    config = LlamaConfig.from_pretrained(model_path)
    if rank == 0:
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
    else:
        with torch.device("meta"):
            model = LlamaForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
            )
    return model, config


def fsdp_config(rank: int) -> dict[str: Any]:
    """Return the config to be used by FSDP.

    Args:
    ----
        rank: The worker rank

    Returns:
    -------
        A dictionary containing keyword -> respective configuration.
    """
    def _module_init_fn(module: nn.Module) -> Callable:
        """Return the function used for initializing modules on FSDP workers."""
        return module.to_empty(
            device=torch.cuda.current_device(),
            recurse=False,
        )

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )
    sharding_strategy = ShardingStrategy.FULL_SHARD
    device_id = torch.cuda.current_device()
    sync_module_states = True
    param_init_fn = _module_init_fn if rank != 0 else None
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

    config = {
        "auto_wrap_policy": auto_wrap_policy,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "sync_module_states": sync_module_states,
        "param_init_fn": param_init_fn,
        "mixed_precision": mp_policy,
    }
    return config


def calculate_induction_score(
        config: LlamaConfig,
        activation_dict: dict[str, torch.Tensor],
        module_names: list[str],
        sequence_length: int,
    ) -> None:
    """Calculate and save a heatmap of the induction scores for each attention
    head.

    Args:
    ----
        config: The model's HF config
        activation_dict: Dictionary containing the activations retrieved using
            FlexModel
        module_names: A list of the module names to which we have attached
            hooks
        sequence_length: The sequence length of the prompt passed into the
            model
    """
    induction_score_store = torch.zeros(
        (
            config.num_hidden_layers,
            config.num_attention_heads,
        ),
        device=torch.cuda.current_device(),
    )

    for i, module_name in enumerate(module_names):
        attn_maps = activation_dict[module_name][0].detach().to(
            torch.cuda.current_device(),
        )
        induction_stripe = attn_maps.diagonal(
            dim1=-2, dim2=-1, offset=1-sequence_length,
        )
        induction_score = einops.reduce(
            induction_stripe, "batch head_index position -> head_index", "mean",
        )
        induction_score_store[i, :] = induction_score

    plt.imshow(induction_score_store.cpu().numpy(), origin="lower")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Induction Score by Head")
    plt.colorbar()
    plt.savefig("induction_score_by_head.png", bbox_inches="tight")


def get_module_names(config: LlamaConfig) -> list[str]:
    """Return the list of module names to apply hooks onto.

    Args:
    ----
        config: The model's config

    Returns:
    -------
        A list of model names that we're applying HookFunctions to
    """
    name_placeholder = """_fsdp_wrapped_module.model.layers.
        {}._fsdp_wrapped_module.self_attn.dummy"""
    module_names = [
        name_placeholder.format(i) for i in range(config.num_hidden_layers)
    ]
    return module_names


def calculate_per_token_loss(
        logits: torch.Tensor,
        prompt: torch.Tensor,
    ) -> None:
    """Calculate and plot the cross-entropy loss per token.

    Args:
    ----
        logits: The model's output logits
        prompt: The input prompt sequence
    """
    # Calculate per token loss
    log_probs = F.log_softmax(logits, dim=-1)
    predicted_log_probs = -log_probs[..., :-1, :].gather(
        dim=-1, index=prompt[..., 1:, None],
    )[..., 0]

    # Average loss across the batch dimension
    loss_by_position = einops.reduce(
        predicted_log_probs, "batch position -> position", "mean",
    )

    plt.plot(
        list(range(len(loss_by_position))),
        loss_by_position.detach().cpu().numpy(),
    )
    plt.xlabel("Token Index")
    plt.ylabel("Loss")
    plt.title("Loss by position on random repeated tokens")
    plt.savefig("induction_loss.png", bbox_inches="tight")


def main(args: Namespace) -> None:
    """Execute main demo.

    Args:
    ----
        args: Command-line arguments
    """
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)

    seq_len = 50
    prompt = torch.randint(
        500, 15000, (4, seq_len)).to(torch.cuda.current_device(),
    )
    repeated_tokens = einops.repeat(
        prompt, "batch seq_len -> batch (2 seq_len)",
    )

    model, config = setup_model(args.model_path, rank)
    fsdp_cfg = fsdp_config(rank)

    model = FSDP(
        model,
        **fsdp_cfg,
    )

    # Wrap the model
    output_dict = {}
    model = FlexModel(
        model,
        output_dict,
        data_parallel_size=dist.get_world_size(),
    )

    # Register hooks for activations
    module_names = get_module_names(config)
    for i in range(config.num_hidden_layers):
        module_name = module_names[i]
        model.register_forward_hook(
            HookFunction(
                module_name,
                (None, None, None, None),
            ),
        )

    out = model(repeated_tokens).logits

    # Do plotting on main rank
    if dist.get_rank() == 0:
        calculate_induction_score(
            config,
            output_dict,
            module_names,
            seq_len,
        )
        plt.clf()

        # Note: we are only calculating this over the main rank's output
        # for the purpose of demonstration
        calculate_per_token_loss(out, repeated_tokens)


if __name__ == "__main__":
    parsed_args = args()
    setup()
    main(parsed_args)
    cleanup()
