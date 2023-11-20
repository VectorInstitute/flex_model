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

LOCAL_RANK = None


def setup() -> None:
    """Instantiate process group."""
    dist.init_process_group("nccl")
    global LOCAL_RANK
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def cleanup() -> None:
    """Destroy process group."""
    dist.destroy_process_group()


def args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seq_length", default=50, type=int, required=False)
    return parser.parse_args()


def setup_model(model_path: str) -> tuple[nn.Module, LlamaConfig]:
    """Instantiate model, tokenizer, and config.

    Args:
    ----
        model_path: A path to the model being instantiated

    Returns:
    -------
        A tuple of length two containing the model and the config.
    """
    config = LlamaConfig.from_pretrained(model_path)
    if LOCAL_RANK == 0:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
    else:
        with torch.device("meta"):
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
    return model, config


def fsdp_config() -> dict[str:Any]:
    """Return the config to be used by FSDP.

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
    param_init_fn = _module_init_fn if LOCAL_RANK != 0 else None
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
    num_hidden_layers: int,
    num_attention_heads: int,
    activation_dict: dict[str, torch.Tensor],
    module_names: list[str],
    sequence_length: int,
) -> None:
    """Calculate and save a heatmap of the induction scores for each attention
    head.

    Args:
    ----
        num_hidden_layers: The number of transformer blocks in the model
        num_attention_heads: The number of attention heads in the model
        activation_dict: Dictionary containing the activations retrieved using
            FlexModel
        module_names: A list of the module names to which we have attached
            hooks
        sequence_length: The sequence length of the prompt passed into the
            model
    """
    # Create the matrix to store the induction scores for each head across
    # all layers
    induction_score_store = torch.zeros(
        (
            num_hidden_layers,
            num_attention_heads,
        ),
        device=torch.cuda.current_device(),
    )

    for i, module_name in enumerate(module_names):
        # Retrieve the gathered activation maps for a given module
        attn_maps = (
            activation_dict[module_name][0]
            .detach()
            .to(
                torch.cuda.current_device(),
            )
        )

        # Attention maps are of shape [batch, head, seq, seq]

        # We take the diagonal over the last two dims i.e. the query/key dims

        # We offset by 1-sequence_length because we want to see how much
        # attention is paid from the *current* token to the token that occurred
        # right after the *previous occurrence* of the *current* token (which
        # is 1-sequence_length tokens back). A better visualization can be
        # found on Anthropic's In-context Learning and Induction Heads paper
        induction_stripe = attn_maps.diagonal(
            dim1=-2,
            dim2=-1,
            offset=1 - sequence_length,
        )

        # We average across the diagonal and the batch dims to get the final
        # induction scores
        induction_score = einops.reduce(
            induction_stripe,
            "batch head_index position -> head_index",
            "mean",
        )
        induction_score_store[i, :] = induction_score

    plt.imshow(induction_score_store.detach().cpu().numpy(), origin="lower")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Induction Score by Head")
    plt.colorbar()
    plt.savefig("induction_score_by_head.png", bbox_inches="tight")


def get_module_names(num_hidden_layers: int) -> list[str]:
    """Return the list of module names to apply hooks onto.

    Args:
    ----
        num_hidden_layers: The number of transformer blocks in the model

    Returns:
    -------
        A list of model names that we're applying HookFunctions to
    """
    prefix = "_fsdp_wrapped_module.model.layers."
    postfix = "._fsdp_wrapped_module.self_attn.dummy"
    module_names = [f"{prefix}{i}{postfix}" for i in range(num_hidden_layers)]
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

    # First take log softmax across the vocab dim to get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # log_probs[..., :-1, :] takes the log probs up to the final token while
    # keeping the shape the same.

    # .gather(...) collects the correct log probs across the vocab dim given
    # the prompt

    # The reason we need prompt[..., 1:, None] is to ensure that the index
    # argument has the same rank as log_probs

    # Finally, we need [..., 0] at the end so that we get rid of the extra
    # trailing rank we created (we also could've done a .squeeze())
    predicted_log_probs = -log_probs[..., :-1, :].gather(
        dim=-1,
        index=prompt[..., 1:, None],
    )[..., 0]

    # Average loss across the batch dimension
    loss_by_position = einops.reduce(
        predicted_log_probs,
        "batch position -> position",
        "mean",
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
    torch.cuda.set_device(LOCAL_RANK)

    seq_len = args.seq_length
    batch_size = 4
    min_vocab_idx, max_vocab_idx = 500, 15000

    prompt = torch.randint(
        min_vocab_idx, max_vocab_idx, (batch_size, seq_len)
    ).to(
        torch.cuda.current_device(),
    )
    repeated_tokens = einops.repeat(
        prompt,
        "batch seq_len -> batch (2 seq_len)",
    )

    model, config = setup_model(args.model_path)
    fsdp_cfg = fsdp_config()

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
    module_names = get_module_names(config.num_hidden_layers)
    for module_name in module_names:
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
            config.num_hidden_layers,
            config.num_attention_heads,
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
