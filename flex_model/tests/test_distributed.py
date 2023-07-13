from functools import partial
import logging
import os
from typing import Tuple, Dict, List, Optional, Any, Callable

from accelerate import Accelerator
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import torch.nn as nn
from torch import Tensor

from flex_model._traverse_utils import _recursively_find_first_tensor

from flex_model.tests.testing_utils import (
    print_named_modules,
    print_return_dict,
    parse_base_model_output_with_past,
    module_comparison_mapping,
    compare_tensor_dicts,
    apply_torch_fwd_hooks,
    apply_flex_model_fwd_hooks,
    get_llama_13b_hf,
)
from flex_model.tests.testing_constants import (
    _PROMPTS,
    _LLAMA_VANILLA_TORCH_MODULES,
    _FSDP_PREFIX,
    _LLAMA_FSDP_MODULES,
)


logger = logging.getLogger(__name__)



def _llama_vanilla_torch_run() -> Dict[str, Tensor]:
    """Forward pass through single gpu llama model and apply forward hooks."""
    model, tokenize_fn = get_llama_13b_hf()
    print_named_modules(model)

    inputs = tokenize_fn(_PROMPTS).cuda()

    output_dict = apply_torch_fwd_hooks(
        model=model,
        inputs=inputs,
        module_names_with_shapes=_LLAMA_VANILLA_TORCH_MODULES,
        parse_fn=parse_base_model_output_with_past,
    )
    return output_dict


def _llama_fsdp_run() -> Dict[str, Tensor]:
    """Forward pass through dual gpu fsdp llama model and apply forward hooks.
    """
    model, tokenize_fn = get_llama_13b_hf()
    print_named_modules(model)

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    inputs = tokenize_fn(_PROMPTS).to(accelerator.device)
    logger.info(f"Rank{torch.distributed.get_rank()} inputs: {inputs}")

    output_dict = apply_flex_model_fwd_hooks(
        model=model,
        inputs=inputs,
        module_names_with_shapes=_LLAMA_FSDP_MODULES,
    )
    return output_dict


def _llama_megatron_run() -> Dict[str, Tensor]:
    """Forward pass through dual gpu fsdp llama model and apply forward hooks.
    """
    raise NotImplementedError


def test_distributed_flex_model_fsdp():
    """Compare single gpu llama to fsdp llama."""
    vanilla_torch_output = _llama_vanilla_torch_run()

    fsdp_output = _llama_fsdp_run()
    
    # Prune non-rank0 workers
    if len(fsdp_output) == 0:
        return

    
    print("*" * 50)
    print_return_dict(vanilla_torch_output)
    print("*" * 50)
    print_return_dict(fsdp_output)
    print("*" * 50)

    mapping = module_comparison_mapping(
        _LLAMA_VANILLA_TORCH_MODULES,
        _LLAMA_FSDP_MODULES,
    )
    print(mapping)
    assert compare_tensor_dicts(vanilla_torch_output, fsdp_output, mapping)

    logger.info("Test successful!")


# TODO: Be more consistent with logging messages
def main():
    # Distributed config
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"

    logger.info("Testing Huggingface llama-13b dual gpu...")
    test_distributed_flex_model_fsdp()


if __name__ == "__main__":
    main()
