from functools import partial
import logging
from typing import Dict, Optional, List, Callable, Tuple, Any

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
import torch
from torch import Tensor
import torch.nn as nn

import flex_model.distributed as dist
from flex_model.core import FlexModel, HookFunction


logger = logging.getLogger(__name__)


def print_named_modules(model: nn.Module) -> None:
    assert isinstance(model, nn.Module)
    names = [n for n, _ in model.named_modules()]
    print(names)


def print_return_dict(d):
    if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
        for n, m in d.items():
            dist.print_rank0(f"{n}: {m.shape}")


def parse_base_model_output_with_past(x):
    """Hard-coded parse function for huggingface models.

    Note: Retrieval only.
    """
    if isinstance(x, tuple):
        return x[0]
    if isinstance(x, BaseModelOutputWithPast):
        return x.last_hidden_state
    return x


def dummy_editing_fn_with_log(x):
    logger.info(f"Running dummy editing function on tensor: {x.shape}")
    return x


def module_comparison_mapping(ref_modules, cmp_modules):
    assert len(ref_modules) == len(cmp_modules)
    return {
        ref_n: cmp_n for ref_n, cmp_n in zip(ref_modules.keys(), cmp_modules.keys())
    }


def compare_tensor_dicts(
    dict1: Dict[str, Tensor],
    dict2: Dict[str, Tensor],
    mapping: Optional[Dict[str, str]] = None,
) -> bool:
    """Check equality between two dicts."""
    # TODO: Fix impl
    if mapping is not None:
        for name1, name2 in mapping.items():
            assert name1 in dict1, f"Module {name1} not in dict"
            assert name2 in dict2, f"Module {name2} not in dict"

            act1 = dict1[name1]
            act2 = dict2[name2]

            # NOTE: Megatron vs vanilla huggingface llama need high tol
            if not torch.allclose(
                act1.to(torch.float32),
                act2.to(torch.float32),
                atol=2e-1,
                rtol=1e-1,
            ):
                logger.info(
                    f"Allclose FAILED for {name1} - {name2}"
                    f". Max diff: {torch.abs(act1 - act2).max()}"
                )

            else:
                logger.info(
                    f"Allclose PASSED for {name1} - {name2}. "
                    f"Max diff: {torch.abs(act1-act2).max()}"
                )
        return True


class SimpleModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = in_dim * 4
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, self.latent_dim)
        self.nonlin = nn.ReLU()
        self.fc2 = nn.Linear(self.latent_dim, out_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        y = self.fc1(inputs)
        y = self.nonlin(y)
        y = self.fc2(y)
        return y


def get_opt_125m():
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-125m",
    )
    return model, tokenizer


def get_llama_13b_hf():
    model = AutoModelForCausalLM.from_pretrained(
        "/ssd005/projects/llm/llama-2-13b-hf",
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        "/ssd005/projects/llm/llama-2-13b-hf",
        # "/scratch/ssd002/projects/opt_test/llama-13b-hf",
        local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    def tokenize(ps):
        return tokenizer(ps, padding=True, return_tensors="pt")["input_ids"]

    return model, tokenize


def get_llama_13b_megatron():
    from llama import Llama

    generator = Llama.build(
        ckpt_dir="/ssd005/projects/llm/llama-2-13b",
        tokenizer_path="/ssd005/projects/llm/llama-2-13b/tokenizer.model",
        max_seq_len=512,
        max_batch_size=32,
    )
    model = generator.model
    tokenizer = generator.tokenizer

    def tokenize(prompts):
        input_tokens = [
            generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts
        ]
        bsz = len(input_tokens)
        total_len = max(len(t) for t in input_tokens)
        pad_id = 0
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(input_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        return tokens

    return model, tokenize


def apply_torch_fwd_hooks(
    model: nn.Module,
    inputs: Tensor,
    module_names_with_shapes: Dict[str, Tuple[int, ...]],
    parse_fn: Optional[Callable] = None,
) -> Dict[str, Tensor]:
    """Retrieve activation using vanilla pytorch forward hooks."""

    def _fwd_hook(
        registered_name: str,
        return_dict: Dict[str, Tensor],
        module: nn.Module,
        inputs: Any,
        outputs: Any,
        parse_fn: Optional[Callable] = None,
        shape: Optional[Tuple[int]] = None,
    ) -> None:
        if parse_fn is not None:
            outputs = parse_fn(outputs)

        res = outputs.detach().cpu()

        if shape is not None:
            res = res.reshape(*shape)

        dist.print_rank0(f"Dumping: {res.shape} to output dict")
        return_dict[registered_name] = res

    output_dict: Dict[str, Tensor] = {}
    hook_handles = []
    for name, module in model.named_modules():
        if name in module_names_with_shapes:
            handle = module.register_forward_hook(
                partial(
                    _fwd_hook,
                    name,
                    output_dict,
                    parse_fn=parse_fn if parse_fn else lambda x: x,
                    shape=module_names_with_shapes[name],
                )
            )
            hook_handles.append(handle)

            logger.info(f"Installing module: {name}")

    logger.info("Running forward")
    outputs = model.forward(inputs)

    for handle in hook_handles:
        handle.remove()

    return output_dict, outputs


def apply_flex_model_fwd_hooks(
    model: nn.Module,
    inputs: Tensor,
    module_names_with_shapes: Dict[str, Tuple[int, ...]],
    *args,
    **kwargs,
) -> Dict[str, Tensor]:
    """Retrieve activations using flex model forward hooks."""
    output_dict: Dict[str, Tensor] = {}
    flex_model = FlexModel(model, output_dict)
    for name, shape in module_names_with_shapes.items():
        hook_fn = HookFunction(
            module_name=name,
            expected_shape=shape,
            editing_function=dummy_editing_fn_with_log,
        )
        flex_model.register_hook_function(hook_fn)

    outputs = flex_model.forward(inputs, *args, **kwargs)

    return output_dict, outputs
