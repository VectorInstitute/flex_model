import logging

import pytest
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

import flex_model.distributed as dist
import tests.testing_utils as utils
from flex_model.core import FlexModel, HookFunction
from flex_model.utils import setup_logger

logger = logging.getLogger(__name__)

LLAMA_MODULES = {
    "model.layers.5.self_attn": (None, None, None),
    "model.layers.20.self_attn": (None, None, None),
    "model.layers.39.self_attn": (None, None, None),
    "model.layers.6.self_attn.o_proj_dummy": (None, None, None),
    "model.layers.7.self_attn.v_proj_dummy": (None, None, None),
    "model.layers.11.self_attn.k_proj_dummy": (None, None, None),
    "model.layers.8.self_attn.q_proj_dummy": (None, None, None),
    "model.layers.11.post_attention_layernorm": (None, None, None),
    "model.layers.32.mlp": (None, None, None),
    "model.layers.7.mlp.gate_proj_dummy": (None, None, None),
    "model.layers.28.mlp.up_proj_dummy": (None, None, None),
    "model.layers.9.mlp.down_proj_dummy": (None, None, None),
    "model.embed_tokens": (None, None, None),
    "model.layers.7": (None, None, None),
    "lm_head_dummy": (None, None, None),
}
LLAMA_MODULES_FSDP = {
    "_fsdp_wrapped_module.model.layers.5._fsdp_wrapped_module.self_attn": (
        None,
        None,
        5120,
    ),
    "_fsdp_wrapped_module.model.layers.20._fsdp_wrapped_module.self_attn": (
        None,
        None,
        5120,
    ),
    "_fsdp_wrapped_module.model.layers.39._fsdp_wrapped_module.self_attn": (
        None,
        None,
        5120,
    ),
    "_fsdp_wrapped_module.model.layers.6._fsdp_wrapped_module.self_attn.o_proj_dummy": (
        None,
        None,
        5120,
    ),
    # "_fsdp_wrapped_module.model.layers.2._fsdp_wrapped_module.mlp.act_fn",
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.self_attn.v_proj_dummy": (
        None,
        None,
        5120,
    ),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.self_attn.k_proj_dummy": (
        None,
        None,
        5120,
    ),
    "_fsdp_wrapped_module.model.layers.8._fsdp_wrapped_module.self_attn.q_proj_dummy": (
        None,
        None,
        5120,
    ),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.post_attention_layernorm": (
        None,
        None,
        5120,
    ),
    "_fsdp_wrapped_module.model.layers.32._fsdp_wrapped_module.mlp": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.mlp.gate_proj_dummy": (
        None,
        None,
        13824,
    ),
    "_fsdp_wrapped_module.model.layers.28._fsdp_wrapped_module.mlp.up_proj_dummy": (
        None,
        None,
        13824,
    ),
    "_fsdp_wrapped_module.model.layers.9._fsdp_wrapped_module.mlp.down_proj_dummy": (
        None,
        None,
        5120,
    ),
    "_fsdp_wrapped_module.model.embed_tokens": (None, None, 5120),
    # "_fsdp_wrapped_module.model": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.7": (None, None, 5120),
    "_fsdp_wrapped_module.lm_head_dummy": (None, None, 32000),
}


@pytest.mark.skip(reason="distributed")
def test_huggingface_llama(hook_type: str = "forward"):
    """
    Make sure an accelerate-FSDP model gives the same output as a model
    running on one gpu. The single-gpu model will process one batch at a time.
    """
    accelerator = Accelerator()

    model = utils.llama_13b()
    model = model.to(accelerator.device)

    tokenizer = utils.llama_tokenizer()

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    inputs = tokenizer(prompts, padding="max_length", return_tensors="pt",)[
        "input_ids"
    ].to(accelerator.device)

    # Multi-gpu FSDP
    multi_gpu_activations = {}
    model = accelerator.prepare(model)

    flex_model = FlexModel(
        model, multi_gpu_activations, data_parallel_size=accelerator.num_processes,
    )
    for module_name, expected_shape in LLAMA_MODULES_FSDP.items():
        flex_model.register_hook_function(
            HookFunction(module_name, expected_shape, hook_type=hook_type,)
        )

    chunked_inputs = inputs.chunk(accelerator.num_processes, dim=0)

    _ = flex_model(chunked_inputs[accelerator.process_index])

    multi_gpu_activations_ = {
        k.replace("_fsdp_wrapped_module.", ""): v
        for k, v in multi_gpu_activations.items()
    }

    # Shut-down the distributed processes
    if not accelerator.is_main_process:
        return
    dist.destroy_activation_parallel()
    dist.destroy_distributed_backend()
    torch.distributed.destroy_process_group()

    # Single-gpu
    all_single_gpu_activations = {}
    single_gpu_activations = {}
    model = utils.llama_13b().cuda()

    flex_model = FlexModel(model, single_gpu_activations)

    for module_name, expected_shape in LLAMA_MODULES.items():
        flex_model.register_hook_function(
            HookFunction(module_name, expected_shape, hook_type=hook_type,)
        )

    for chunk in chunked_inputs:
        _ = flex_model(chunked_inputs[0])
        for k, v in single_gpu_activations.items():
            all_single_gpu_activations[k] = v
        single_gpu_activations.clear()

    # Make sure activations are equal
    for k in single_gpu_activations.keys():
        assert torch.allclose(
            all_single_gpu_activations[k], multi_gpu_activations_[k],
        ), (
            f"Failed: {k}, max diff: "
            f"{(all_single_gpu_activations[k] - multi_gpu_activations_[k]).abs().max()}"
        )
