import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from flex_model.core import FlexModel, HookFunction
import flex_model.distributed as dist
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
    "_fsdp_wrapped_module.model.layers.5._fsdp_wrapped_module.self_attn": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.20._fsdp_wrapped_module.self_attn": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.39._fsdp_wrapped_module.self_attn": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.6._fsdp_wrapped_module.self_attn.o_proj_dummy": (None, None, 5120),
    #"_fsdp_wrapped_module.model.layers.2._fsdp_wrapped_module.mlp.act_fn",
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.self_attn.v_proj_dummy": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.self_attn.k_proj_dummy": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.8._fsdp_wrapped_module.self_attn.q_proj_dummy": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.post_attention_layernorm": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.32._fsdp_wrapped_module.mlp": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.mlp.gate_proj_dummy": (None, None, 13824),
    "_fsdp_wrapped_module.model.layers.28._fsdp_wrapped_module.mlp.up_proj_dummy": (None, None, 13824),
    "_fsdp_wrapped_module.model.layers.9._fsdp_wrapped_module.mlp.down_proj_dummy": (None, None, 5120),
    "_fsdp_wrapped_module.model.embed_tokens": (None, None, 5120),
    #"_fsdp_wrapped_module.model": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.7": (None, None, 5120),
    "_fsdp_wrapped_module.lm_head_dummy": (None, None, 32000),

}


def make_model_and_tokenizer():
    """Helper function to construct a llama-2 model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "/ssd005/projects/llm/llama-2-13b-hf",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/ssd005/projects/llm/llama-2-13b-hf",
        local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 128

    return model, tokenizer


def test_huggingface_llama():
    """
    Make sure an accelerate-FSDP model gives the same output as a model
    running on one gpu. The single-gpu model will process one batch at a time.
    """
    setup_logger("debug")

    accelerator = Accelerator()

    model, tokenizer = make_model_and_tokenizer()
    model = model.to(accelerator.device)

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]
    
    inputs = tokenizer(
        prompts,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"].to(accelerator.device)
    print(inputs)

    # Multi-gpu FSDP
    multi_gpu_activations = {}
    model = accelerator.prepare(model)

    flex_model = FlexModel(
        model,
        multi_gpu_activations,
        data_parallel_size=2,
    )
    for module_name, expected_shape in LLAMA_MODULES_FSDP.items():
        flex_model.register_hook_function(
            HookFunction(module_name, expected_shape)
        )

    chunked_inputs = inputs.chunk(2, dim=0)

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
    model, _ = make_model_and_tokenizer()
    model = model.cuda()

    flex_model = FlexModel(model, single_gpu_activations)

    for module_name, expected_shape in LLAMA_MODULES.items():
        flex_model.register_hook_function(
            HookFunction(module_name, expected_shape)
        )

    # Process microbatch 0
    _ = flex_model(chunked_inputs[0])
    for k, v in single_gpu_activations.items():
        all_single_gpu_activations[k] = v
    single_gpu_activations.clear()

    # Process microbatch 1
    _, flex_model(chunked_inputs[1])
    for k, v in single_gpu_activations.items():
        all_single_gpu_activations[k] = torch.cat(
            (single_gpu_activations[k], v),
            dim=0,
        )
    single_gpu_activations.clear()

    # Make sure activations are equal
    for k in single_gpu_activations.keys():
        assert torch.allclose(
            all_single_gpu_activations[k],
            multi_gpu_activations_[k],
        ), (f"Failed: {k}, max diff: "
            f"{(all_single_gpu_activations[k] - multi_gpu_activations_[k]).abs().max()}")


test_huggingface_llama()
