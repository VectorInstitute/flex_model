import os
import functools
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from flex_model.core import FlexModel, HookFunction
import flex_model.distributed as fm_dist
import test.multi_gpu.testing_utils as utils
from test.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry


(
    register_fsdp_test,
    get_fsdo_test,
) = make_test_registry("fsdp", SlurmJobResourceSpec(time=10))


logger = logging.getLogger(__name__)


LLAMA_MODULES = {
    "model.layers.5.self_attn": (None, None, None),
    "model.layers.20.self_attn": (None, None, None),
    "model.layers.3.self_attn": (None, None, None),
    "model.layers.6.self_attn.o_proj": (None, None, None),
    "model.layers.7.self_attn.v_proj": (None, None, None),
    "model.layers.11.self_attn.k_proj": (None, None, None),
    "model.layers.8.self_attn.q_proj": (None, None, None),
    "model.layers.11.post_attention_layernorm": (None, None, None),
    "model.layers.2.mlp": (None, None, None),
    "model.layers.7.mlp.gate_proj": (None, None, None),
    "model.layers.28.mlp.up_proj": (None, None, None),
    "model.layers.9.mlp.down_proj": (None, None, None),
    "model.embed_tokens": (None, None, None),
    "model.layers.7": (None, None, None),
    "lm_head": (None, None, None),
}
LLAMA_MODULES_FSDP = {
    "_fsdp_wrapped_module.model.layers.5._fsdp_wrapped_module.self_attn": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.layers.20._fsdp_wrapped_module.self_attn": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.layers.3._fsdp_wrapped_module.self_attn": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.layers.6._fsdp_wrapped_module.self_attn.o_proj": (
        None,
        None,
        4096,
    ),
    # "_fsdp_wrapped_module.model.layers.2._fsdp_wrapped_module.mlp.act_fn",
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.self_attn.v_proj": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.self_attn.k_proj": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.layers.8._fsdp_wrapped_module.self_attn.q_proj": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.post_attention_layernorm": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.layers.2._fsdp_wrapped_module.mlp": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.mlp.gate_proj": (
        None,
        None,
        11008,
    ),
    "_fsdp_wrapped_module.model.layers.28._fsdp_wrapped_module.mlp.up_proj": (
        None,
        None,
        11008,
    ),
    "_fsdp_wrapped_module.model.layers.9._fsdp_wrapped_module.mlp.down_proj": (
        None,
        None,
        4096,
    ),
    "_fsdp_wrapped_module.model.embed_tokens": (None, None, 4096),
    # "_fsdp_wrapped_module.model": (None, None, 4096),
    "_fsdp_wrapped_module.model.layers.7": (None, None, 4096),
    "_fsdp_wrapped_module.lm_head": (None, None, 32000),
}


def make_llama2():
    base_model = LlamaForCausalLM.from_pretrained(
        "/model-weights/Llama-2-7b-hf",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    return base_model


def make_llama2_fsdp():
    # Load llama-2 model and prepare it for FSDP (CPU RAM-efficient)
    if dist.get_rank() == 0:
        base_model = make_llama2()
        param_init_fn = None
    else:
        with torch.device("meta"):
            base_model = make_llama2()

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


@register_fsdp_test
def test_fsdp_llama():
    init_dist()

    model = make_llama2_fsdp()

    tokenizer = utils.llama_tokenizer()

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
    )["input_ids"]

    # Multi-gpu FSDP
    multi_gpu_activations = {}

    flex_model = FlexModel(
        model,
        multi_gpu_activations,
        data_parallel_size=dist.get_world_size(),
    )
    for module_name, expected_shape in LLAMA_MODULES_FSDP.items():
        flex_model.register_forward_hook(
            HookFunction(module_name, expected_shape)
        )

    chunked_inputs = inputs.chunk(dist.get_world_size(), dim=0)

    _ = flex_model(chunked_inputs[dist.get_rank()])

    multi_gpu_activations_ = {
        k.replace("_fsdp_wrapped_module.", ""): v
        for k, v in multi_gpu_activations.items()
    }

    # Shut-down the distributed processes
    if dist.get_rank() != 0:
        return
    fm_dist.destroy_activation_parallel()
    fm_dist.destroy_distributed_backend()
    dist.destroy_process_group()

    # Single-gpu
    all_single_gpu_activations = {}
    single_gpu_activations = {}
    model = make_llama2().cuda()

    flex_model = FlexModel(model, single_gpu_activations)

    for module_name, expected_shape in LLAMA_MODULES.items():
        flex_model.register_forward_hook(
            HookFunction(module_name, expected_shape)
        )

    for chunk in chunked_inputs:
        _ = flex_model(inputs.cuda())
        for k, v in single_gpu_activations.items():
            all_single_gpu_activations[k] = v
        single_gpu_activations.clear()

    # Make sure activations are equal
    for k in single_gpu_activations.keys():
        assert torch.allclose(
            all_single_gpu_activations[k][0],
            multi_gpu_activations_[k][0],
        ), (
            f"Failed: {k}, max diff: "
            f"{(all_single_gpu_activations[k] - multi_gpu_activations_[k]).abs().max()}"
        )

    logger.info("Tests successful.")
