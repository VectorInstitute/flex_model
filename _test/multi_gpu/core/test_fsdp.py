import logging

import torch
import torch.distributed as dist

from flex_model.core import FlexModel, HookFunction
import _test.multi_gpu.testing_utils as utils
from _test.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry


(
    register_fsdp_test,
    get_fsdp_test,
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


@register_fsdp_test
def test_fsdp_llama():
    utils.Utils.initialize_torch_distributed()

    model = utils.wrap_fsdp("llama_7b")

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

    del flex_model  # Run finalizer.

    with utils.Utils.single_gpu_context():
        # Single-gpu
        all_single_gpu_activations = {}
        single_gpu_activations = {}
        model = utils.llama_7b().cuda()

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
        assert len(all_single_gpu_activations) == len(LLAMA_MODULES)
        assert len(multi_gpu_activations) == len(LLAMA_MODULES_FSDP)
        for k in single_gpu_activations.keys():
            assert torch.allclose(
                all_single_gpu_activations[k][0],
                multi_gpu_activations_[k][0],
            ), (
                f"Failed: {k}, max diff: "
                f"{(all_single_gpu_activations[k] - multi_gpu_activations_[k]).abs().max()}"
            )

    logger.info("Tests successful.")


test_fsdp_llama()
