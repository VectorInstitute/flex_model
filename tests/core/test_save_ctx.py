import logging
from functools import partial

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

import flex_model.distributed as dist
from flex_model.core import FlexModel, HookFunction
from flex_model.utils import setup_logger
from tests.registry import register_test
from tests.test_utilities import make_model_and_tokenizer

logger = logging.getLogger(__name__)


@register_test
def test_save_ctx():
    accelerator = Accelerator()

    model, tokenizer = make_model_and_tokenizer()
    model = accelerator.prepare(model)

    activations = {}
    model = FlexModel(model, activations, data_parallel_size=accelerator.num_processes,)

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    inputs = tokenizer(prompts, padding="max_length", return_tensors="pt",)["input_ids"]

    # Function to save an activation tensor for later use. The same activation
    # tensor is also saved into the `activations` dict we passed initially to
    # the `FlexModel.__init__()`. Hence we can verify that the `save_ctx` and
    # `activations` dict versions of the same tensor are indeed `torch.equal`.
    def retrieve_fn(current_module, inputs, save_ctx, modules):
        # Detach activation tensor and dump to cpu
        save_ctx.activation = inputs.detach().cpu()
        return inputs

    # Function to verify we still have access to the saved tensor
    def verify_fn(current_module, inputs, save_ctx, modules, act_dict):
        act_dict["save_ctx_activation"] = save_ctx.activation
        return inputs

    retrieve_hook_fn = HookFunction(
        "_fsdp_wrapped_module.model.layers.12", (None, None, 5120), retrieve_fn,
    )
    verify_hook_fn = HookFunction(
        "_fsdp_wrapped_module.model.layers.30",
        (None, None, 5120),
        partial(verify_fn, act_dict=activations),
    )
    model.register_hook_function(retrieve_hook_fn)
    model.register_hook_function(verify_hook_fn)

    _ = model(inputs)

    # Verify that the two verions of the same tensor are equal
    if accelerator.is_main_process:
        assert torch.equal(
            activations["save_ctx_activation"],
            activations["_fsdp_wrapped_module.model.layers.12"],
        )
    logger.debug("Test successful")


test_save_ctx()
