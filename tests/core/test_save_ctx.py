from functools import partial
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from flex_model.core import FlexModel, HookFunction
import flex_model.distributed as dist
from flex_model.utils import setup_logger


logger = logging.getLogger(__name__)


def make_model_and_tokenizer():
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


def test_save_ctx():
    setup_logger("debug")

    accelerator = Accelerator()

    model, tokenizer = make_model_and_tokenizer()
    model = accelerator.prepare(model)

    activations = {}
    model = FlexModel(model, activations)

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
        "_fsdp_wrapped_module.model.layers.12",
        (None, None, 5120),
        retrieve_fn,
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
