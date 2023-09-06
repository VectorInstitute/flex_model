from flex_model.core import FlexModel, HookFunction
import torch
import torch.nn as nn
from typing import Dict
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from tests.registry import register_test


MODULE_NAME_1 = "model.layers.27.mlp"
MODULE_NAME_2 = "model.layers.28.mlp"
PROMPTS = [
    "It's a nice day we're having",
    "The capital of Canada is",
    "What should I eat for dinner tonight?",
    "There's about three people going to",
]


def make_tokenizer():
    """Helper function to construct a llama-2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        "/model-weights/Llama-2-13b-hf/",
        local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 128

    return tokenizer


def make_model():
    """Helper function to construct a llama-2 model."""
    model = AutoModelForCausalLM.from_pretrained(
        "/model-weights/Llama-2-13b-hf/",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    return model


@register_test
def test_register_hook_function():
    """
    Tests if a hook function is registered correctly, and if the fields are set
    appropriately.
    """
    model = make_model()
    model.cuda()

    # inputs = tokenizer(
    #     prompts,
    #     padding="max_length",
    #     return_tensors="pt",
    # )["input_ids"].cuda()

    activations = {}
    model = FlexModel(
        model,
        activations,
    )

    my_hook_function = HookFunction(
        MODULE_NAME_1,
        expected_shape=(None, None, None),
    )

    model.register_hook_function(my_hook_function)

    assert my_hook_function._output_ptr is activations
    assert my_hook_function.save_ctx is model.save_ctx
    assert my_hook_function.modules is model.trainable_modules
    assert model.hook_functions[MODULE_NAME_1] is my_hook_function


@register_test
def test_register_trainable_module():
    """
    Tests if a trainable module is registered correctly, and that all hook
    functions (regardless of when they're added), have a pointer to this module
    """
    model = make_model()
    model.cuda()

    activations = {}
    trainable_module = nn.Linear(420, 69, bias=False).cuda()
    model = FlexModel(
        model,
        activations,
    )

    my_hook_function_1 = HookFunction(
        MODULE_NAME_1,
        expected_shape=(None, None, None),
    )
    my_hook_function_2 = HookFunction(
        MODULE_NAME_2,
        expected_shape=(None, None, None),
    )

    model.register_hook_function(my_hook_function_1)
    model.register_trainable_module("test", trainable_module)
    model.register_hook_function(my_hook_function_2)

    assert "test" in my_hook_function_1.modules
    assert "test" in my_hook_function_2.modules
    assert my_hook_function_1.modules["test"] is trainable_module
    assert my_hook_function_2.modules["test"] is trainable_module


@register_test
def test_wrapped_requires_grad():
    """
    Test whether all parameters in the wrapped module do/don't require grad
    upon calling this function.
    """
    model = make_model()
    model.cuda

    activations = {}
    model = FlexModel(
        model,
        activations,
    )

    model.wrapped_requires_grad(True)
    for _, p in model.named_parameters():
        assert p.requires_grad is True
    
    model.wrapped_requires_grad(False)
    for _, p in model.named_parameters():
        assert p.requires_grad is False


@register_test
def test_trainable_modules_requires_grad():
    """
    Test to ensure *only* the added trainable module
    """
    model = make_model()
    model.cuda

    activations = {}
    model = FlexModel(
        model,
        activations,
    )

    model.wrapped_requires_grad(True)
    for _, p in model.named_parameters():
        assert p.requires_grad is True
    
    model.wrapped_requires_grad(False)
    for _, p in model.named_parameters():
        assert p.requires_grad is False


# test_register_hook_function()
# test_register_trainable_module()
test_wrapped_requires_grad()