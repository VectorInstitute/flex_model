from argparse import Namespace

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from flex_model.core import FlexModel, HookFunction

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
        "/model-weights/Llama-2-13b-hf/", local_files_only=True,
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


def test_register_hook_function():
    """
    Tests if a hook function is registered correctly, and if the fields are set
    appropriately.
    """
    model = make_model()
    model.cuda()

    activations = {}
    model = FlexModel(model, activations,)

    my_hook_function = HookFunction(MODULE_NAME_1, expected_shape=(None, None, None),)

    model.register_hook_function(my_hook_function)

    assert my_hook_function._output_ptr is activations
    assert my_hook_function.save_ctx is model.save_ctx
    assert my_hook_function.modules is model.trainable_modules
    assert model.hook_functions[MODULE_NAME_1] is my_hook_function


def test_register_trainable_module():
    """
    Tests if a trainable module is registered correctly, and that all hook
    functions (regardless of when they're added), have a pointer to this
    module.
    """
    model = make_model()
    model.cuda()

    activations = {}
    trainable_module = nn.Linear(420, 69, bias=False).cuda()
    model = FlexModel(model, activations,)

    my_hook_function_1 = HookFunction(MODULE_NAME_1, expected_shape=(None, None, None),)
    my_hook_function_2 = HookFunction(MODULE_NAME_2, expected_shape=(None, None, None),)

    model.register_hook_function(my_hook_function_1)
    model.register_trainable_module("test", trainable_module)
    model.register_hook_function(my_hook_function_2)

    assert "test" in my_hook_function_1.modules
    assert "test" in my_hook_function_2.modules
    assert my_hook_function_1.modules["test"] is trainable_module
    assert my_hook_function_2.modules["test"] is trainable_module


def test_wrapped_module_requires_grad():
    """
    Test whether all parameters in the wrapped module do/don't require grad
    upon calling this method.
    """
    model = make_model()
    model.cuda

    activations = {}
    trainable_module = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    model = FlexModel(model, activations,)
    model.register_trainable_module("test", trainable_module)

    model.wrapped_module_requires_grad(True)
    for _, p in model.module.named_parameters():
        assert p.requires_grad is True

    model.wrapped_module_requires_grad(False)
    for _, p in model.module.named_parameters():
        assert p.requires_grad is False

    for train_mod in model.trainable_modules.values():
        for _, p in train_mod.named_parameters():
            assert p.requires_grad is True


def test_trainable_modules_requires_grad():
    """
    Test to ensure *only* the added trainable module is affected by upon
    calling this method.
    """
    model = make_model()
    model.cuda

    activations = {}
    trainable_module_1 = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    trainable_module_2 = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    model = FlexModel(model, activations,)
    model.register_trainable_module("test1", trainable_module_1)
    model.register_trainable_module("test2", trainable_module_2)

    model.trainable_modules_requires_grad(True)
    model.wrapped_module_requires_grad(True)

    for train_mod in model.trainable_modules.values():
        for _, p in train_mod.named_parameters():
            assert p.requires_grad is True

    for _, p in model.module.named_parameters():
        assert p.requires_grad is True

    model.trainable_modules_requires_grad(False)
    for train_mod in model.trainable_modules.values():
        for _, p in train_mod.named_parameters():
            assert p.requires_grad is False

    for _, p in model.module.named_parameters():
        assert p.requires_grad is True


def test_destroy():
    """
    Tests the destroy method to ensure everything is cleared appropriately.
    """
    model = make_model()
    model.cuda

    activations = {}
    trainable_module_1 = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    trainable_module_2 = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    model = FlexModel(model, activations,)
    model.register_trainable_module("test1", trainable_module_1)
    model.register_trainable_module("test2", trainable_module_2)

    my_hook_function = HookFunction(MODULE_NAME_1, expected_shape=(None, None, None),)

    model.register_hook_function(my_hook_function)
    model.destroy()

    assert model.hook_functions == {}
    assert model._hook_function_handles == {}
    assert model._hooks_active is False
    assert model.output_ptr is activations
    assert len(model.trainable_modules) == 0
    assert model.save_ctx == Namespace()


test_register_hook_function()
test_register_trainable_module()
test_wrapped_module_requires_grad()
test_trainable_modules_requires_grad()
test_destroy()
