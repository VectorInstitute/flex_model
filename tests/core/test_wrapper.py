from argparse import Namespace
from functools import partial

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from flex_model.core import FlexModel, HookFunction
from tests.fixtures import opt_350m, opt_tokenizer

MODULE_NAME_1 = "model.decoder.layers.17.fc2"
MODULE_NAME_2 = "model.decoder.layers.18.fc2"
PROMPTS = [
    "It's a nice day we're having",
    "The capital of Canada is",
    "What should I eat for dinner tonight?",
    "There's about three people going to",
]


def test_register_forward_hook(opt_350m):
    """
    Tests if a hook function is registered correctly, and if the fields are set
    appropriately.
    """
    model = opt_350m.cuda()

    activations = {}
    model = FlexModel(model, activations,)

    my_hook_function = HookFunction(MODULE_NAME_1, expected_shape=(None, None, None))

    model.register_forward_hook(my_hook_function)

    assert my_hook_function._shared_state.output_ptr is activations
    assert my_hook_function._shared_state.save_ctx is model.save_ctx
    assert my_hook_function._shared_state.modules is model.trainable_modules
    assert my_hook_function._shared_state.offload_mode == "CPU"


def test_register_trainable_module(opt_350m):
    """
    Tests if a trainable module is registered correctly, and that all hook
    functions (regardless of when they're added), have a pointer to this
    module.
    """
    model = opt_350m.cuda()

    activations = {}
    trainable_module = nn.Linear(420, 69, bias=False).cuda()
    model = FlexModel(model, activations)

    my_hook_function_1 = HookFunction(MODULE_NAME_1, expected_shape=(None, None, None))
    my_hook_function_2 = HookFunction(MODULE_NAME_2, expected_shape=(None, None, None))

    model.register_forward_hook(my_hook_function_1)
    model.register_trainable_module("test", trainable_module)
    model.register_forward_hook(my_hook_function_2)

    assert "test" in my_hook_function_1._shared_state.modules
    assert "test" in my_hook_function_2._shared_state.modules
    assert my_hook_function_1._shared_state.modules["test"] is trainable_module
    assert my_hook_function_2._shared_state.modules["test"] is trainable_module


def test_destroy(opt_350m):
    """
    Tests the destroy method to ensure everything is cleared appropriately.
    """
    model = opt_350m.cuda()

    activations = {}
    trainable_module_1 = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    trainable_module_2 = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    model = FlexModel(model, activations,)
    model.register_trainable_module("test1", trainable_module_1)
    model.register_trainable_module("test2", trainable_module_2)

    my_hook_function = HookFunction(MODULE_NAME_1, expected_shape=(None, None, None),)

    model.register_forward_hook(my_hook_function)
    model = model.module  # Calls FlexModel.__exit__().

    assert not isinstance(model, FlexModel)
    assert not getattr(model, "hook_functions", False)
    assert not getattr(model, "_hook_function_handles", False)
    assert not getattr(model, "_hooks_active", False)
    assert not getattr(model, "output_ptr", False)
    assert not getattr(model, "save_ctx", False)
    assert not getattr(model, "trainable_modules", False)

    hook_types = {"_forward", "_forward_pre", "_backward"}
    for m in model.modules():
        for hook_type in hook_types:
            attr = hook_type + "_hooks"
            assert len(getattr(m, attr)) == 0


def test_save_ctx(opt_350m, opt_tokenizer):
    model = opt_350m.cuda()

    tokenizer = opt_tokenizer

    activations = {}
    model = FlexModel(model, activations)

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")["input_ids"].cuda()

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
        "model.decoder.layers.12", (None, None, None), retrieve_fn,
    )
    verify_hook_fn = HookFunction(
        "model.decoder.layers.18",
        (None, None, None),
        partial(verify_fn, act_dict=activations),
    )
    model.register_forward_hook(retrieve_hook_fn)
    model.register_forward_hook(verify_hook_fn)

    _ = model(inputs)

    # Verify that the two verions of the same tensor are equal
    assert torch.equal(
        activations["save_ctx_activation"], activations["model.decoder.layers.12"],
    )
