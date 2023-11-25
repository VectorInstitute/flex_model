from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist

from flex_model.core import FlexModel, HookFunction

MODULE_NAME_1 = "model.decoder.layers.17.fc2"
MODULE_NAME_2 = "model.decoder.layers.18.fc2"
PROMPTS = [
    "It's a nice day we're having",
    "The capital of Canada is",
    "What should I eat for dinner tonight?",
    "There's about three people going to",
]


def test_register_forward_hook(make_opt_350m):
    """
    Tests if a hook function is registered correctly, and if the fields are set
    appropriately.
    """
    dist.init_process_group("nccl")
    model = make_opt_350m().cuda()

    activations = {}
    model = FlexModel(
        model,
        activations,
    )

    my_hook_function = HookFunction(MODULE_NAME_1)

    model.register_forward_hook(my_hook_function)

    assert my_hook_function._shared_state.output_ptr is activations
    assert my_hook_function._shared_state.save_ctx is model.save_ctx
    assert my_hook_function._shared_state.modules is model.trainable_modules

    dist.destroy_process_group()


def test_register_trainable_module(make_opt_350m):
    """
    Tests if a trainable module is registered correctly, and that all hook
    functions (regardless of when they're added), have a pointer to this
    module.
    """
    dist.init_process_group("nccl")
    model = make_opt_350m().cuda()

    activations = {}
    trainable_module = nn.Linear(420, 69, bias=False).cuda()
    model = FlexModel(model, activations)

    my_hook_function_1 = HookFunction(
        MODULE_NAME_1,
    )
    my_hook_function_2 = HookFunction(
        MODULE_NAME_2,
    )

    model.register_forward_hook(my_hook_function_1)
    model.register_trainable_module("test", trainable_module)
    model.register_forward_hook(my_hook_function_2)

    assert "test" in my_hook_function_1._shared_state.modules
    assert "test" in my_hook_function_2._shared_state.modules
    assert my_hook_function_1._shared_state.modules["test"] is trainable_module
    assert my_hook_function_2._shared_state.modules["test"] is trainable_module

    dist.destroy_process_group()


def test_trainable_module_gradient(make_opt_350m):
    dist.init_process_group("nccl")
    model = make_opt_350m().cuda()

    activations = {}
    fc = nn.Linear(1024, 1024, bias=False, dtype=model.dtype).cuda()
    model = FlexModel(model, activations)

    inputs = torch.randint(
        low=0,
        high=15000,
        size=(4, 64),
    ).cuda()

    model.register_trainable_module("test", fc)

    def _apply_test_fc(m, inputs, save_ctx, trainable_modules):
        return trainable_modules["test"](inputs)

    hook_func = HookFunction(
        MODULE_NAME_1, expected_shape=None, editing_function=_apply_test_fc
    )

    model.register_forward_hook(hook_func)

    outputs = model(inputs)
    loss = outputs.logits.mean()
    loss.backward()

    for n, p in model.named_parameters():
        assert p.grad is not None, f"Parameter: {n} has None grad field."
        assert (
            torch.count_nonzero(p.grad) != 0
        ), f"Parameter: {n} has all-zero grad field."

    dist.destroy_process_group()


def test_destroy(make_opt_350m):
    """
    Tests the destroy method to ensure everything is cleared appropriately.
    """
    dist.init_process_group("nccl")
    model = make_opt_350m().cuda()

    activations = {}
    trainable_module_1 = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    trainable_module_2 = nn.Linear(420, 69, bias=False).cuda().requires_grad_()
    model = FlexModel(
        model,
        activations,
    )
    model.register_trainable_module("test1", trainable_module_1)
    model.register_trainable_module("test2", trainable_module_2)

    my_hook_function = HookFunction(
        MODULE_NAME_1,
    )

    model.register_forward_hook(my_hook_function)
    model = model.module  # Calls finalizer.

    assert not isinstance(model, FlexModel)
    assert not hasattr(model, "hook_functions")
    assert not hasattr(model, "_hook_function_handles")
    assert not hasattr(model, "_hooks_active")
    assert not hasattr(model, "output_ptr")
    assert not hasattr(model, "save_ctx")
    assert not hasattr(model, "trainable_modules")

    hook_types = {"_forward", "_forward_pre", "_backward"}
    for m in model.modules():
        for hook_type in hook_types:
            attr = hook_type + "_hooks"
            assert len(getattr(m, attr)) == 0

    dist.destroy_process_group()


def test_save_ctx(make_opt_350m, opt_tokenizer):
    dist.init_process_group("nccl")
    model = make_opt_350m().cuda()

    tokenizer = opt_tokenizer

    activations = {}
    model = FlexModel(model, activations)

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ].cuda()

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
        "model.decoder.layers.12",
        editing_function=retrieve_fn,
    )
    verify_hook_fn = HookFunction(
        "model.decoder.layers.18",
        editing_function=partial(verify_fn, act_dict=activations),
    )
    model.register_forward_hook(retrieve_hook_fn)
    model.register_forward_hook(verify_hook_fn)

    _ = model(inputs)

    # Verify that the two verions of the same tensor are equal
    assert torch.equal(
        activations["save_ctx_activation"],
        activations["model.decoder.layers.12"][0],
    )

    dist.destroy_process_group()


def test_FlexModel_group_all(make_opt_350m):
    dist.init_process_group("nccl")
    model = make_opt_350m().cuda()

    activations = {}
    model = FlexModel(model, activations)

    layers = [
        f"model.decoder.layers.{i}"
        for i in range(len(model.module.model.decoder.layers))
    ]
    hook_functions = [HookFunction(name) for name in layers]
    for hf in hook_functions:
        model.register_forward_hook(hf)

    manager = model._hook_fn_group_manager
    assert len(manager.hook_fn_to_groups_map) == len(hook_functions)
    for hf, group in manager.hook_fn_to_groups_map.items():
        assert group == set(["all"])

    dist.destroy_process_group()


def test_FlexModel_group_creation(make_opt_350m, opt_tokenizer):
    dist.init_process_group("nccl")
    model = make_opt_350m().cuda()
    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]
    inputs = opt_tokenizer(prompts, padding=True, return_tensors="pt")[
        "input_ids"
    ].cuda()

    activations = {}
    model = FlexModel(model, activations)

    # Run the model forward pass on a group, on the hook functions not in the
    # group, and on all hook functions.
    model.create_hook_group(
        group_name="new_group",
        group_constructor="self_attn",
    )

    _ = model(inputs)

    all_group_tensors = {**activations}
    activations.clear()

    _ = model(inputs, groups="new_group")

    for name, tensor in activations.items():
        assert "self_attn" in name

    new_group_tensors = {**activations}
    activations.clear()

    _ = model(inputs, groups="new_group", complement=True)

    non_new_group_tensors = {**activations}
    activations.clear()

    assert len(all_group_tensors) == len(new_group_tensors) + len(
        non_new_group_tensors
    )
    for name, tensor in all_group_tensors.items():
        assert name in new_group_tensors or name in non_new_group_tensors
        if name in new_group_tensors:
            new_ten = new_group_tensors.pop(name)
            assert torch.allclose(tensor, new_ten)
        else:
            non_new_ten = non_new_group_tensors.pop(name)
            assert torch.allclose(tensor, non_new_ten)

    assert len(new_group_tensors) == 0
    assert len(non_new_group_tensors) == 0

    for (
        hook_fn,
        groups,
    ) in model._hook_fn_group_manager.hook_fn_to_groups_map.items():
        if "self_attn" in hook_fn.module_name:
            assert "new_group" in groups
        assert "all" in groups

    dist.destroy_process_group()
