# FlexModel
[![Documentation Status](https://readthedocs.org/projects/flexmodel/badge/?version=latest)](https://flexmodel.readthedocs.io/en/latest/)

## Current WIPs
* Adding more rigorous tess for hook function behaviours
* Implement strategies for hanlding distributed `save_ctx` and `trainable_modules`
* Visualizations of model architecture showing where hooks can be placed
* Visualizations of model architecture showing where hooks are currently
located, and what groups they are tagged in
* Editing function presets
* Distributed debugging tools and user-provided editing function parsing
* Remove the need to pass an `expected_shape`

## Introduction

`FlexModel` is a tool designed for distributed interpretability of Large
Language Models (LLMs). `FlexModel` allows you to retrieve and/or edit
**unsharded** activations within your LLM (among other things).

For more detailed information, checkout our [docs](https://flexmodel.readthedocs.io/en/latest/)
or read the paper (coming soon!). There are also a myriad of examples/demos
to showcase what you can do with `FlexModel`. Feel free to raise a github issue
for new features/bugs!

## Introduction: The `FlexModel` Wrapper
`FlexModel` wraps any `nn.Module`, and replaces the typical PyTorch
`nn.Module` hook registration functions. It contains all the state necessary
for doing model surgery, while leaving the wrapped module invariant.

## Introduction: The `HookFunction`
The replaced hook registration functions now receive `HookFunction` instances
as input. It is the `HookFunction`'s job to retrieve activations and/or
edit them within the wrapped model. To edit an activation (which will affect
subsequent model operation), you can simply provide your `HookFunction` with
an editing function. The best part is that the editing function can contain
arbitrary code, and runs single-threaded. So you don't have to worry about any
SPMD parallelism in your editing function!

## What's a hook function?
PyTorch exposes endpoints in each `torch.nn.Module`, which calls your
"hook function" at a specified time during module operation. There's a great
introductory blogpost
[here](https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/).

## Why not just use PyTorch hooks?
Vanilla PyTorch hooks work great for single-gpu/process models. However if you
need access to full activations for retrieval/editing, then you'll need to
figure out how to unshard them in each hook function. Given the parallelism
dimensions, `FlexModel` can figure out which collectives to call if necessary,
so your activations are always unsharded. For example, `FlexModel` integrates
simply with distributed frameworks like DDP, FSDP, Fairscale Megatron and
Megatron-LM.

## What can I hook?
You can attach hooks to anything which native PyTorch would allow you to
hook into! `FlexModel` simply intercepts the `nn.Module` hook function
registration API to inject our own logic. Concretely, we support the following
hook function registration functions:
* `nn.Module.register_forward_hook(...)`: [Usage](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
* `nn.Module.register_full_backward_hook(...)`: [Usage](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook)
* `torch.Tensor.register_hook(...)`: [Usage](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch-tensor-register-hook)
* `nn.Module.register_forward_pre_hook(...)`: [Usage](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)
* `nn.Module.register_full_backward_pre_hook(...)`: [Usage](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook)


# Installation
Run `pip install -e .` from the root directory. PyPi package coming soon with
full release before Neurips 2023!


# Usage
Here's a short example on how you would use the FlexModel and HookFunction
classes. When using distributed models using FSDP or Megatron layers, the
`FlexModel` class requires specification of data parallel (DP), tensor parallel
(TP), and pipeline parallel (PP) sizes.
```python
from torch.distributed.fsdp import FullyShardedDataParallel
from flex_model import FlexModel, HookFunction

# Load some model
model = FSDP(model, ...)
inputs = ...

# Activations will be dumped here
output_dict = {}

# Wrap the model
model = FlexModel(
    model,
    output_dict,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=4, # For FSDP over 4 GPUs
)

# Only need to provide a shape hint on the dimension which may be sharded
# across gpus.
# (batch, sequence, hidden) -> sharded along hidden
expected_shape = (None, None, hidden_dim)

# Define a hook function
def editing_function(current_module, inputs, save_ctx, modules) -> Tensor
    # Save input activation tensor for later use
    save_ctx.activation = inputs

    # Edit the activation tensor
    edited_activations = inputs * 2

    # Apply torch.nn.Modules to the activation tensor (can generate grads)
    edited_activations = modules["mlp"].forward(edited_activations)

    return edited_activations

# Create hook function
hook_function = HookFunction(
    module_name="layers.7.feed_forward.w1",
    expected_shape=expected_shape,
    editing_function=editing_function,
)

# Register the hook function into our model
model.register_forward_hook(hook_function)

# Run a forward pass, activations will be placed in the output_dict
model.forward(inputs)
```


# `HookFunction` Groups
`HookFunction`s can be associated with group tags. Using these tags, you can
choose which groups are run during a given forward pass. There are two primary
ways of interacting with `HookFunction` groups:
1. `FlexModel.create_hook_group`: This function creates a collection of uniform
`HookFunction` instances, and tags them under the same group name. Let's
inspect the function signature:
```python
def create_hook_group(
    self,
    group_name: str,
    group_constructor: str,
    expected_shape: Optional[Tuple[Optional[int], ...]] = None,
    editing_function: Optional[Callable] = None,
    unpack_idx: Optional[int] = 0,
```
- `group_name`: Name of the group to tag the created `HookFunctions` under.
- `group_constructor`: String pattern which is used to match against
submodule names. For example, setting this to "self_attn" will match any
submodule with "self_attn" `in` its name. If 10 submodules match this, then
10 `HookFunction` instances will be created and registered on its respective
submodule.
- `expected_shape`, `editing_function` and `unpack_idx` will all be the
same for each `HookFunction` created.
2. `FlexModel.update_hook_groups`: This function updates the group tags for
existing `HookFunction` instances already registered. It takes either a list of
`HookFunction`s to tag, a single `HookFunction` to tag, or a string to pattern-
match against submodules to automatically tag any associated `HookFunction`s.

## Enabling/Disabling Certain Groups
Note that `HookFunction` groups follow `set` semantics. When running forward passes, all
`HookFunction`s are enabled by default (ie. all `HookFunction`s are members of
the `all` group). Specifying the groups to run as a list of strings in the
models's forward function will enable the union set of `HookFunction`s withing
the groups. You can also enable the `complement` argument, which will enable
all hooks **not** in the union set.

## Adding/Removing Group Tags
Each `HookFunction` instance can be tagged in as many groups as you'd like.
`HookFunction`s can also be removed from groups via `remove_hook_groups` with
similar semantics to the `update_hook_groups` method.

Note that you **cannot** remove the `all` group tag from any `HookFunction`
instance, which will cause an exception.


# Running Tests
Running single-gpu tests from the project folder using `pytest` can be done with
the command:
```
torchrun --nnodes 1 --nproc_per_node -m pytest --ignore=_test/multi_gpu _test/
```

Multi-gpu tests are run via `submitit` on a `slurm` cluster. Navigate to
`_test/multi_gpu` and run the command:
```
python run_multi_gpu_tests_slurm.py
```
The multi-gpu tests require 4 GPUs to run.


# Important Notes
- Make sure to replace any instances of `module.forward(inputs)` with
`module(inputs)`. The forward hooks are not run by PyTorch if you directly call
the forward function of a module (this is the case with LLaMA).
- If you would like to create `HookFunction` entrypoints arbitrarily in the
wrapped model, you can place `DummyModule`s with identity forward functions
which can be hooked into. `DummyModule` is located in the `core/core_utils.py`
file.
