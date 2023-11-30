import copy
from functools import partial

import torch
import torch.distributed as dist

from flex_model.core import FlexModel
import _test.multi_gpu.testing_utils as utils
from _test.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry


register_wrapper_test, get_wrapper_test = make_test_registry(
    "wrappers",
    SlurmJobResourceSpec(),
)


MODEL_NAME = "test_model"
MODULE_NAME = "fc1"
MODULE_SHAPE = None
FSDP_WRAP_LAYER = torch.nn.Linear


WRAPPERS = {
    "ddp": utils.wrap_ddp,
    "fsdp": partial(utils.wrap_fsdp, layer_to_wrap=FSDP_WRAP_LAYER),
}


def _setup_model_and_inputs(wrap_fn):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Construct single-gpu (regular) and multi-gpu (ddp) models.
    base_model = utils.TestModel()

    wrapped_model = copy.deepcopy(base_model)
    wrapped_model = wrap_fn(wrapped_model, pg=None)

    # Dummy inputs.
    base_inputs = torch.randn(4, 10, device="cuda", dtype=torch.float32)
    wrapped_inputs = base_inputs.chunk(world_size)[rank]

    # Set requires grad for backward hooks testing.
    base_inputs.requires_grad = True
    wrapped_inputs.requires_grad = True

    return base_model, base_inputs, wrapped_model, wrapped_inputs


def _run_base_model(base_model, base_inputs, register_fn, acc):
    rank = dist.get_rank()
    base_pg = dist.new_group([rank])

    fm_base_model = FlexModel(base_model, acc, process_group=base_pg)
    register_fn(fm_base_model)

    base_outputs = fm_base_model(base_inputs)

    return base_outputs


def _run_wrapped_model(wrapped_model, wrapped_inputs, register_fn, acc):
    world_size = dist.get_world_size()

    # Use default pg for whole world.
    fm_base_model = FlexModel(wrapped_model, acc, data_parallel_size=world_size)
    register_fn(fm_base_model)

    wrapped_outputs = fm_base_model(wrapped_inputs)

    return wrapped_outputs


def _run_base_and_wrapped_models(
    acc, edit_fn, hook_type, wrap_fn, run_backward=False
):
    (
        base_model,
        base_inputs,
        wrapped_model,
        wrapped_inputs,
    ) = _setup_model_and_inputs(wrap_fn)

    register_fn = partial(
        utils.register_hook_functions,
        editing_function=edit_fn,
        hook_type=hook_type,
        module_name_to_shape_map={MODULE_NAME: MODULE_SHAPE},
    )

    # Regular model.
    base_register_fn = partial(register_fn, module_prefix="")
    base_outputs = _run_base_model(
        base_model,
        base_inputs,
        base_register_fn,
        acc,
    )
    if run_backward:
        base_outputs.mean().backward()

    # Wrapped model.
    wrapped_register_fn = partial(register_fn, module_prefix="module.")
    wrapped_outputs = _run_wrapped_model(
        wrapped_model,
        wrapped_inputs,
        wrapped_register_fn,
        acc,
    )
    if run_backward:
        wrapped_outputs.mean().backward()
    wrapped_outputs = utils.all_gather(wrapped_outputs, dim=0)

    return base_outputs, wrapped_outputs, base_model, wrapped_model


@register_wrapper_test
def test_forward_hooks_wrapped():
    utils.init_process_group()

    acc = {}

    def _edit(module, outputs, save_ctx, trainable_modules):
        return outputs * 2

    for wrapper_name, wrapper_fn in WRAPPERS.items():
        acc[wrapper_name] = {}
        base_outputs, wrapped_outputs, _, _ = _run_base_and_wrapped_models(
            acc[wrapper_name],
            _edit,
            "register_forward_hook",
            wrapper_fn,
        )

        # Validation.
        torch.testing.assert_close(base_outputs, wrapped_outputs)
        torch.testing.assert_close(
            acc[wrapper_name][MODULE_NAME][0],
            acc[wrapper_name][f"module.{MODULE_NAME}"][0],
        )
        utils.print_success(f"test_forward_hooks [{wrapper_name}]")


@register_wrapper_test
def test_full_backward_hooks_wrapped():
    utils.init_process_group()

    acc = {}

    def _edit(module, grad_inputs, save_ctx, trainable_modules):
        return grad_inputs * 2

    for wrapper_name, wrapper_fn in WRAPPERS.items():
        acc[wrapper_name] = {}
        (
            base_outputs,
            wrapped_outputs,
            base_model,
            wrapped_model,
        ) = _run_base_and_wrapped_models(
            acc[wrapper_name],
            _edit,
            "register_full_backward_hook",
            wrapper_fn,
            run_backward=True,
        )

        # Validate.
        torch.testing.assert_close(base_outputs, wrapped_outputs)

        torch.testing.assert_close(
            acc[wrapper_name][MODULE_NAME][0],
            acc[wrapper_name][f"module.{MODULE_NAME}"][0],
        )

        if wrapper_name == "ddp":
            for (base_name, base_param), (wrapped_name, wrapped_param) in zip(
                base_model.named_parameters(), wrapped_model.named_parameters()
            ):
                torch.testing.assert_close(base_param, wrapped_param)
                torch.testing.assert_close(base_param.grad, wrapped_param.grad)

        utils.print_success(f"test_full_backward_hooks [{wrapper_name}]")


@register_wrapper_test
def test_forward_pre_hooks_wrapped():
    utils.init_process_group()

    acc = {}

    def _edit(module, inputs, save_ctx, trainable_modules):
        return inputs * 2

    for wrapper_name, wrapper_fn in WRAPPERS.items():
        acc[wrapper_name] = {}
        base_outputs, wrapped_outputs, _, _ = _run_base_and_wrapped_models(
            acc[wrapper_name],
            _edit,
            "register_forward_hook",
            wrapper_fn,
        )

        # Validation.
        torch.testing.assert_close(base_outputs, wrapped_outputs)
        torch.testing.assert_close(
            acc[wrapper_name][MODULE_NAME][0],
            acc[wrapper_name][f"module.{MODULE_NAME}"][0],
        )
        utils.print_success(f"test_forward_pre_hooks [{wrapper_name}]")


@register_wrapper_test
def test_full_backward_pre_hooks_wrapped():
    utils.init_process_group()

    acc = {}

    def _edit(module, grad_outputs, save_ctx, trainable_modules):
        return grad_outputs * 2

    for wrapper_name, wrapper_fn in WRAPPERS.items():
        acc[wrapper_name] = {}
        (
            base_outputs,
            wrapped_outputs,
            base_model,
            wrapped_model,
        ) = _run_base_and_wrapped_models(
            acc[wrapper_name],
            _edit,
            "register_full_backward_hook",
            wrapper_fn,
            run_backward=True,
        )

        # Validate.
        torch.testing.assert_close(base_outputs, wrapped_outputs)

        torch.testing.assert_close(
            acc[wrapper_name][MODULE_NAME][0],
            acc[wrapper_name][f"module.{MODULE_NAME}"][0],
        )

        if wrapper_name == "ddp":
            for (base_name, base_param), (wrapped_name, wrapped_param) in zip(
                base_model.named_parameters(), wrapped_model.named_parameters()
            ):
                torch.testing.assert_close(base_param, wrapped_param)
                torch.testing.assert_close(base_param.grad, wrapped_param.grad)

            utils.print_success(
                f"test_full_backward_pre_hooks [{wrapper_name}]"
            )
