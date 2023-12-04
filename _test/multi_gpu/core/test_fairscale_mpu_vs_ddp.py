from functools import partial

import fairscale.nn.model_parallel as mpu
import torch
import torch.distributed as dist

from flex_model.core import FlexModel
from _test.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry
import _test.multi_gpu.testing_utils as utils


(
    register_fairscale_megatron_test,
    get_fairscale_megatron_test,
) = make_test_registry(
    "fairscale_megatron",
    SlurmJobResourceSpec(),
)

TP_SIZE = 2
PP_SIZE = 1
HIDDEN = 10
VOCAB = 420
EXPANSION = 2
HOOK_ACTIVATIONS = {
    "register_forward_hook": {
        "vocab_parallel_embedding": None,
        "parallel_embedding": None,
        "column_parallel_linear": (None, None, HIDDEN * EXPANSION),
        "row_parallel_linear": None,
    },
    "register_full_backward_hook": {
        "vocab_parallel_embedding": None,
        "parallel_embedding": None,
        "column_parallel_linear": (None, None, HIDDEN * EXPANSION),
        "row_parallel_linear": None,
    },
    "register_hook": {
        "vocab_parallel_embedding.weight": (VOCAB, None),
        "parallel_embedding.weight": (None, HIDDEN),
        "column_parallel_linear.weight": (HIDDEN * EXPANSION, None),
        "row_parallel_linear.weight": (None, HIDDEN * EXPANSION),
    },
    "register_forward_pre_hook": {
        "vocab_parallel_embedding": None,
        "parallel_embedding": None,
        "column_parallel_linear": None,
        "row_parallel_linear": (None, None, HIDDEN * EXPANSION),
    },
    "register_full_backward_pre_hook": {
        "vocab_parallel_embedding": None,
        "parallel_embedding": None,
        "column_parallel_linear": (None, None, HIDDEN * EXPANSION),
        "row_parallel_linear": None,
    },
}


@register_fairscale_megatron_test
def test_fairscale():
    utils.init_fairscale_mpu(1, 1)

    model = utils.TestFairscaleModel(HIDDEN, VOCAB)
    inputs = torch.randint(0, 420, size=(4, 42)).cuda()

    outputs = model(inputs)

    utils.destroy_fairscale_mpu()

    utils.init_fairscale_mpu(TP_SIZE, PP_SIZE)

    sharded_inputs = inputs.chunk(mpu.get_data_parallel_world_size())[
        mpu.get_data_parallel_rank()
    ]
    ddp_tp_model = utils.TestFairscaleModel(HIDDEN, VOCAB)
    ddp_tp_model.copy_state_from_unsharded(model, other_tp_world_size=1)
    ddp_tp_model = utils.wrap_ddp(
        ddp_tp_model,
        pg=mpu.get_data_parallel_group(),
    )

    dist_outputs = ddp_tp_model(sharded_inputs)
    dist_outputs = utils.all_gather(
        dist_outputs, pg=mpu.get_data_parallel_group()
    )

    torch.testing.assert_close(outputs, dist_outputs)

    utils.destroy_fairscale_mpu()
    utils.print_success("test_fairscale")


def _run_ddp_model(register_fn, acc, run_backward=False):
    utils.init_fairscale_mpu(1, 1)

    # Base model.
    model = utils.TestFairscaleModel(HIDDEN, VOCAB)

    # DDP model.
    ddp_model = utils.wrap_ddp(model, pg=mpu.get_data_parallel_group())

    fm_ddp_model = FlexModel(
        ddp_model, acc, data_parallel_size=dist.get_world_size()
    )
    register_fn(fm_ddp_model)
    inputs = torch.randint(0, 420, size=(4, 42)).cuda()
    sharded_inputs = inputs.chunk(mpu.get_data_parallel_world_size())[
        mpu.get_data_parallel_rank()
    ]

    ddp_outputs = fm_ddp_model(sharded_inputs)
    if run_backward:
        ddp_outputs.mean().backward()
    ddp_outputs = utils.all_gather(
        ddp_outputs, pg=mpu.get_data_parallel_group()
    )

    return ddp_model, inputs, ddp_outputs


def _run_ddp_tp_model(model, inputs, register_fn, acc, run_backward=False):
    utils.init_fairscale_mpu(TP_SIZE, PP_SIZE)

    # TP + DDP model.
    ddp_tp_model = utils.TestFairscaleModel(HIDDEN, VOCAB)
    ddp_tp_model.copy_state_from_unsharded(model, other_tp_world_size=1)
    ddp_tp_model = utils.wrap_ddp(
        ddp_tp_model,
        pg=mpu.get_data_parallel_group(),
    )
    fm_ddp_tp_model = FlexModel(
        ddp_tp_model,
        acc,
        tensor_parallel_size=TP_SIZE,
        data_parallel_size=dist.get_world_size() // TP_SIZE,
    )
    register_fn(fm_ddp_tp_model)

    sharded_inputs = inputs.chunk(mpu.get_data_parallel_world_size())[
        mpu.get_data_parallel_rank()
    ]

    ddp_tp_outputs = fm_ddp_tp_model(sharded_inputs)
    if run_backward:
        ddp_tp_outputs.mean().backward()
    dist_outputs = utils.all_gather(
        ddp_tp_outputs, pg=mpu.get_data_parallel_group()
    )

    return ddp_tp_model, dist_outputs


def _run_ddp_and_ddp_tp_models(acc, edit_fn, hook_type, run_backward=False):
    register_fn = partial(
        utils.register_hook_functions,
        editing_function=edit_fn,
        hook_type=hook_type,
        module_name_to_shape_map=HOOK_ACTIVATIONS[hook_type],
        module_prefix="module.",
    )

    # DDP model outputs and states.
    ddp_model, inputs, ddp_outputs = _run_ddp_model(
        register_fn, acc, run_backward
    )
    ddp_model_states = ddp_model.module.get_unsharded_params_and_grads()

    # Reset MPU states.
    utils.destroy_fairscale_mpu()

    # DDP + TP model outputs and states.
    ddp_tp_model, ddp_tp_outputs = _run_ddp_tp_model(
        ddp_model.module, inputs, register_fn, acc, run_backward
    )
    ddp_tp_model_states = ddp_tp_model.module.get_unsharded_params_and_grads()

    return ddp_model_states, ddp_tp_model_states, ddp_outputs, ddp_tp_outputs


@register_fairscale_megatron_test
def test_forward_hooks_fairscale():
    # Fm params.
    acc = {}

    def _edit(module, outputs, save_ctx, trainable_modules):
        return outputs * 2

    _, _, ddp_outputs, ddp_tp_outputs = _run_ddp_and_ddp_tp_models(
        acc,
        _edit,
        hook_type="register_forward_hook",
    )

    # Validate.
    torch.testing.assert_close(ddp_outputs, ddp_tp_outputs)

    for acts in acc.values():
        torch.testing.assert_close(acts[0], acts[1])

    utils.destroy_fairscale_mpu()
    utils.print_success("test_forward_hooks_fairscale")


@register_fairscale_megatron_test
def test_full_backward_hooks_fairscale():
    # Fm params.
    acc = {}

    def _edit(module, grad_inputs, save_ctx, trainable_modules):
        return grad_inputs * 2

    (
        ddp_model_states,
        ddp_tp_model_states,
        ddp_outputs,
        ddp_tp_outputs,
    ) = _run_ddp_and_ddp_tp_models(
        acc,
        _edit,
        hook_type="register_full_backward_hook",
        run_backward=True,
    )

    # Validate.
    torch.testing.assert_close(ddp_outputs, ddp_tp_outputs)

    utils.assert_same_state(ddp_model_states, ddp_tp_model_states)

    utils.destroy_fairscale_mpu()
    utils.print_success("test_full_backward_hooks_fairscale")


@register_fairscale_megatron_test
def test_forward_pre_hooks_fairscale():
    # Fm params.
    acc = {}

    def _edit(module, inputs, save_ctx, trainable_modules):
        return torch.where(inputs < 200, inputs + 10, inputs)

    _, _, ddp_outputs, ddp_tp_outputs = _run_ddp_and_ddp_tp_models(
        acc,
        _edit,
        hook_type="register_forward_pre_hook",
    )

    # Validate.
    torch.testing.assert_close(ddp_outputs, ddp_tp_outputs)

    for acts in acc.values():
        torch.testing.assert_close(acts[0], acts[1])

    utils.destroy_fairscale_mpu()
    utils.print_success("test_forward_pre_hooks_fairscale")


@register_fairscale_megatron_test
def test_full_backward_pre_hooks_fairscale():
    # Fm params.
    acc = {}

    def _edit(module, grad_outputs, save_ctx, trainable_modules):
        return grad_outputs * 2

    (
        ddp_model_states,
        ddp_tp_model_states,
        ddp_outputs,
        ddp_tp_outputs,
    ) = _run_ddp_and_ddp_tp_models(
        acc,
        _edit,
        hook_type="register_full_backward_hook",
        run_backward=True,
    )

    # Validate.
    torch.testing.assert_close(ddp_outputs, ddp_tp_outputs)

    utils.assert_same_state(ddp_model_states, ddp_tp_model_states)

    utils.destroy_fairscale_mpu()
    utils.print_success("test_full_backward_pre_hooks_fairscale")
