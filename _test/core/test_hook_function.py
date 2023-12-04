from functools import reduce

import torch
import torch.distributed as dist

from flex_model.core import FlexModel, HookFunction


# could be any MLP layer and the code won't break. The test doesn't generalize
# to other kinds of layers
MODULE_NAME = "model.decoder.layers.9.fc2"


# For getting submodule by name.
def rgetattr(module, attr):
    def _getattr(module, attr):
        return getattr(module, attr)

    return reduce(_getattr, [module] + attr.split("."))


def test_forward_hooks(make_opt_350m):
    dist.init_process_group("nccl")
    model = make_opt_350m().eval().cuda()

    inputs = torch.randint(0, 6400, size=(4, 32)).cuda()

    acc = {}

    # Test pytorch hook.
    def hook_fn(module, inputs, outputs):
        acc["torch"] = outputs.detach().cpu()
        return outputs * 2

    submodule_to_hook = rgetattr(model, MODULE_NAME)
    handle = submodule_to_hook.register_forward_hook(hook_fn)
    gt_retval = model(inputs).logits

    handle.remove()

    # Test flexmodel hook.
    def editing_fn(module, outputs, save_ctx, trainable_modules):
        return outputs * 2

    flexmodel = FlexModel(model, acc)
    flexmodel.register_forward_hook(
        HookFunction(MODULE_NAME, editing_function=editing_fn)
    )
    fm_retval = flexmodel(inputs).logits

    torch.testing.assert_close(gt_retval, fm_retval)
    torch.testing.assert_close(acc["torch"], acc[MODULE_NAME][0])
    dist.destroy_process_group()


def test_full_backward_hooks(make_opt_350m):
    dist.init_process_group("nccl")
    inputs = torch.randint(0, 6400, size=(4, 32)).cuda()

    acc = {}

    # Test pytorch hook.
    gt_grad_model = make_opt_350m().cuda()

    def hook_fn(module, grad_inputs, grad_outputs):
        acc["torch"] = grad_inputs[0].detach().cpu()
        grad_inputs = (grad_inputs[0] * 2, *grad_inputs[1:])
        return grad_inputs

    submodule_to_hook = rgetattr(gt_grad_model, MODULE_NAME)
    handle = submodule_to_hook.register_full_backward_hook(hook_fn)
    gt_retval = gt_grad_model(inputs).logits
    gt_retval.mean().backward()

    handle.remove()

    # Test flexmodel hook.
    fm_grad_model = make_opt_350m().cuda()

    def editing_fn(module, grad_inputs, save_ctx, trainable_modules):
        return grad_inputs * 2

    flexmodel = FlexModel(fm_grad_model, acc)
    flexmodel.register_full_backward_hook(
        HookFunction(MODULE_NAME, editing_function=editing_fn)
    )
    fm_retval = flexmodel(inputs).logits
    fm_retval.mean().backward()

    torch.testing.assert_close(gt_retval, fm_retval)
    torch.testing.assert_close(acc["torch"], acc[MODULE_NAME][0])
    for (gt_name, gt_grad), (fm_name, fm_grad) in zip(
        gt_grad_model.named_parameters(), fm_grad_model.named_parameters()
    ):
        assert gt_name == fm_name
        torch.testing.assert_close(gt_grad, fm_grad)

    dist.destroy_process_group()


def test_tensor_hooks(make_opt_350m):
    dist.init_process_group("nccl")
    inputs = torch.randint(0, 6400, size=(4, 32)).cuda()

    acc = {}

    # Test pytorch hook.
    gt_grad_model = make_opt_350m().cuda()

    def hook_fn(grad):
        acc["torch"] = grad.detach().cpu()
        return grad

    submodule_to_hook = rgetattr(gt_grad_model, f"{MODULE_NAME}.weight")
    handle = submodule_to_hook.register_hook(hook_fn)
    gt_retval = gt_grad_model(inputs).logits
    gt_retval.mean().backward()

    handle.remove()

    # Test flexmodel hook.
    fm_grad_model = make_opt_350m().cuda()

    def editing_fn(module, grad, save_ctx, trainable_modules):
        return grad * 2

    flexmodel = FlexModel(fm_grad_model, acc)
    flexmodel.register_hook(
        HookFunction(f"{MODULE_NAME}.weight", editing_function=editing_fn)
    )
    fm_retval = flexmodel(inputs).logits
    fm_retval.mean().backward()

    torch.testing.assert_close(gt_retval, fm_retval)
    torch.testing.assert_close(acc["torch"], acc[f"{MODULE_NAME}.weight"][0])
    for (gt_name, gt_grad), (fm_name, fm_grad) in zip(
        gt_grad_model.named_parameters(), fm_grad_model.named_parameters()
    ):
        assert gt_name == fm_name
        torch.testing.assert_close(gt_grad, fm_grad)

    dist.destroy_process_group()


def test_forward_pre_hooks(make_opt_350m):
    dist.init_process_group("nccl")
    model = make_opt_350m().eval().cuda()

    inputs = torch.randint(0, 6400, size=(4, 32)).cuda()

    acc = {}

    # Test pytorch hook.
    def hook_fn(module, inputs_):
        acc["torch"] = inputs_[0].detach().cpu()
        inputs_ = (inputs_[0] * 2, *inputs_[1:])
        return inputs_

    submodule_to_hook = rgetattr(model, MODULE_NAME)
    handle = submodule_to_hook.register_forward_pre_hook(hook_fn)
    gt_retval = model(inputs).logits

    handle.remove()

    # Test flexmodel hook.
    def editing_fn(module, inputs_, save_ctx, trainable_modules):
        return inputs_ * 2

    flexmodel = FlexModel(model, acc)
    flexmodel.register_forward_pre_hook(
        HookFunction(MODULE_NAME, editing_function=editing_fn)
    )
    fm_retval = flexmodel(inputs).logits

    torch.testing.assert_close(gt_retval, fm_retval)
    torch.testing.assert_close(acc["torch"], acc[MODULE_NAME][0])

    dist.destroy_process_group()


def test_full_backward_pre_hooks(make_opt_350m):
    dist.init_process_group("nccl")
    inputs = torch.randint(0, 6400, size=(4, 32)).cuda()

    acc = {}

    # Test pytorch hook.
    gt_grad_model = make_opt_350m().cuda()

    def hook_fn(module, grad_outputs):
        acc["torch"] = grad_outputs[0].detach().cpu()
        grad_outputs = (grad_outputs[0] * 2, *grad_outputs[1:])
        return grad_outputs

    submodule_to_hook = rgetattr(gt_grad_model, MODULE_NAME)
    handle = submodule_to_hook.register_full_backward_pre_hook(hook_fn)
    gt_retval = gt_grad_model(inputs).logits
    gt_retval.mean().backward()

    handle.remove()

    # Test flexmodel hook.
    fm_grad_model = make_opt_350m().cuda()

    def editing_fn(module, grad_outputs, save_ctx, trainable_modules):
        return grad_outputs * 2

    flexmodel = FlexModel(fm_grad_model, acc)
    flexmodel.register_full_backward_pre_hook(
        HookFunction(MODULE_NAME, editing_function=editing_fn)
    )
    fm_retval = flexmodel(inputs).logits
    fm_retval.mean().backward()

    torch.testing.assert_close(gt_retval, fm_retval)
    torch.testing.assert_close(acc["torch"], acc[MODULE_NAME][0])
    for (gt_name, gt_grad), (fm_name, fm_grad) in zip(
        gt_grad_model.named_parameters(), fm_grad_model.named_parameters()
    ):
        assert gt_name == fm_name
        torch.testing.assert_close(gt_grad, fm_grad)

    dist.destroy_process_group()
