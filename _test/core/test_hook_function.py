from argparse import Namespace
from functools import reduce

import torch
import torch.nn as nn

from flex_model.core import FlexModel, HookFunction
from flex_model.utils import setup_logger


# could be any MLP layer and the code won't break. The test doesn't generalize
# to other kinds of layers
MODULE_NAME = "model.decoder.layers.9.fc2"


def rgetattr(module, attr):
    def _getattr(module, attr):
        return getattr(module, attr)

    return reduce(_getattr, [module] + attr.split("."))


# Call fixture twice.
def test_hook_function(make_opt_350m, opt_tokenizer):
    """
    Tests if HookFunction implements a forward hook correctly
    """
    setup_logger("debug")

    model = make_opt_350m().cuda().eval()
    tokenizer = opt_tokenizer

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )["input_ids"].cuda()

    activations = {}

    # Regular hook fn impl.
    def regular_hook_fn(module, inputs, outputs):
        activations["regular"] = outputs.detach().cpu()
        return outputs * 2

    submodule = rgetattr(model, MODULE_NAME)
    handle = submodule.register_forward_hook(regular_hook_fn)

    gt_out = model(inputs).logits

    handle.remove()

    # FlexModel impl.
    model = FlexModel(model, activations)

    def editing_func(
        module,
        outputs,
        save_ctx: Namespace,
        trainable_modules: nn.ModuleDict,
    ) -> torch.Tensor:
        return outputs * 2

    my_hook_function = HookFunction(
        MODULE_NAME,
        expected_shape=None,
        editing_function=editing_func,
    )
    model.register_forward_hook(my_hook_function)

    fm_out = model(inputs).logits

    assert torch.allclose(gt_out, fm_out)

    assert torch.allclose(activations["regular"], activations[MODULE_NAME][0])
    assert activations[MODULE_NAME][0].device.type == "cpu"
