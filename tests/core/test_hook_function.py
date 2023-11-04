from argparse import Namespace
from functools import partial, reduce
from typing import Dict

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from flex_model.core import FlexModel, HookFunction
from tests.fixtures import opt_350m, opt_tokenizer

# could be any MLP layer and the code won't break. The test doesn't generalize
# to other kinds of layers
MODULE_NAME = "model.decoder.layers.9.fc2"


def rgetattr(module, attr):
    def _getattr(module, attr):
        return getattr(module, attr)

    return reduce(_getattr, [module] + attr.split("."))


def rsetattr(module, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(module, pre) if pre else module, post, val)


class LayerInjection(nn.Module):
    def __init__(self, module_name: str, model: nn.Module):
        super(LayerInjection, self).__init__()
        self.original_out: torch.Tensor = None
        self.saved_out: torch.Tensor = None
        self.decoder_layer: nn.Module = None

        for n, m in model.named_modules():
            if n == module_name:
                self.decoder_layer = m

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        out = self.decoder_layer(*args, **kwargs)
        self.original_out = out.detach()
        self.saved_out = self.original_out
        return self.original_out * 2


def inject_module(injection_layer: nn.Module, model: nn.Module) -> nn.Module:
    """Replace <MODULE_NAME> in <model> with <injection_layer>"""
    rsetattr(model, MODULE_NAME, injection_layer)


def editing_func(
    current_module: nn.Module,
    inputs: torch.Tensor,
    save_ctx: Namespace,
    modules: nn.ModuleDict,
    in_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    multiplied = inputs * 2
    return multiplied


# Call fixture twice.
opt_350m_gt = opt_350m
opt_350m_hook = opt_350m


@pytest.mark.parametrize("offload_mode", ["CPU", "GPU"])
def test_hook_function(opt_350m_gt, opt_350m_hook, opt_tokenizer, offload_mode):
    """
    Tests if HookFunction implements a forward hook correctly
    """
    model = opt_350m_gt.cuda().eval()
    tokenizer = opt_tokenizer

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    inputs = tokenizer(prompts, padding=True, return_tensors="pt",)["input_ids"].cuda()

    # first get our ground truth activations
    inject_layer = LayerInjection(MODULE_NAME, model).to(model.dtype).cuda()
    inject_module(inject_layer, model)
    gt_out = model(inputs)

    ground_truth = inject_layer.saved_out.cpu()

    # now try with HookFunction API
    model = opt_350m_hook.cuda().eval()

    # ensure the layer injection is out
    for _, m in model.named_modules():
        assert not isinstance(m, LayerInjection)

    activations = {}
    model = FlexModel(model, activations, offload_mode=offload_mode)

    my_hook_function = HookFunction(
        MODULE_NAME,
        expected_shape=(None, None, None),
        editing_function=partial(editing_func, in_dict=activations),
    )

    model.register_forward_hook(my_hook_function)
    fm_out = model(inputs)

    assert torch.allclose(gt_out.logits, fm_out.logits)

    if offload_mode == "CPU":
        assert torch.allclose(ground_truth.cpu(), activations[MODULE_NAME])
        assert activations[MODULE_NAME].device.type == "cpu"
    else:
        assert torch.allclose(ground_truth.cuda(), activations[MODULE_NAME])
        assert activations[MODULE_NAME].device.type == "cuda"
