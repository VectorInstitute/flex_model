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
    """
    Purpose is to replace an MLP module in the model. In short, this class
    allows us to store and modify an incoming activation without using any
    sort of forward hooks. It acts as the absolute ground truth for later
    comparison.
    """

    def __init__(self, module_name: str, model: nn.Module):
        super(LayerInjection, self).__init__()
        self.original_out: torch.Tensor = None
        self.modified_out: torch.Tensor = None
        self.decoder_layer: nn.Module = None

        for n, m in model.named_modules():
            if n == module_name:
                self.decoder_layer = m

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        out = self.decoder_layer(*args, **kwargs)
        self.original_out = out.detach()
        self.modified_out = self.original_out * 2
        return out


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
    """
    Passed into HookFunction. Takes the input tensor, multiplies it by 2, saves it and
    returns the input tensor.
    """
    multiplied = inputs * 2
    in_dict["modified"] = multiplied.detach().cpu()
    return inputs


# Call fixture twice.
opt_350m_gt = opt_350m
opt_350m_hook = opt_350m


def test_hook_function(opt_350m_gt, opt_350m_hook, opt_tokenizer):
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
    model(inputs)

    ground_truth = inject_layer.modified_out.cpu()

    # now try with HookFunction API
    model = opt_350m_hook.cuda().eval()

    # ensure the layer injection is out
    for _, m in model.named_modules():
        assert not isinstance(m, LayerInjection)

    activations = {}
    model = FlexModel(model, activations,)

    my_hook_function = HookFunction(
        MODULE_NAME,
        expected_shape=(None, None, None),
        editing_function=partial(editing_func, in_dict=activations),
    )

    model.register_hook_function(my_hook_function)
    model(inputs, with_hooks=True)

    assert torch.equal(ground_truth, activations["modified"])
