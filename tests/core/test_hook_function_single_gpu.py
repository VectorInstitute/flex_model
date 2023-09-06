from flex_model.core import FlexModel, HookFunction
import torch
import torch.nn as nn
from typing import Dict
from argparse import Namespace
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from tests.registry import register_test


# could be any MLP layer and the code won't break. The test doesn't generalize
# to other kinds of layers
MODULE_NAME = "model.layers.27.mlp"


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


def inject_module(injection_layer: nn.Module, model: nn.Module) -> None:
    """Replace <MODULE_NAME> in <model> with <injection_layer>"""
    sliced = MODULE_NAME.split(".")
    layer_num = int(sliced[2])
    model.model.layers[layer_num].mlp = injection_layer


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


def make_model_and_tokenizer():
    """Helper function to construct a llama-2 model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "/model-weights/Llama-2-13b-hf/",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/model-weights/Llama-2-13b-hf/",
        local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 128

    return model, tokenizer


@register_test
def test_hook_function():
    """
    Tests if HookFunction implements a forward hook correctly
    """
    model, tokenizer = make_model_and_tokenizer()
    model.cuda().eval()  # disable dropout

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    inputs = tokenizer(
        prompts,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"].cuda()

    # first get our ground truth activations
    inject_layer = LayerInjection(MODULE_NAME, model)
    inject_module(inject_layer, model)
    model(inputs)

    ground_truth = inject_layer.modified_out.cpu()

    # now try with HookFunction API
    model, tokenizer = make_model_and_tokenizer()
    model.cuda().eval()

    # ensure the layer injection is out
    for _, m in model.named_modules():
        assert not isinstance(m, LayerInjection)

    activations = {}
    model = FlexModel(
        model,
        activations,
    )

    my_hook_function = HookFunction(
        MODULE_NAME,
        expected_shape=(None, None, None),
        editing_function=partial(editing_func, in_dict=activations),
    )

    model.register_hook_function(my_hook_function)
    model(inputs)

    assert torch.equal(ground_truth, activations["modified"])


test_hook_function()
