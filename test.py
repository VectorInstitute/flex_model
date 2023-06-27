from functools import partial
import logging
from typing import Tuple, Dict, List, Optional, Any, Callable

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import torch.nn as nn
from torch import Tensor

from flex_model import HookFunctionTriple, FlexModel
from utils import _recursively_find_first_tensor


logger = logging.getLogger(__name__)


# TODO: Break tests into smaller bite-sized chunks and prepare for pytest


class Model(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = in_dim * 4
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, self.latent_dim)
        self.nonlin = nn.ReLU()
        self.fc2 = nn.Linear(self.latent_dim, out_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        y = self.fc1(inputs)
        y = self.nonlin(y)
        y = self.fc2(y)
        return y


def print_named_modules(model: nn.Module) -> None:
    assert isinstance(model, nn.Module)
    names = [n for n, _ in model.named_modules()]
    print(names)


def apply_torch_fwd_hooks(
    model: nn.Module,
    inputs: Tensor,
    module_names: List[str],
    shapes: Optional[List[Tuple[int]]] = None,
    parse_fn: Optional[Callable] = None,
) -> Dict[str, Tensor]:
    """Retrieve activation using vanilla pytorch forward hooks."""
    module_dict = {name: shape for name, shape in zip(module_names, shapes)}

    def _fwd_hook(
        registered_name: str,
        return_dict: Dict[str, Tensor],
        module: nn.Module,
        inputs: Any,
        outputs: Any,
        parse_fn: Optional[Callable] = None,
        shape: Optional[Tuple[int]] = None,
    ) -> None:
        if parse_fn is not None:
            outputs = parse_fn(outputs)

        res = outputs.detach().cpu()

        if shape is not None:
            res.reshape(*shape)

        return_dict[registered_name] = res

    output_dict: Dict[str, Tensor] = {}
    hook_handles = []
    for name, module in model.named_modules():
        if name in module_dict:
            handle = module.register_forward_hook(
                partial(
                    _fwd_hook,
                    name,
                    output_dict,
                    parse_fn=parse_fn if parse_fn else lambda x: x,
                    shape=module_dict[name],
                )
            )
            hook_handles.append(handle)

            logger.info(f"Installing module: {name}")

    logger.info("Running forward")
    _ = model.forward(inputs)

    for handle in hook_handles:
        handle.remove()

    return output_dict


def apply_flex_model_fwd_hooks(
    model: nn.Module,
    inputs: Tensor,
    module_names: List[str],
    shapes: List[Tuple[int]],
) -> Dict[str, Tensor]:
    """Retrieve activaations using flex model forward hooks."""
    output_dict: Dict[str, Tensor] = {}
    flex_model = FlexModel(model, output_dict)
    for name, shape in zip(module_names, shapes):
        hfs = HookFunctionTriple(
            module_name=name,
            shape=shape,
            editing_fn=lambda x: x,
        )
        flex_model.register_hook_function_triple(hfs)

    flex_model.forward(inputs)

    return output_dict


def compare_tensor_dicts(
    dict1: Dict[str, Tensor],
    dict2: Dict[str, Tensor],
) -> bool:
    """Check equality between two dicts."""
    _pack = zip(dict1.items(), dict2.items())
    for (d1_name, d1_act), (d2_name, d2_act) in _pack:
        if d1_name != d2_name:
            return False
        if not torch.allclose(d1_act, d2_act):
            return False

    return True


def test_simple_model():
    # Regular fwd pass
    model = Model(16, 32).cuda()

    inputs = torch.randn(8, 4, 16).cuda()
    outputs = model(inputs)

    # Test forward hooks
    module_names = ["fc1"]
    module_shapes = [(8 * 4, 16 * 4)]

    # Vanilla pytorch forward hooks
    logger.info("Running base forward hooks")
    test_base_dict = apply_torch_fwd_hooks(
        model,
        inputs,
        module_names,
        module_shapes,
    )

    # Flex model forward hooks
    logger.info("Running flex model forward hooks")
    test_flex_dict = apply_flex_model_fwd_hooks(
        model,
        inputs,
        module_names,
        module_shapes,
    )

    # Shape dumped is different, reshape back for comparison
    test_flex_dict["fc1"] = test_flex_dict["fc1"].reshape(8, 4, 16 * 4)

    # Correctness check
    assert compare_tensor_dicts(test_base_dict, test_flex_dict)


# TODO: More model test cases and editing functions
def test_huggingface_opt_model():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    prompt = "Hi I'm Matt, where am I?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Regular forward pass
    outputs = model.forward(inputs.input_ids)

    # Test forward hooks
    module_names = [
        "model.decoder.layers.5.self_attn",
        "model.decoder.layers.6.self_attn.out_proj",
        "model.decoder.layers.2.activation_fn",
        "model.decoder.layers.7.self_attn.v_proj",
        "model.decoder.layers.11.self_attn.k_proj",
        "model.decoder.layers.8.self_attn.q_proj",
        "model.decoder.layers.11.self_attn_layer_norm",
        "model.decoder.layers.7.fc1",
        "model.decoder.embed_tokens",
        "model.decoder.embed_positions",
        "model",
        "model.decoder",
        "model.decoder.layers",
        "model.decoder.layers.7" "model.decoder.layers.11.fc2",
        "lm_head",
    ]
    module_shapes = [None for n in range(len(module_names))]

    def _huggingface_parse_fn(x):
        """Hard-coded parse function for huggingface models.

        Note: Retrieval only.
        """
        if isinstance(x, tuple):
            return x[0]
        if isinstance(x, BaseModelOutputWithPast):
            return x.last_hidden_state
        return x

    logger.info("Running base forward hooks")
    test_base_dict = apply_torch_fwd_hooks(
        model,
        inputs.input_ids,
        module_names,
        module_shapes,
        _huggingface_parse_fn,
    )

    logger.info("Running flex model forward hooks")
    test_flex_dict = apply_flex_model_fwd_hooks(
        model,
        inputs.input_ids,
        module_names,
        module_shapes,
    )

    assert compare_tensor_dicts(test_base_dict, test_flex_dict)


def test_traversal_utils():
    from utils import _recursively_find_first_tensor

    target_tensor = torch.randn(2, 2)
    outputs = (
        (1, 3),
        ["abs", "sd", 2],
        [[target_tensor, torch.randint(10, size=(3, 3))], (2, 3)],
        (6, 5, 4, "123", nn.Linear(10, 20), torch.randn(3, 3)),
    )

    tensor = _recursively_find_first_tensor(outputs)

    assert torch.equal(tensor, target_tensor)

    from utils import _flatten, _unflatten

    # TODO: More test cases here
    outputs = [
        1,
        2,
        (torch.randn(2, 2), 3),
        "ab",
        (2, torch.randn(3, 3)),
    ]
    treedef, leaves = _flatten(outputs)

    leaves = [l * 10 for l in leaves]

    new_outputs = _unflatten(treedef, leaves)

    new_treedef, _ = _flatten(new_outputs)

    assert treedef == new_treedef


def main():
    logger.info("Testing simple PyTorch model...")
    test_simple_model()
    logger.info("Test successful!")

    logger.info("Testing traversal/parse utils...")
    test_traversal_utils()
    logger.info("Test successful!")

    logger.info("Testing Huggingface OPT-125m single gpu...")
    test_huggingface_opt_model()
    logger.info("Test successful!")


if __name__ == "__main__":
    main()
