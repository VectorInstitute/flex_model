from functools import partial
import logging
import os
from typing import Tuple, Dict, List, Optional, Any, Callable

from transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import torch.nn as nn
from torch import Tensor

from flex_model import HookFunctionTriple, FlexModel, DistributedFlexModel
from utils import _recursively_find_first_tensor, print_rank0
from testing_utils import (
    parse_base_model_output_with_past,
    SimpleModel,
    compare_tensor_dicts,
    apply_torch_fwd_hooks,
    apply_flex_model_fwd_hooks,
    get_opt_125m,
)


logger = logging.getLogger(__name__)


def test_simple_model():
    # Regular fwd pass
    model = SimpleModel(16, 32).cuda()

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



def test_huggingface_opt_model():
    model, tokenizer = get_opt_125m()

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

    logger.info("Running base forward hooks")
    test_base_dict = apply_torch_fwd_hooks(
        model,
        inputs.input_ids,
        module_names,
        module_shapes,
        parse_base_model_output_with_past,
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


# TODO: Be more consistent with logging messages
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
