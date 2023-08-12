from functools import partial
import logging
import os
from typing import Tuple, Dict, List, Optional, Any, Callable

from transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import torch.nn as nn
from torch import Tensor

from flex_model.model_wrappers import FlexModel, setup_logger
from flex_model._traverse_utils import (
    _recursively_find_first_tensor,
    _flatten,
    _unflatten,
)
from flex_model.tests.testing_utils import (
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
    module_names_with_shapes = {
        "fc1": (8, 4, 16 * 4),
    }

    # Vanilla pytorch forward hooks
    logger.info("Running base forward hooks")
    test_base_dict, _ = apply_torch_fwd_hooks(
        model,
        inputs,
        module_names_with_shapes,
    )

    # Flex model forward hooks
    logger.info("Running flex model forward hooks")
    
    test_flex_dict, _ = apply_flex_model_fwd_hooks(
        model,
        inputs,
        module_names_with_shapes,
    )

    # Correctness check
    for (n1, a1), (n2, a2) in zip(test_base_dict.items(), test_flex_dict.items()):
        assert n1 == n2
        assert torch.allclose(a1, a2)


def test_traversal_utils():

    target_tensor = torch.randn(2, 2)
    outputs = (
        (1, 3),
        ["abs", "sd", 2],
        [[target_tensor, torch.randint(10, size=(3, 3))], (2, 3)],
        (6, 5, 4, "123", nn.Linear(10, 20), torch.randn(3, 3)),
    )

    tensor = _recursively_find_first_tensor(outputs)

    assert torch.equal(tensor, target_tensor)

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
    setup_logger("info")
    logger.info("Testing simple PyTorch model...")
    test_simple_model()
    logger.info("Test successful!")

    logger.info("Testing traversal/parse utils...")
    test_traversal_utils()
    logger.info("Test successful!")


if __name__ == "__main__":
    main()
