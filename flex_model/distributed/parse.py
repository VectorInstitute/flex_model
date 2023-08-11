from functools import partial, reduce
import logging
from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor

from flex_model.distributed.initialize import (
    get_world_size,
)
from flex_model.distributed.mappings import (
    unity,
    broadcast_rank0_sync,
    all_gather_sync,
    all_reduce_sync,
    scatter_rank0_sync,
)


logger = logging.getLogger(__name__)


def _get_different_dim(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> int:
    """Find non-matching dims."""
    assert len(shape1) == len(shape2), "Shapes have different ndims"

    diff = [i for i in range(len(shape1)) if shape1[i] != shape2[i]]
    assert len(diff) == 1, f"Multiple sharded axes found: {shape1}, {shape2}"
    sharded_dim = diff[0]

    return sharded_dim


def _autofill_expected_shape(
    tensor: Tensor, expected_shape: Tuple[Optional[int], ...]
) -> Tuple[int, ...]:
    """
    Fill in unspecified dimensions in the expected shape of the full-size
    activation.
    """
    tensor_shape = tensor.shape
    assert len(tensor_shape) == len(
        expected_shape
    ), f"Shape have different ndims: {len(tensor_shape)}, {len(expected_shape)}"

    # Expected shape contains None dimensions because it's annoying for the
    # user to know them exactly (ie. seq len after tokenization). Hence we will
    # infer them from the activation tensor.
    filled_shape = tuple(
        d1 if d2 is None else d2 for d1, d2 in zip(tensor_shape, expected_shape)
    )
    logger.debug(f"Inferring shape: User-{expected_shape}, Inferred-" f"{filled_shape}")
    return filled_shape


def parse_collect_and_distribute_from_tensor(
    tensor: Tensor, expected_shape: Tuple[Optional[int], ...]
) -> Tuple[Callable, Callable]:
    """Parse the activation tensor vs expected shape for distributed strategy."""
    world_size = get_world_size()

    # Handle unspecified dimensions
    expected_shape = _autofill_expected_shape(tensor, expected_shape)

    # Single gpu fallback
    if world_size == 1:
        return unity, unity

    # Make sure tensor ndims are same and sharding is valid
    tensor_numel = tensor.numel()
    expected_numel = reduce(lambda x, y: x * y, expected_shape)
    assert tensor_numel in [
        expected_numel // world_size,
        expected_numel,
    ], f"tensor: {tensor.shape}, expected: {expected_shape} or sharded version"
    assert len(tensor.shape) == len(expected_shape)

    is_sharded = tensor_numel != expected_numel

    # Need to clone since all-reduce acts in-place
    cmp_tensor = tensor.clone()
    cmp_tensor = all_reduce_sync(cmp_tensor)
    same_data = torch.allclose(tensor * world_size, cmp_tensor)

    # All ranks have replicated activations
    if not is_sharded and same_data:
        collect = unity
        disperse = broadcast_rank0_sync

    # Ranks have different activations
    elif not is_sharded and not same_data:
        # Case for partial sums resulting from RowParallel matmul
        # TODO: Need to somehow undo the reduce
        raise NotImplementedError

    # Tensors are sharded along one dim
    else:
        sharded_axis = _get_different_dim(tensor.shape, expected_shape)

        collect = partial(all_gather_sync, axis=sharded_axis)
        disperse = partial(scatter_rank0_sync, axis=sharded_axis)

    return collect, disperse
