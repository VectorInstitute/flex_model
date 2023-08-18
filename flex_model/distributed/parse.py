from functools import partial, reduce
import logging
from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor

import flex_model.distributed as dist


logger = logging.getLogger(__name__)


def _get_different_dim(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> int:
    """Find non-matching dims."""
    assert len(shape1) == len(shape2), "Shapes have different ndims"
    different_dims: List[int] = []
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            different_dims.append(i)

    assert (
        len(different_dims) == 1 or
        len(different_dims) == 0
    ), (f"Multiple sharded axes found: {shape1},"
        f" {shape2}")

    return different_dims[0] if len(different_dims) == 1 else -1


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
    tensor: Tensor,
    expected_shape: Tuple[Optional[int], ...],
    tp_world_size: int,
    dp_world_size: int,
) -> Tuple[Callable, Callable]:
    """Parse the activation tensor vs expected shape for distributed strategy."""

    # Handle unspecified dimensions
    expected_shape = _autofill_expected_shape(tensor, expected_shape)

    # Validate tensor sharding scheme
    tensor_numel = tensor.numel()
    expected_numel = reduce(lambda x, y: x * y, expected_shape)
    assert tensor_numel in [
        expected_numel // tp_world_size,    # Evenly sharded across TP
        expected_numel,                     # Full-rank
    ], (f"Imperfect activation sharding: Given {tensor_numel} expected "
        f"{expected_numel}")

    # Single gpu fallback
    if tp_world_size == dp_world_size == 1:
        return dist.unity, dist.unity

    # Activation possibly sharded over tp, no need to gather over dp
    if tp_world_size > 1 and dp_world_size == 1:
        sharded_dim = _get_different_dim(tensor.shape, expected_shape)

        # Not sharded over tp
        if sharded_dim == -1:
            collect_fn = dist.unity
            disperse_fn = dist.unity

        # Sharded over tp
        else:
            collect_fn = lambda t: dist.all_gather_tensor_parallel(t, dim=sharded_dim)
            disperse_fn = lambda t: dist.scatter_tensor_parallel(t, dim=sharded_dim)

    # Only data parallelism, always gather
    elif tp_world_size == 1 and dp_world_size > 1:
        collect_fn = lambda t: dist.all_gather_data_parallel(t, dim=0)
        disperse_fn = lambda t: dist.scatter_data_parallel(t, dim=0)

    # Activation possibly sharded over tp, always gather over dp
    elif tp_world_size > 1 and dp_world_size > 1:
        sharded_dim = _get_different_dim(tensor.shape, expected_shape)

        # Not sharded over tp
        if sharded_dim == -1:
            collect_fn = lambda t: dist.all_gather_data_parallel(t, dim=0)
            disperse_fn = lambda t: dist.scatter_data_parallel(t, dim=0)

        # Sharded over tp
        else:
            collect_fn = lambda t: dist.all_gather_data_parallel(
                dist.all_gather_tensor_parallel(t, dim=sharded_dim),
                dim=0,
            )
            disperse_fn = lambda t: dist.scatter_tensor_parallel(
                dist.scatter_data_parallel(t, dim=0),
                dim=sharded_dim,
            )

    else:
        raise Exception("Invalid world sizes: tp{tp_world_size}, dp{dp_world_size}")

    return collect_fn, disperse_fn
