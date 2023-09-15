from functools import partial, reduce
import logging
from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor

import flex_model.distributed as dist


logger = logging.getLogger(__name__)


def _get_different_dim(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> int:
    """Find the indices of elements which differ between two tuples.

    Get all indices where the first tuple does not match the second tuple. In
    the context of tensor shapes, the non-matching indices correspond to
    dimensions which are sharded.

    :param shape1: The first tensor shape.
    :type shape1: Tuple[int, ...]
    :param shape2: The second tensor shape.
    :type shape2: Tuple[int, ...]

    :returns: A single index of the matching dimension. If there all dimensions match
        then returns -1.
    :rtype: int

    :raises AssertionError: Input shapes have different number of dimensions.
    :raises AssertionError: Two or more dimensions do not match.
    """
    assert len(shape1) == len(shape2), "Shapes have different ndims"
    different_dims: List[int] = []
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            different_dims.append(i)

    assert len(different_dims) == 1 or len(different_dims) == 0, (
        f"Multiple sharded axes found: {shape1}," f" {shape2}"
    )

    return different_dims[0] if len(different_dims) == 1 else -1


def _autofill_expected_shape(
    tensor: Tensor, expected_shape: Tuple[Optional[int], ...]
) -> Tuple[int, ...]:
    """Complete the `None`-annotated tensor shape dimensions.

    Compare the tensor shape and the expected shape. For dimensions in the
    expected shape that are annotated as `None`, fill them in with the
    corresponding dimension in the tensor shape. Note that if the user does
    not annotate the correct dimension, then the collection of activations will
    fail.

    :param Tensor tensor: Local device tensor.
    :param expected_shape: Shape of the non-sharded tensor.
    :type expected_shape: Tuple[Optional[int], ...]

    :returns: A tuple representing the filled-in expected shape with no `None`
        annotations.
    :rtype: Tuple[int, ...]

    :raises AssertionError: There are a different number of dimensions in the
        local device tensor compared to the expected shape.
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


def parse_collect_from_parameter_tensor(
    tensor: Tensor,
    expected_shape: Tuple[Optional[int], ...],
) -> Callable:
    """Find the communication function which gathers the full parameter tensor.

    Similar to gathering activations, parameter tensors can be gathered too as
    a convenience. Given the local device parameter tensor, compare it to the
    expected shape provided and infer the necessary collective communication
    function required to assemble the unsharded parameter tensor.

    :param Tensor tensor: Local device parameter tensor.
    :param expected_shape: Shape of the non-sharded parameter tensor.
    :type expected_shape: Tuple[Optional[int], ...]

    :returns: Collective communication function which assembles the full parameter
        tensor once called.
    :rtype: Callable

    :raises AssertionError: Occurs if the sharding is not evenly distributed across
        devices.
    """
    if not torch.distributed.is_initialized():
        return dist.unity

    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    dp_world_size = dist.get_activation_data_parallel_world_size()

    # Handle unspecifed dimensions
    expected_shape = _autofill_expected_shape(tensor, expected_shape)

    # Validate tensor sharding scheme
    tensor_numel = tensor.numel()
    expected_numel = reduce(lambda x, y: x * y, expected_shape)

    # NOTE: DP sharding support is supposed to be for FSDP, but unused right
    #       now.
    # Both parameter and grad tensors can be sharded arbitraryily in the TP
    # and DP dimensions. The valid cases are:
    # 1. Unsharded
    # 2. Sharded along TP axis.
    # 3. Sharded along DP axis.
    # 4. Sharded along TP and DP axis.
    valid_sharding_schemes = [
        expected_numel,
        expected_numel // tp_world_size,
        expected_numel // dp_world_size,
        expected_numel // (tp_world_size * dp_world_size),
    ]

    assert tensor_numel in valid_sharding_schemes, (
        f"Imperfect activation sharding: Given {tensor_numel} expected "
        f"{expected_numel}"
    )

    sharded_dim = _get_different_dim(tensor.shape, expected_shape)

    if tp_world_size == 1 or sharded_dim == -1:
        return dist.unity

    return lambda t: dist.all_gather_tensor_parallel(t, dim=sharded_dim)


def parse_collect_and_distribute_from_tensor(
    tensor: Tensor,
    expected_shape: Tuple[Optional[int], ...],
) -> Tuple[Callable, Callable]:
    """Find the appropriate collect/disperse communication function.

    Infers the correct collection and dispersion functions required to assemble
    a full activation from local device shards and to disassemble a full
    activation into local device shards respectively.

    :param Tensor tensor: Local activation tensor
    :param expected_shape: Shape of the non-sharded activation tensor.
    :type expected_shape: Tuple[Optional[int], ...]

    :returns: Collection and dispersion collective communication functions.
    :rtype: Tuple[Callable, Callable]

    :raises AssertionError: Occurs if the sharding is not evenly distributed across
        devices.
    :raises Exception: Occurs if the tensor parallel and data parallel world sizes
        return invalid values.
    """
    if not torch.distributed.is_initialized():
        return dist.unity, dist.unity

    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    dp_world_size = dist.get_activation_data_parallel_world_size()

    # Handle unspecified dimensions
    expected_shape = _autofill_expected_shape(tensor, expected_shape)

    # Validate tensor sharding scheme
    tensor_numel = tensor.numel()
    expected_numel = reduce(lambda x, y: x * y, expected_shape)
    assert tensor_numel in [
        expected_numel // tp_world_size,  # Evenly sharded across TP
        expected_numel,  # Full-rank
    ], (
        f"Imperfect activation sharding: Given {tensor_numel} expected "
        f"{expected_numel}"
    )

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
