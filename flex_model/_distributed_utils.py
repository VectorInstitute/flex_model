from functools import partial, reduce
import logging
from typing import List, Tuple, Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor


_GLOBAL_ACTIVATION_GROUP = None

logger = logging.getLogger(__name__)


def _set_activation_group(ranks: List[int]) -> None:
    """Given a list of ranks, initialize a new `ProcessGroup`"""
    global _GLOBAL_ACTIVATION_GROUP

    assert torch.distributed.get_world_size() >= len(ranks)
    assert _GLOBAL_ACTIVATION_GROUP is None

    act_proc_group = dist.new_group(
        ranks=ranks,
        backend="nccl",
    )

    _GLOBAL_ACTIVATION_GROUP = act_proc_group


def _get_activation_parallel_group() -> dist.ProcessGroup:
    """Return global activation processes group handle."""
    global _GLOBAL_ACTIVATION_GROUP
    assert _GLOBAL_ACTIVATION_GROUP is not None
    return _GLOBAL_ACTIVATION_GROUP


def _get_activation_parallel_world_size() -> int:
    """Return global activation processes world size."""
    if not dist.is_initialized():
        return 1

    return dist.get_world_size(
        group=_get_activation_parallel_group(),
    )


def print_rank0(msg: str) -> None:
    """Print to rank 0 worker."""
    if dist.is_initialized() and dist.get_rank() == 0:
        print(msg, flush=True)


def _get_different_dim(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> int:
    """Find non-matching dims."""
    assert len(shape1) == len(shape2), "Shapes have different ndims"

    diff = [i for i in range(len(shape1)) if shape1[i] != shape2[i]]
    assert len(diff) == 1, f"Multiple sharded axes found: {shape1}, {shape2}"
    sharded_dim = diff[0]

    return sharded_dim


def _autofill_expected_shape(
    tensor: Tensor,
    expected_shape: Tuple[Optional[int], ...]
) -> Tuple[int, ...]:
    """
    Fill in unspecified dimensions in the expected shape of the full-size
    activation.
    """
    tensor_shape = tensor.shape
    assert len(tensor_shape) == len(expected_shape), (
        f"Shape have different ndims: {len(tensor_shape)}, {len(expected_shape)}")

    # Expected shape contains None dimensions because it's annoying for the
    # user to know them exactly (ie. seq len after tokenization). Hence we will
    # infer them from the activation tensor.
    filled_shape = tuple(
        d1
        if d2 is None
        else d2
        for d1, d2 in zip(tensor_shape, expected_shape)
    )
    print_rank0(f"Inferring shape: User-{expected_shape}, Inferred-"
                f"{filled_shape}")
    return filled_shape


def _parse_collect_and_distribute_from_tensor(tensor: Tensor, expected_shape: Tuple[int, ...]) -> Tuple[Callable, Callable]:
    """Parse the activation tensor vs expected shape for distributed strategy."""
    world_size = _get_activation_parallel_world_size()

    # Handle unspecified dimensions
    expected_shape = _autofill_expected_shape(tensor, expected_shape)

    # Single gpu fallback
    if world_size == 1:
        return _unity, _unity

    # Make sure tensor ndims are same and sharding is valid
    tensor_numel = tensor.numel()
    expected_numel = reduce(lambda x, y: x * y, expected_shape)
    assert tensor_numel in [expected_numel // world_size, expected_numel], (
        f"tensor: {tensor_numel}, expected: {expected_numel} or "
        f"{expected_numel // world_size}")
    assert len(tensor.shape) == len(expected_shape)

    is_sharded = tensor_numel != expected_numel

    # Need to clone since all-reduce acts in-place
    cmp_tensor = tensor.clone()
    cmp_tensor = _all_reduce_sync(cmp_tensor)
    same_data = torch.allclose(tensor * world_size, cmp_tensor)

    # All ranks have replicated activations
    if not is_sharded and same_data:
        collect = _unity
        disperse = _broadcast_rank0_sync

    # Ranks have different activations
    elif not is_sharded and not same_data:
        # Case for partial sums resulting from RowParallel matmul
        # TODO: Need to somehow undo the reduce
        raise NotImplementedError

    # Tensors are sharded along one dim
    else:
        sharded_axis = _get_different_dim(tensor.shape, expected_shape)

        collect = partial(_gather_rank0_sync, axis=sharded_axis)
        disperse = partial(_scatter_rank0_sync, axis=sharded_axis)

    return collect, disperse


def _parse_edit_from_function(edit_function):
    """Parse the user-provided editing function."""
    if edit_function is None:
        parsed_edit_function = _unity
    else:
        parsed_edit_function = edit_function
    return parsed_edit_function


def _parse_dump_from_function(dump_function):
    """Parse the provided dump function."""
    return dump_function


def _unity(tensor: Tensor):
    print_rank0(f"Unity | In:   {tensor.shape}")
    return tensor


def _broadcast_rank0_sync(
    tensor: Tensor,
) -> Tensor:
    """Synchronous broadcast from rank0."""
    dist.broadcast(
        tensor=tensor,
        src=0,
        group=_get_activation_parallel_group(),
        async_op=False,
    )

    print_rank0(f"Broadcast | In:   {tensor.shape}")

    return tensor


def _gather_rank0_sync(
    tensor: Tensor,
    axis: int = 0,
) -> Tensor:
    """Syncronous gather onto rank0."""
    world_size = _get_activation_parallel_world_size()

    if torch.distributed.get_rank() == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None

    dist.gather(
        tensor=tensor,
        gather_list=gather_list,
        dst=0,
        group=_get_activation_parallel_group(),
        async_op=False,
    )

    # Non-rank0 workers can return and wait until hook exit
    if torch.distributed.get_rank() != 0:
        return tensor

    output_tensor = torch.cat(gather_list, dim=axis)
    print_rank0(
        f"Gather | In:  {tensor.shape} -> {output_tensor.shape} Dim: {axis} "
        f"Collecting: {len(gather_list)} chunks."
    )

    return output_tensor


def _all_reduce_sync(
    tensor: Tensor,
) -> Tensor:
    """Synchronous allreduce."""
    inplace_tensor = tensor.clone()
    dist.all_reduce(
        tensor=inplace_tensor,
        op=dist.ReduceOp.SUM,
        group=_get_activation_parallel_group(),
        async_op=False,
    )
    return inplace_tensor


def _reduce_rank0_sync(
    tensor: Tensor,
) -> Tensor:
    """Synchronous reduce onto rank0.

    Reduce is done inplace, but we don't want to destroy the passed
    activation tensor since the model may need it downstream.
    """
    inplace_tensor = tensor.clone()
    dist.reduce(
        tensor=inplace_tensor,
        dst=0,
        op=dist.ReduceOp.SUM,
        group=_get_activation_parallel_group(),
        async_op=False,
    )
    # Shed non-rank0 workers
    if torch.distributed.get_rank() != 0:
        return tensor

    return inplace_tensor


def _scatter_rank0_sync(
    tensor: Tensor,
    axis: int = 0,
) -> Tensor:
    """Synchronous scatter from rank0 to all."""
    # Rank0 has full-rank act tensor, need to shard output tensor
    if dist.get_rank() == 0:
        shape = list(tensor.shape)
        world_size = _get_activation_parallel_world_size()
        assert shape[axis] % world_size == 0
        shape[axis] //= world_size

    # Non-rank0 workers already have stale sharded tensors
    else:
        shape = tensor.shape

    output_tensor = torch.empty(
        shape,
        dtype=tensor.dtype,
        layout=tensor.layout,
        device=tensor.device,
    )

    if dist.get_rank() == 0:
        scatter_list = list(torch.chunk(
            tensor,
            chunks=world_size,
            dim=axis,
        ))

        print_rank0(
            f"Scatter | In:     {tensor.shape} -> {output_tensor.shape} Dim: {axis} "
            f"Sending: {len(scatter_list)} chunks."
        )

    else:
        scatter_list = None

    dist.scatter(
        tensor=output_tensor,
        scatter_list=scatter_list,
        src=0,
        group=_get_activation_parallel_group(),
        async_op=False,
    )

    return output_tensor
