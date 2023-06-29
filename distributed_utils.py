from functools import partial
from typing import List, Tuple, Callable

import torch
import torch.distributed as dist
from torch import Tensor


_GLOBAL_ACTIVATION_GROUP = None


def _set_activation_group(ranks: List[int]) -> None:
    """Given a list of ranks, initialize a new `ProcessGroup`"""
    assert torch.distributed.get_world_size() >= len(ranks)

    act_proc_group = dist.new_group(
        ranks=ranks,
        backend="nccl",
    )

    global _GLOBAL_ACTIVATION_GROUP
    _GLOBAL_ACTIVATION_GROUP = act_proc_group


def _get_activation_parallel_group() -> dist.ProcessGroup:
    """Return global activation processes group handle."""
    global _GLOBAL_ACTIVATION_GROUP
    assert _GLOBAL_ACTIVATION_GROUP is not None
    return _GLOBAL_ACTIVATION_GROUP


def _get_activation_parallel_world_size() -> int:
    """Return global activation processes world size."""
    return dist.get_world_size(
        group=_get_activation_parallel_group(),
    )


def _infer_collective(tensor: Tensor, shape: Tuple[int]) -> Tuple[Callable, Callable]:
    """
    Given an actiation tensor and expected shape, infer the proper
    comm. collective to materialize the true activation tensor.
    """
    world_size = _get_activation_parallel_world_size()

    # Activation tensor sharded along one dimension
    if tensor.shape != shape:
        # NOTE: We don't care if ranks have same data or not
        assert len(tensor.shape) == len(shape)

        # Find the sharded axis
        diff = []
        for i in range(len(tensor.shape)):
            if tensor.shape[i] != shape[i]:
                diff.append(i)
        assert len(diff) == 1
        sharded_dim = diff[0]
        assert shape[sharded_dim] % tensor.shape[sharded_dim] == 0

        collect_fn = partial(_gather_rank0_sync, axis=sharded_dim)
        distribute_fn = partial(_scatter_rank0_sync, axis=sharded_dim)

    # Activation tensor is full-rank
    else:
        comparison_tensor = _all_reduce_sync(tensor)

        # All ranks have the same data
        if torch.allclose(tensor * world_size, comparison_tensor):
            collect_fn = lambda x: x
        # All ranks have different data
        else:
            collect_fn = _reduce_rank0_sync

        distribute_fn = _broadcast_rank0_sync

    return collect_fn, distribute_fn


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
    return tensor


def _gather_rank0_sync(
    tensor: Tensor,
    axis: int = 0,
) -> Tensor:
    """Syncronous gather onto rank0."""
    world_size = _get_activation_parallel_world_size()

    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    dist.gather(
        tensor=tensor,
        gather_list=gather_list,
        dst=0,
        group=_get_activation_parallel_group(),
        async_op=False,
    )

    return torch.cat(gather_list, dim=axis)


def _all_reduce_sync(
    tensor: Tensor,
) -> Tensor:
    """Synchronous allreduce."""
    inplace_tensor = tensor.clone()
    dist.all_reduce(
        tensor=inplace_tensor,
        op=dist.ReduceOp.SUM,
        grop=_get_activation_parallel_group(),
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
    return inplace_tensor


def _scatter_rank0_sync(
    tensor: Tensor,
    axis: int = 0,
) -> Tensor:
    """Synchronous scatter from rank0 to all."""
    shape = list(tensor.shape)
    world_size = _get_activation_parallel_world_size()
    assert shape[axis] % world_size == 0
    shape[axis] //= world_size

    output_tensor = torch.empty(
        shape,
        dtype=tensor.dtype,
        layout=tensor.layout,
        device=tensor.device,
    )
    dist.scatter(
        tensor=output_tensor,
        scatter_list=torch.chunk(
            tensor,
            chunks=_get_activation_parallel_world_size(),
            dim=0,
        ),
        src=0,
        group=_get_activation_parallel_group(),
        async_op=False,
    )
    return output_tensor
