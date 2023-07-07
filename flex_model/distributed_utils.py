from functools import partial
import logging
from typing import List, Tuple, Callable

import torch
import torch.distributed as dist
from torch import Tensor


_GLOBAL_ACTIVATION_GROUP = None

logger = logging.getLogger(__name__)


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


def _handle_activation_sharded(tensor: Tensor, shape: Tuple[int]) -> Tuple[Callable, Callable]:
    """Get comm. collectives for sharded activation tensors."""
    assert len(tensor.shape) == len(shape), f"Unequal shapes! {tensor.shape}, {shape}"

    # Find the sharded shape axis
    diff = [i for i in range(len(tensor.shape)) if tensor.shape[i] != shape[i]]
    assert len(diff) == 1, f"Multiple sharded axes found: {tensor.shape}, {shape}"
    sharded_dim = diff[0]

    # Case1: Even sharding (chunked) - Assumed TP col parallel or DP
    if shape[sharded_dim] % tensor.shape[sharded_dim] == 0:
        collect_fn = partial(_gather_rank0_sync, axis=sharded_dim)
        distribute_fn = partial(_scatter_rank0_sync, axis=sharded_dim)
    
    # Case2: Uneven sharding, need to keep track of where the split is - 
    # Assumed uneven DP
    else:
        logger.info(f"Uneven sharding detected: {tensor.shape}, {shape}")
        raise NotImplementedError

    return collect_fn, distribute_fn


def _handle_activation_full(tensor: Tensor, shape: Tuple[int]) -> Tuple[Callable, Callable]:
    """Get comm. collectives for full-rank activation tensors."""
    # Named function for ease of logging
    def _unity(x): return x

    # Check if all ranks have the same tensor
    world_size = _get_activation_parallel_world_size()
    comparison_tensor = _all_reduce_sync(tensor)
    same_tensor_data = torch.allclose(tensor * world_size, comparison_tensor)

    # Same data, just use rank0 tensor - Assumed TP replicated
    if same_tensor_data:
        collect_fn = _unity

    # Different data, need to reduce - Assumed TP row parallel partial prods
    else:
        collect_fn = _reduce_rank0_sync

    distribute_fn = _broadcast_rank0_sync

    return collect_fn, distribute_fn


def _infer_collective(tensor: Tensor, shape: Tuple[int]) -> Tuple[Callable, Callable]:
    """
    Given an actiation tensor and expected shape, infer the proper
    comm. collective to materialize the true activation tensor.
    """

    # Activation tensor sharded along one dimension
    if tensor.shape != shape:
        collect_fn, distribute_fn = _handle_activation_sharded(tensor, shape)
        
    # Activation tensor is full-rank
    else:
        collect_fn, distribute_fn = _handle_activation_full(tensor, shape)
        
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

    return torch.cat(gather_list, dim=axis)


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
