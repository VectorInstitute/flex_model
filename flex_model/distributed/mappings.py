import logging
from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor

from flex_model.distributed.initialize import (
    get_activation_parallel_group,
    get_world_size,
    get_rank,
)


logger = logging.getLogger(__name__)


def unity(tensor: Tensor) -> Tensor:
    logger.debug(f"Unity | In:   {tensor.shape}")
    return tensor


def broadcast_rank0_sync(
    tensor: Tensor,
) -> Tensor:
    """Synchronous broadcast from rank0."""
    if get_world_size() == 0:
        return tensor

    torch.distributed.broadcast(
        tensor=tensor,
        src=0,
        group=get_activation_parallel_group(),
        async_op=False,
    )

    logger.debug(f"Broadcast | In:   {tensor.shape}")

    return tensor


def all_gather_sync(
    tensor: Tensor,
    axis: int = 0,
) -> Tensor:
    """Syncronous gather onto rank0."""
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    tensor_list[get_rank()] = tensor

    torch.distributed.all_gather(
        tensor_list,
        tensor,
        group=get_activation_parallel_group(),
        async_op=False,
    )

    output_tensor = torch.cat(tensor_list, dim=axis)
    logger.debug(
        f"Allgather | In:  {tensor.shape} -> {output_tensor.shape} Dim: {axis} "
        f"Collecting: {world_size} chunks."
    )

    return output_tensor


def all_reduce_sync(
    tensor: Tensor,
) -> Tensor:
    """Synchronous allreduce."""
    if get_world_size == 0:
        return tensor

    inplace_tensor = tensor.clone()
    torch.distributed.all_reduce(
        tensor=inplace_tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=get_activation_parallel_group(),
        async_op=False,
    )
    return inplace_tensor


def _reduce_rank0_sync(
    tensor: Tensor,
) -> Tensor:
    """Synchronous reduce onto rank0.

    Reduce is done inplace, but we don't want to destroy the passed
    activation tensor since the model may need it downstream.

    NOTE: Not currently supported officially
    """
    if get_world_size == 0:
        return tensor

    inplace_tensor = tensor.clone()
    torch.distributed.reduce(
        tensor=inplace_tensor,
        dst=0,
        op=torch.distributed.ReduceOp.SUM,
        group=get_activation_parallel_group(),
        async_op=False,
    )
    # Shed non-rank0 workers
    if get_rank() != 0:
        return tensor

    return inplace_tensor


def scatter_rank0_sync(
    tensor: Tensor,
    axis: int = 0,
) -> Tensor:
    """Synchronous scatter from rank0 to all."""
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    input_list = torch.chunk(tensor, world_size, dim=axis)

    rank = get_rank()
    output_tensor = input_list[rank].contiguous()

    logger.debug(
        f"Scatter | In:     {tensor.shape} -> {output_tensor.shape} Dim: "
        f"{axis} Sending: {world_size} chunks."
    )

    return output_tensor
