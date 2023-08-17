import logging
from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor

import flex_model.distributed as dist


logger = logging.getLogger(__name__)


def unity(tensor: Tensor) -> Tensor:
    logger.debug(f"Unity | IN:   {tensor.shape}")
    return tensor


def broadcast_tensor_parallel(tensor: Tensor) -> Tensor:
    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    tp_rank = dist.get_activation_tensor_parallel_rank()
    tp_group = dist.get_activation_tensor_parallel_group()

    if tp_world_size == 1:
        return tensor

    # We only interact among tensor parallel group to bcast
    torch.ditributed.broadcast(
        tensor=tensor,
        src=tp_rank,
        group=tp_group,
        async_op=False,
    )

    logger.debug(f"Broadcast | IN: {tensor.shape}")
    return tensor


def broadcast_data_parallel(tensor: Tensor) -> Tensor:
    dp_world_size = dist.get_activation_data_parallel_world_size()
    dp_rank = dist.get_activation_data_parallel_rank()
    dp_group = dist.get_activation_data_parallel_group()

    if dp_world_size == 1:
        return tensor

    # We only interact among tensor parallel group to bcast
    torch.ditributed.broadcast(
        tensor=tensor,
        src=dp_rank,
        group=dp_group,
        async_op=False,
    )

    logger.debug(f"Broadcast | IN: {tensor.shape}")
    return tensor


def all_gather_tensor_parallel(tensor: Tensor, dim: int = -1) -> Tensor:
    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    tp_rank = dist.get_activation_tensor_parallel_rank()
    tp_group = dist.get_activation_tensor_parallel_group()

    if tp_world_size == 1:
        return tensor

    tensor_list = [torch.empty_like(tensor) for _ in range(tp_world_size)]
    tensor_list[tp_rank] = tensor

    torch.distributed.all_gather(
        tensor_list,
        tensor,
        group=tp_group,
        async_op=False,
    )

    output_tensor = torch.cat(tensor_list, dim=dim)

    logger.debug(f"Allgather | IN: {tensor.shape} -> {output_tensor.shape}")
    return output_tensor


def all_gather_data_parallel(tensor: Tensor, dim: int = 0) -> Tensor:
    dp_world_size = dist.get_activation_data_parallel_world_size()
    dp_rank = dist.get_activation_data_parallel_rank()
    dp_group = dist.get_activation_data_parallel_group()

    if dp_world_size == 1:
        return tensor

    tensor_list = [torch.empty_like(tensor) for _ in range(dp_world_size)]
    tensor_list[dp_rank] = tensor

    torch.distributed.all_gather(
        tensor_list,
        tensor,
        group=dp_group,
        async_op=False,
    )

    output_tensor = torch.cat(tensor_list, dim=dim)

    logger.debug(f"Allgather | IN: {tensor.shape} -> {output_tensor.shape}")
    return output_tensor


def all_reduce_tensor_parallel(tensor: Tensor) -> Tensor:
    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    tp_rank = dist.get_activation_tensor_parallel_rank()
    tp_group = dist.get_activation_tensor_parallel_group()

    if tp_world_size == 1:
        return tensor

    tensor = tensor.clone()
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=tp_group,
        async_op=False,
    )
    return tensor

def scatter_tensor_parallel(tensor: Tensor, dim: int = -1):
    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    tp_rank = dist.get_activation_tensor_parallel_rank()

    if tp_world_size == 1:
        return tensor

    input_list = torch.chunk(tensor, tp_world_size, dim=dim)
    output_tensor = input_list[tp_rank].contiguous()

    logger.debug(f"Scatter | IN: {tensor.shape} -> {output_tensor.shape}")
    return output_tensor


def scatter_data_parallel(tensor: Tensor, dim: int = 0):
    dp_world_size = dist.get_activation_data_parallel_world_size()
    dp_rank = dist.get_activation_data_parallel_rank()

    if dp_world_size == 1:
        return tensor

    input_list = torch.chunk(tensor, dp_world_size, dim=dim)
    output_tensor = input_list[dp_rank].contiguous()

    return output_tensor
