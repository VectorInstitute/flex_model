import logging
from typing import List, Tuple, Callable, Optional, Any

import torch
from torch import Tensor

import flex_model.distributed as dist


logger = logging.getLogger(__name__)


def unity(tensor: Tensor) -> Tensor:
    """No-op function.

    Args:
        tensor: Activation tensor.
    """
    logger.debug(f"Unity | IN:   {tensor.shape}")
    return tensor


def broadcast_tensor_parallel(tensor: Tensor) -> Tensor:
    """Send a copy of the tensor to all members of the tensor parallel group.

    Args:
        tensor: Tensor to broadcast.

    Returns:
        All tensor parallel ranks will get the same tensor.
    """
    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    tp_group = dist.get_activation_tensor_parallel_group()

    if tp_world_size == 1:
        return tensor

    # We only interact among tensor parallel group to bcast
    torch.distributed.broadcast(
        tensor=tensor,
        src=0,
        group=tp_group,
        async_op=False,
    )

    logger.debug(f"Broadcast | IN: {tensor.shape}")
    return tensor


def broadcast_data_parallel(tensor: Tensor) -> Tensor:
    """Send a copy of the data to all members of the data parallel group.

    Args:
        tensor: Tensor to broadcast.

    Returns:
        All data parallel ranks will get the same tensor.
    """
    if not dist.in_data_parallel_group():
        return tensor

    dp_world_size = dist.get_activation_data_parallel_world_size()
    dp_group = dist.get_activation_data_parallel_group()

    if dp_world_size == 1:
        return tensor

    # We only interact among tensor parallel group to bcast
    torch.distributed.broadcast(
        tensor=tensor,
        src=0,
        group=dp_group,
        async_op=False,
    )

    logger.debug(f"Broadcast | IN: {tensor.shape}")
    return tensor


def all_gather_tensor_parallel(tensor: Tensor, dim: int = -1) -> Tensor:
    """Gather all tensors from tensor parallel group to each device in group.

    Each device in the tensor parallel group has a tensor, and every tensor
    will be sent to all other members of the tensor parallel group. For
    example, if gpu0 has T0 and gpu1 has T1, then after this operation gpu0
    will have [T0, T1] and gpu1 will also have [T0, T1]. A concatenation is
    done after the communication to generate a single tensor.

    Args:
        tensor: Tensor to all-gather.
        dim: Dimension to concatenate the gathered tensors along.

    Returns:
        The gathered and concatenated tensor.
    """
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
    """Gather all tensors from data parallel group to each device in group.

    Each device in the data parallel group has a tensor, and every tensor
    will be sent to all other members of the data parallel group. For
    example, if gpu0 has T0 and gpu1 has T1, then after this operation gpu0
    will have [T0, T1] and gpu1 will also have [T0, T1]. A concatenation is
    done after the communication to generate a single tensor.

    Args:
        tensor: Tensor to all-gather.
        dim: Dimension to concatenate the gathered tensors along.

    Returns:
        The gathered and concatenated tensor.
    """
    if not dist.in_data_parallel_group():
        return tensor

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
    """Unused."""
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
    """Chunk a tensor and send chunks to corresponding tensor parallel ranks.

    Given a tensor, chunk it along a specific dimension. Each device rank will
    get the corresponding tensor chunk. For example, T = [T0, T1], where gpu0
    will get T0 and gpu1 will get T1. Notably, the scatter functions are always
    used after all-gather functions. Hence each rank has a full copy of the
    tensor T, so each rank instead discards all chunks besides their own
    corresponding chunk.

    Args:
        tensor: Tensor to scatter.
        dim: Dimension to chunk the tensor along.

    Returns:
        The corresponding chunk of the full tensor.
    """
    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    tp_rank = dist.get_activation_tensor_parallel_rank()

    if tp_world_size == 1:
        return tensor

    input_list = torch.chunk(tensor, tp_world_size, dim=dim)
    output_tensor = input_list[tp_rank].contiguous()

    logger.debug(f"Scatter | IN: {tensor.shape} -> {output_tensor.shape}")
    return output_tensor


def scatter_data_parallel(tensor: Tensor, dim: int = 0):
    """Chunk a tensor and send chunks to corresponding data parallel ranks.

    Given a tensor, chunk it along a specific dimension. Each device rank will
    get the corresponding tensor chunk. For example, T = [T0, T1], where gpu0
    will get T0 and gpu1 will get T1. Notably, the scatter functions are always
    used after all-gather functions. Hence each rank has a full copy of the
    tensor T, so each rank instead discards all chunks besides their own
    corresponding chunk.

    Args:
        tensor: Tensor to scatter.
        dim: Dimension to chunk the tensor along.

    Returns:
        The corresponding chunk of the full tensor.
    """
    if not dist.in_data_parallel_group():
        return tensor

    dp_world_size = dist.get_activation_data_parallel_world_size()
    dp_rank = dist.get_activation_data_parallel_rank()

    if dp_world_size == 1:
        return tensor

    input_list = torch.chunk(tensor, dp_world_size, dim=dim)
    output_tensor = input_list[dp_rank].contiguous()

    logger.debug(f"Scatter | IN: {tensor.shape} -> {output_tensor.shape}")
    return output_tensor


def gather_pipeline_parallel(objects: Any) -> Any:
    """Gather an object from non-zero ranks to rank0 in the pipeline group.

    Takes a pickle-able collection of python objects and tensors and sends
    them to rank0 in the pipeline parallel group.

    Args:
        objects: Some python object that can be pickled. May contain tensors.

    Returns:
        A collection of the objects sent from all pipeline paralel group ranks.
    """
    if not dist.in_pipeline_parallel_group():
        return objects

    pp_world_size = dist.get_activation_pipeline_parallel_world_size()
    pp_rank = dist.get_activation_pipeline_parallel_rank()

    output = [None for _ in range(pp_world_size)]
    torch.distributed.gather_object(
        objects,
        output if pp_rank == 0 else None,
        dst=0,
    )

    if pp_rank == 0:
        logger.debug(f"Gather | IN: {len(objects)} -> {len(output)}")
    return output
