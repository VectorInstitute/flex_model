from collections import defaultdict
import logging
from typing import List, Tuple, Callable, Optional, Any, Dict, Union

import torch
from torch import Tensor

import flex_model.distributed as dist


logger = logging.getLogger(__name__)


def unity(tensor: Tensor) -> Tensor:
    """No-op function.

    :param Tensor tensor: Activation tensor.

    :returns: Input tensor unmodified.
    :rtype: Tensor
    """
    logger.debug(f"Unity | IN:   {tensor.shape}")
    return tensor


def broadcast_tensor_parallel(tensor: Tensor) -> Tensor:
    """Send a copy of the tensor to all members of the tensor parallel group.

    :param Tensor tensor: Tensor to broadcast.

    :returns: All tensor parallel ranks will get the same tensor.
    :rtype: Tensor
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

    :param Tensor tensor: Tensor to broadcast.

    :returns: All data parallel ranks will get the same tensor.
    :rtype: Tensor
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

    :param Tensor tensor: Tensor to all-gather.
    :param int dim: Dimension to concatenate the gathered tensors along.

    :returns: The gathered and concatenated tensor.
    :rtype: Tensor
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

    :param Tensor tensor: Tensor to all-gather.
    :param int dim: Dimension to concatenate the gathered tensors along.

    :returns: The gathered and concatenated tensor.
    :rtype: Tensor
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


def _all_reduce_tensor_parallel(tensor: Tensor) -> Tensor:
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


def scatter_tensor_parallel(tensor: Tensor, dim: int = -1) -> Tensor:
    """Chunk a tensor and send chunks to corresponding tensor parallel ranks.

    Given a tensor, chunk it along a specific dimension. Each device rank will
    get the corresponding tensor chunk. For example, T = [T0, T1], where gpu0
    will get T0 and gpu1 will get T1. Notably, the scatter functions are always
    used after all-gather functions. Hence each rank has a full copy of the
    tensor T, so each rank instead discards all chunks besides their own
    corresponding chunk.

    :param Tensor tensor: Tensor to scatter.
    :param int dim: Dimension to chunk the tensor along.

    :returns: The corresponding chunk of the full tensor.
    :rtype: Tensor
    """
    tp_world_size = dist.get_activation_tensor_parallel_world_size()
    tp_rank = dist.get_activation_tensor_parallel_rank()

    if tp_world_size == 1:
        return tensor

    input_list = torch.chunk(tensor, tp_world_size, dim=dim)
    output_tensor = input_list[tp_rank].contiguous()

    logger.debug(f"Scatter | IN: {tensor.shape} -> {output_tensor.shape}")
    return output_tensor


def scatter_data_parallel(tensor: Tensor, dim: int = 0) -> Tensor:
    """Chunk a tensor and send chunks to corresponding data parallel ranks.

    Given a tensor, chunk it along a specific dimension. Each device rank will
    get the corresponding tensor chunk. For example, T = [T0, T1], where gpu0
    will get T0 and gpu1 will get T1. Notably, the scatter functions are always
    used after all-gather functions. Hence each rank has a full copy of the
    tensor T, so each rank instead discards all chunks besides their own
    corresponding chunk.

    :param Tensor tensor: Tensor to scatter.
    :param int dim: Dimension to chunk the tensor along.

    :returns: The corresponding chunk of the full tensor.
    :rtype: Tensor
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


def _group_by_dtype(tensor_dict: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
    fp32_groups = {}
    fp16_groups = {}
    bf16_groups = {}
    for name, tensor in tensor_dict.items():
        if tensor.dtype == torch.float32:
            fp32_groups[name] = tensor
        elif tensor.dtype == torch.float16:
            fp16_groups[name] = tensor
        elif tensor.dtype == torch.bfloat16:
            bf16_groups[name] = tensor
        else:
            raise NotImplementedError(
                f"Tensor with dtype: {tensor.dtype} is not supported for "
                f"gathering across PP ranks."
            )

    dtype_groups = {
        torch.float32: fp32_groups,
        torch.float16: fp16_groups,
        torch.bfloat16: bf16_groups,
    }

    return dtype_groups


def _make_flat_buffer(tensor_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Any]]:
    tensors = []
    name_to_index_map = {}
    name_to_shape_map = {}
    curr_idx = 0
    for name, tensor in tensor_dict.items():
        shape = tensor.shape
        numel = tensor.numel()
        tensors.append(tensor.flatten())

        name_to_index_map[name] = (curr_idx, curr_idx + numel)
        name_to_shape_map[name] = shape

        curr_idx += numel

    if len(tensors) == 0:
        return None, None

    tensor_buffer = torch.cat(tensors)

    meta = {
        "buffer_rank": dist.get_activation_pipeline_parallel_rank(),
        "buffer_size": tensor_buffer.numel(),
        "buffer_dtype": tensor_buffer.dtype,
        "name_to_index_map": name_to_index_map,
        "name_to_shape_map": name_to_shape_map,
    }

    return tensor_buffer, meta


def _gather_pipeline_parallel(
    dtype_to_tensor_buffer: Dict[str, Tensor],
    all_dtype_to_metadata: List[Dict[str, Any]],
) -> Union[Dict[str, List[Tensor]], Dict[str, List[None]]]:
    world_size = dist.get_activation_pipeline_parallel_world_size()
    rank = dist.get_activation_pipeline_parallel_rank()

    # Setup collections for communication
    # Rank 0 uses all of the metadata objects.
    if rank == 0:
        recv_tensors = defaultdict(list)
        recv_ranks = defaultdict(list)
        for dtype_to_metadata in all_dtype_to_metadata:
            for dtype, tensor_buffer_metadata in dtype_to_metadata.items():
                # Skip if there's nothing to receive.
                if tensor_buffer_metadata is None:
                    recv_tensors[dtype] = []
                    recv_ranks[dtype] = []
                    continue

                buffer_rank = tensor_buffer_metadata["buffer_rank"]
                buffer_size = tensor_buffer_metadata["buffer_size"]
                buffer_dtype = tensor_buffer_metadata["buffer_dtype"]
                assert buffer_dtype == dtype

                # Skip rank 0, nothing to receive.
                if buffer_rank == 0:
                    continue

                recv_tensor = torch.empty((buffer_size,), dtype=buffer_dtype)
                recv_rank = buffer_rank

                recv_tensors[dtype].append(recv_tensor)
                recv_ranks[dtype].append(recv_rank)

        send_tensors = {dtype: [] for dtype in dtype_to_tensor_buffer.keys()}
        send_ranks = {dtype: [] for dtype in dtype_to_tensor_buffer.keys()}

    # Other ranks handle their local tensor buffers.
    else:
        recv_tensors = {dtype: [] for dtype in dtype_to_tensor_buffer.keys()}
        recv_ranks = {dtype: [] for dtype in dtype_to_tensor_buffer.keys()}

        send_tensors = {}
        send_ranks = {}
        for dtype, tensor_buffer in dtype_to_tensor_buffer.items():
            # Skip if there's nothing to send.
            if tensor_buffer is None:
                send_tensors[dtype] = []
                send_ranks[dtype] = []
                continue

            send_tensors[dtype] = [tensor_buffer]
            send_ranks[dtype] = [0]

    logger.debug(f"Rank{rank}: recv - {recv_tensors}")
    logger.debug(f"Rank{rank}: recv_ranks - {recv_ranks}")
    logger.debug(f"Rank{rank}: send - {send_tensors}")
    logger.debug(f"Rank{rank}: send_ranks - {send_ranks}")

    # Batched p2p communication for each dtype.
    def _set_device(_buffer_list, device):
        return [_buffer.to(device) for _buffer in _buffer_list]

    for dtype in dtype_to_tensor_buffer.keys():
        recv_tensors_ = _set_device(recv_tensors[dtype], device=torch.cuda.current_device())
        send_tensors_ = _set_device(send_tensors[dtype], device=torch.cuda.current_device())

        batch_isend_irecv_pipeline_parallel(
            recv_tensors_,
            recv_ranks[dtype],
            send_tensors_,
            send_ranks[dtype],
        )

        recv_tensors[dtype] = _set_device(recv_tensors_, device="cpu")
        send_tensors[dtype] = _set_device(send_tensors_, device="cpu")

    # Update input with the gathered tensors.
    updated_tensor_dict = {}
    for dtype in dtype_to_tensor_buffer.keys():
        rank0_tensor = dtype_to_tensor_buffer[dtype]
        gathered_tensors = recv_tensors[dtype]
        updated_tensor_dict[dtype] = [rank0_tensor] if rank0_tensor is not None else []
        updated_tensor_dict[dtype].extend(gathered_tensors)

    return updated_tensor_dict


def batch_isend_irecv_pipeline_parallel(
    recv_tensors: List[Tensor],
    recv_from_ranks: List[int],
    send_tensors: List[Tensor],
    send_to_ranks: List[int],
) -> None:
    world_size = dist.get_activation_pipeline_parallel_world_size()
    rank = dist.get_activation_pipeline_parallel_rank()
    group = dist.get_activation_pipeline_parallel_group()

    assert len(recv_tensors) == len(recv_from_ranks)
    assert len(send_tensors) == len(send_to_ranks)

    p2p_ops = []
    for recv_t, recv_r in zip(recv_tensors, recv_from_ranks):
        op = torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_t,
            peer=recv_r,
            group=group,
        )
        p2p_ops.append(op)

        logger.debug(f"Rank{rank}: P2POp (irecv) [{rank}] <- [{recv_r}]")

    for send_t, send_r in zip(send_tensors, send_to_ranks):
        op = torch.distributed.P2POp(
            torch.distributed.isend,
            send_t,
            peer=send_r,
            group=group,
        )
        p2p_ops.append(op)

        logger.debug(f"Rank{rank}: P2POp (isend) [{rank}] <- [{send_r}]")

    if len(p2p_ops) == 0:
        return

    logger.debug(f"Rank{rank}: Launching P2POps list - {p2p_ops}")

    reqs = torch.distributed.batch_isend_irecv(p2p_ops)
    for req in reqs:
        req.wait()

    # TODO: Remove after verification that no race cond. occurs.
    torch.cuda.synchronize()


def gather_pipeline_parallel_tensor_dicts(
    tensor_dict: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """Gather tensors from non-zero ranks to rank0 in the pipeline group.

    :param tensor_dict: Some python object that can be pickled. May contain tensors.
    :type tensor_dict Dict[str, Tensor]:

    :returns: A collection of the objects sent from all pipeline paralel group ranks.
    :rtype: Dict[str, Tensor]
    """
    if not dist.in_pipeline_parallel_group():
        return tensor_dict

    world_size = dist.get_activation_pipeline_parallel_world_size()
    rank = dist.get_activation_pipeline_parallel_rank()
    group = dist.get_activation_pipeline_parallel_group()

    dtype_to_tensor_dict = _group_by_dtype(tensor_dict)

    # Convert tensor dicts into flattened buffers with metadata.
    dtype_to_tensor_buffer = {}
    dtype_to_metadata = {}
    for dtype, tensor_dict in dtype_to_tensor_dict.items():
        tensor_buffer, metadata = _make_flat_buffer(tensor_dict)

        if tensor_buffer is not None and metadata is not None:
            assert tensor_buffer.dtype == metadata["buffer_dtype"] == dtype, (
                f"Dtype mismatch: {tensor_buffer.dtype}, {metadata['buffer_dtype']}, "
                f"{dtype}."
            )

        dtype_to_tensor_buffer[dtype] = tensor_buffer
        dtype_to_metadata[dtype] = metadata

    # Gather metadata on rank 0 to setup recv tensors.
    all_dtype_to_metadata: List[Dict[str, Any]] = [
        None for _ in range(world_size)
    ]
    torch.distributed.gather_object(
        dtype_to_metadata,
        all_dtype_to_metadata if rank == 0 else None,
        dst=0,
        group=group,
    )

    # Communicate.
    dtype_to_all_tensor_buffers = _gather_pipeline_parallel(
        dtype_to_tensor_buffer,
        all_dtype_to_metadata,
    )

    # Re-constitute tensor buffers.
    tensor_dict = {}
    if rank == 0:
        # Change map to: dtype -> list of meta for all ranks.
        dtype_to_all_tensor_buffer_metadata = defaultdict(list)
        for dtype_to_meta in all_dtype_to_metadata:
            for dtype, meta in dtype_to_meta.items():
                dtype_to_all_tensor_buffer_metadata[dtype].append(meta)

        # Unflatten buffers and separate the tensors by index.
        for dtype, tensor_buffers in dtype_to_all_tensor_buffers.items():
            metadata = dtype_to_all_tensor_buffer_metadata[dtype]

            for t_buf, meta in zip(tensor_buffers, metadata):
                for name in meta["name_to_index_map"].keys():
                    start, end = meta["name_to_index_map"][name]
                    shape = meta["name_to_shape_map"][name]
                    tensor = t_buf[start: end].reshape(shape)
                    tensor_dict[name] = tensor

    return tensor_dict
