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


def _group_by_dtype(
    tensor_dict: Dict[str, Tensor]
) -> Dict[torch.dtype, Dict[str, Tensor]]:
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    dtype_groups: Dict[torch.dtype, Dict[str, Tensor]] = {
        dtype: {}
        for dtype in dtypes
    }

    for name, tensor in tensor_dict.items():
        assert tensor.dtype in dtype_groups, (
            f"Tensor with dtype: {tensor.dtype} is not supported for "
            f"gathering across PP ranks."
        )
        dtype_groups[tensor.dtype][name] = tensor

    return dtype_groups


# Tensor buffer metadata type.
_TBUF_META = Dict[
    str, Union[
        int,
        torch.dtype,
        Dict[str, Tuple[int, int]],
        Dict[str, torch.Size],
    ]
]


def _make_flat_buffer(
    tensor_dict: Dict[str, Tensor],
) -> Tuple[Optional[Tensor], Optional[_TBUF_META]]:
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

    meta: _TBUF_META = {
        "buffer_rank": dist.get_activation_pipeline_parallel_rank(),
        "buffer_size": tensor_buffer.numel(),
        "buffer_dtype": tensor_buffer.dtype,
        "name_to_index_map": name_to_index_map,
        "name_to_shape_map": name_to_shape_map,
    }

    return tensor_buffer, meta


def _gather_pipeline_parallel(
    tbuf_groups,
    all_metadata_groups,
) -> Dict[str, Tensor]:
    world_size = dist.get_activation_pipeline_parallel_world_size()
    rank = dist.get_activation_pipeline_parallel_rank()

    # Setup collections for communication
    def _empty_groups() -> Dict[torch.dtype, List[Union[Tensor, int]]]:
        return {dtype: [] for dtype in tbuf_groups.keys()}

    recv_tbuf_groups = _empty_groups()
    recv_rank_groups = _empty_groups()
    send_tbuf_groups = _empty_groups()
    send_rank_groups = _empty_groups()

    # Construct recv tensors and src ranks.
    # NOTE: Only rank0 participates in recvs.
    if rank == 0:
        for metadata_groups in all_metadata_groups:
            # Skip if the rank has no tbufs to recv for any dtype.
            if metadata_groups is not None:
                continue

            for dtype, metadata in metadata_groups.items():
                # Skip if there's no tbuf to recv for the dtype or the source
                # rank is 0 (rank0 never sends).
                if metadata is None or metadata["buffer_rank"] == 0:
                    continue

                buffer_rank = metadata["buffer_rank"]
                buffer_size = metadata["buffer_size"]
                buffer_dtype = metadata["buffer_dtype"]
                assert buffer_dtype == dtype, (
                    f"Dtype mismatch: {buffer_dtype} and {dtype}")

                tbuf = torch.empty((buffer_size,), dtype=buffer_dtype)
                src_rank = buffer_rank
                recv_tbuf_groups[dtype].append(tbuf)
                recv_rank_groups[dtype].append(src_rank)

                logger.debug(
                    f"Rank{rank}: Constructed recv - "
                    f"({tbuf.numel()}) [{src_rank}] -> [0]"
                )

    # Construct send tensors and dst ranks.
    # NOTE: Only non-rank0 participate in sends.
    else:
        for dtype, tbuf in tbuf_groups.items():
            # Skip if there's no tbuf to send for the dtype.
            if tbuf is None:
                continue

            # Send dst always rank0.
            send_tbuf_groups[dtype].append(tbuf)
            send_rank_groups[dtype].append(0)

            logger.debug(
                f"Rank{rank}: Constructed send - "
                f"({tbuf.numel()}) [{rank}] -> [0]"
            )

    def _set_device(_buffer_list, device):
        return [_buffer.to(device) for _buffer in _buffer_list]

    # Batched communication across all dtype groups.
    all_recv_tbufs = []
    all_recv_ranks = []
    all_send_tbufs = []
    all_send_ranks = []
    for dtype in tbuf_groups.keys():
        recv_tbufs = _set_device(recv_tbuf_groups[dtype], device=torch.cuda.current_device())
        send_tbufs = _set_device(send_tbuf_groups[dtype], device=torch.cuda.current_device())
        all_recv_tbufs.extend(recv_tbufs)
        all_recv_ranks.extend(recv_rank_groups[dtype])
        all_send_tbufs.extend(send_tbufs)
        all_send_ranks.extend(send_rank_groups[dtype])

    batch_isend_irecv_pipeline_parallel(
        all_recv_tbufs,
        all_recv_ranks,
        all_send_tbufs,
        all_send_ranks,
    )
    all_recv_tbufs = _set_device(all_recv_tbufs, device="cpu")
    all_send_tbufs = _set_device(all_send_tbufs, device="cpu")

    # Unshard each tbuf into individual tensors.
    output_tensor_dict: Dict[str, Tensor] = {}
    if rank == 0:
        def _reshard_tbuf(meta, tbuf):
            for name, (start, end) in meta["name_to_index_map"].items():
                shape = meta["name_to_shape_map"][name]
                output_tensor_dict[name] = tbuf[start: end].reshape(shape)

        # Add rank0 local tbufs.
        for dtype, tbuf in tbuf_groups.items():
            meta = all_metadata_groups[0][dtype]
            if meta is not None:
                _reshard_tbuf(meta, tbuf)

        # Add gathered tbufs.
        for recv_tbuf, recv_r in zip(all_recv_tbufs, all_recv_ranks):
            dtype = recv_tbuf.dtype
            meta = all_metadata_groups[recv_r][dtype]

            buf_rank = meta["buffer_rank"]
            buf_dtype = meta["buffer_dtype"]
            assert buf_dtype == dtype, (
                f"Dtype mismatch: {buf_dtype} and {dtype}")
            assert buf_rank == recv_r, (
                f"Rank mismatch: {buf_rank} and {recv_r}")

            _reshard_tbuf(meta, recv_tbuf)

    return output_tensor_dict


def batch_isend_irecv_pipeline_parallel(
    recv_tensors: List[Tensor],
    recv_from_ranks: List[int],
    send_tensors: List[Tensor],
    send_to_ranks: List[int],
) -> None:
    """Run batched peer-to-peer communications.

    :param List[Tensor] recv_tensors: Tensors to receive.
    :param List[int] recv_from_ranks: Ranks to receive from.
    :param List[Tensor] send_tensors: Tensors to send.
    :param List[int] send_to_ranks: Ranks to send to.
    """
    world_size = dist.get_activation_pipeline_parallel_world_size()
    rank = dist.get_activation_pipeline_parallel_rank()
    group = dist.get_activation_pipeline_parallel_group()

    assert len(recv_tensors) == len(recv_from_ranks), (
        f"Mistmatch in recv tensors({len(recv_tensors)}) and "
        f"recv ranks({len(recv_from_ranks)})"
    )
    assert len(send_tensors) == len(send_to_ranks), (
        f"Mistmatch in send tensors({len(send_tensors)}) and "
        f"send ranks({len(send_to_ranks)})"
    )

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

        logger.debug(f"Rank{rank}: P2POp (isend) [{rank}] -> [{send_r}]")

    if len(p2p_ops) == 0:
        return

    logger.debug(f"Rank{rank}: Launching P2POps")

    reqs = torch.distributed.batch_isend_irecv(p2p_ops)
    for req in reqs:
        req.wait()

    _msg = lambda t_list: ", ".join([f"({t.numel()}, {t.dtype})" for t in t_list])
    logger.debug(f"Rank{rank}: Received buffers - [{_msg(recv_tensors)}]")
    logger.debug(f"Rank{rank}: Sent buffers - [{_msg(send_tensors)}]")

    # TODO: Remove after verification that no race cond. occurs.
    torch.cuda.synchronize()


def gather_pipeline_parallel_tensor_dicts(
    tensor_dict: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """Gather tensors from non-zero ranks to global rank0 in the pipeline group.

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

    tensor_dict_groups = _group_by_dtype(tensor_dict)

    # Convert tensor dicts into flattened buffers with metadata.
    tbuf_groups = {}
    metadata_groups = {}
    for dtype, tensor_dict in tensor_dict_groups.items():
        tbuf, meta = _make_flat_buffer(tensor_dict)

        tbuf_groups[dtype] = tbuf
        metadata_groups[dtype] = meta

    # Gather metadata on rank 0 to setup recv tensors.
    all_metadata_groups: List[Optional[Dict[torch.dtype, _TBUF_META]]] = [
        None for _ in range(world_size)
    ]
    torch.distributed.gather_object(
        metadata_groups,
        all_metadata_groups if rank == 0 else None,
        dst=0,
        group=group,
    )

    # Communicate.
    output_tensor_dict = _gather_pipeline_parallel(
        tbuf_groups,
        all_metadata_groups,
    )

    return output_tensor_dict
