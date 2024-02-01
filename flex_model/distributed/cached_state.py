from typing import Dict, Union, Optional, Tuple, List
import logging

import torch
from torch import Tensor

from flex_model.distributed.distributed_state import _ParallelStateAPI
from flex_model.distributed.mappings import _log_shape
from flex_model.distributed.stratgies import (
    SaveCtxStrategy,
)

logger = logging.getLogger(__name__)


def _group_by_dtype(
    tensors: Dict[str, Tensor]
) -> Dict[torch.dtype, Dict[str, Tensor]]:
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    dtype_groups = {dtype: {} for dtype in dtypes}

    for name, tensor in tensors.items():
        assert tensor.dtype in dtype_groups, (
            f"Tensor with dtype: {tensor.dtype} is not supported for "
            f"gathering across PP ranks."
        )
        dtype_groups[tensor.dtype][name] = tensor

    return dtype_groups


# Tensor buffer metadata type.
_TBUF_META = Dict[
    str,
    Union[
        int,
        torch.dtype,
        Dict[str, Tuple[int, int]],
        Dict[str, torch.Size],
    ],
]


def _make_flat_buffer(
    fmps: _ParallelStateAPI,
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
        "buffer_rank": fmps.get_pipeline_parallel_rank(),
        "buffer_size": tensor_buffer.numel(),
        "buffer_dtype": tensor_buffer.dtype,
        "name_to_index_map": name_to_index_map,
        "name_to_shape_map": name_to_shape_map,
    }

    return tensor_buffer, meta


def _broadcast_pipeline_parallel(
    fmps: _ParallelStateAPI,
    tbuf_groups: Dict[torch.dtype, Optional[Tensor]],
    all_metadata_groups: List[Optional[Dict[torch.dtype, _TBUF_META]]],
) -> Dict[str, Tensor]:
    rank = fmps.get_pipeline_parallel_rank()

    # Setup collections for communication
    def _empty_groups() -> Dict[torch.dtype, List[Union[Tensor, int]]]:
        return {dtype: [] for dtype in tbuf_groups.keys()}

    recv_tbuf_groups = _empty_groups()
    recv_rank_groups = _empty_groups()
    send_tbuf_groups = _empty_groups()
    send_rank_groups = _empty_groups()

    # Construct recv tensors and src ranks.
    for metadata_groups in all_metadata_groups:
        # Skip if the rank has no tbufs to recv for any dtype.
        if metadata_groups is None:
            continue

        for dtype, metadata in metadata_groups.items():
            # Skip if there's no tbuf to recv for the dtype or the source
            # rank is 0 (rank0 never sends).
            if metadata is None:
                continue

            buffer_rank = metadata["buffer_rank"]
            buffer_size = metadata["buffer_size"]
            buffer_dtype = metadata["buffer_dtype"]
            assert (
                buffer_dtype == dtype
            ), f"Dtype mismatch: {buffer_dtype} and {dtype}"

            tbuf = torch.empty((buffer_size,), dtype=buffer_dtype)
            src_rank = buffer_rank
            recv_tbuf_groups[dtype].append(tbuf)
            recv_rank_groups[dtype].append(src_rank)

            logger.debug(
                f"Rank{rank}: Constructed recv - "
                f"({tbuf.numel()}) [{src_rank}] -> [{fmps.get_rank()}]"
            )

    # Construct send tensors and dst ranks.
    for dtype, tbuf in tbuf_groups.items():
        # Skip if there's no tbuf to send for the dtype.
        if tbuf is None:
            continue

        # Send dst always rank0.
        for r in fmps.get_world_size():
            send_tbuf_groups[dtype].append(tbuf)
            send_rank_groups[dtype].append(r)

            logger.debug(
                f"Rank{rank}: Constructed send - "
                f"({tbuf.numel()}) [{rank}] -> [{r}]"
            )

    def _set_device(_buffer_list, device):
        return [_buffer.to(device) for _buffer in _buffer_list]

    # Batched communication across all dtype groups.
    all_recv_tbufs = []
    all_recv_ranks = []
    all_send_tbufs = []
    all_send_ranks = []
    for dtype in tbuf_groups.keys():
        recv_tbufs = _set_device(
            recv_tbuf_groups[dtype], device=torch.cuda.current_device()
        )
        send_tbufs = _set_device(
            send_tbuf_groups[dtype], device=torch.cuda.current_device()
        )
        all_recv_tbufs.extend(recv_tbufs)
        all_recv_ranks.extend(recv_rank_groups[dtype])
        all_send_tbufs.extend(send_tbufs)
        all_send_ranks.extend(send_rank_groups[dtype])

    batch_isend_irecv_pipeline_parallel(
        fmps,
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
                output_tensor_dict[name] = tbuf[start:end].reshape(shape)

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
            assert (
                buf_dtype == dtype
            ), f"Dtype mismatch: {buf_dtype} and {dtype}"
            assert buf_rank == recv_r, f"Rank mismatch: {buf_rank} and {recv_r}"

            _reshard_tbuf(meta, recv_tbuf)

    return output_tensor_dict


def batch_isend_irecv_pipeline_parallel(
    fmps: _ParallelStateAPI,
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
    rank = fmps.get_pipeline_parallel_rank()
    group = fmps.get_pipeline_parallel_group()

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

    def _gen_debug_msg(t_list):
        return ", ".join([f"({t.numel()}, {t.dtype})" for t in t_list])

    logger.debug(
        f"Rank{rank}: Received buffers - [{_gen_debug_msg(recv_tensors)}]"
    )
    logger.debug(f"Rank{rank}: Sent buffers - [{_gen_debug_msg(send_tensors)}]")

    # TODO: Remove after verification that no race cond. occurs.
    torch.cuda.synchronize()


def gather_pipeline_parallel_tensor_dicts(
    fmps: _ParallelStateAPI,
    tensor_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Gather groups of tensors from ranks of the pipeline group to pipeline rank0.

    Note: Assumes input tensors are on CPU and placed output tensors on CPU.
    - This behaviour is subject to change depending on various optimizations.

    :param _ParallelStateAPI fmps: FlexModel parallel state handle.
    :param tensor_dict: Some python object that can be pickled. May contain tensors.
    :type tensor_dict Dict[str, Tensor]:

    :returns: A collection of the objects sent from all pipeline paralel group ranks.
    :rtype: Dict[str, Tensor]
    """
    in_shapes = []
    for tensor in tensor_dict.values():
        in_shapes.append(tensor.shape)

    world_size = fmps.get_pipeline_parallel_world_size()
    rank = fmps.get_pipeline_parallel_rank()
    group = fmps.get_pipeline_parallel_group()

    tensor_dict_groups = _group_by_dtype(tensor_dict)

    # Convert tensor dicts into flattened buffers with metadata.
    tbuf_groups = {}
    metadata_groups = {}
    for dtype, tensor_dict in tensor_dict_groups.items():
        tbuf, meta = _make_flat_buffer(fmps, tensor_dict)

        tbuf_groups[dtype] = tbuf
        metadata_groups[dtype] = meta

    # Gather metadata on all ranks to setup recv tensors.
    all_metadata_groups: List[Optional[Dict[torch.dtype, _TBUF_META]]] = [
        None for _ in range(world_size)
    ]
    torch.distributed.all_gather_object(
        all_metadata_groups,
        metadata_groups,
        group=group,
    )

    # Communicate.
    output_tensor_dict = _broadcast_pipeline_parallel(
        fmps, tbuf_groups, all_metadata_groups
    )

    for in_shape, out_tensor in zip(in_shapes, output_tensor_dict.values()):
        _log_shape(
            rank,
            "gather_pipeline_parallel_tensor_dicts",
            in_shape,
            out_tensor.shape,
        )

    return output_tensor_dict


def _pipeline_sync(tensor_dict: Dict[str, Tensor], fmps: _ParallelStateAPI):
    gather_pipeline_parallel_tensor_dicts(tensor_dict, fmps)


class SaveContext:
    def __init__(
        self,
        fmps: _ParallelStateAPI,
        strategy: SaveCtxStrategy = SaveCtxStrategy.REPLICATE_ALL,
    ):
        self.strategy = strategy
        self.fmps = fmps
        self.cached_tensors = {}

    def save(self, name: str, tensor: Tensor):
        self.cached_tensors[name] = tensor

    def get_tensor(self, name: str) -> Tensor:
        return self.cached_tensors[name]

    def sync(self):
        _pipeline_sync(self.cached_tensors, self.fmps)
