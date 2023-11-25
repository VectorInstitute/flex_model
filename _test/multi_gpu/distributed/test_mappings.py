import logging

import torch
import torch.nn as nn
import torch.distributed as dist

import flex_model.distributed as fm_dist
from _test.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry

logger = logging.getLogger(__name__)


_NUM_GPUS = 2


register_mappings_test, get_mappings_test = make_test_registry(
    "mappings",
    SlurmJobResourceSpec(
        gpus_per_node=_NUM_GPUS,
        ntasks_per_node=_NUM_GPUS,
    ),
)


@register_mappings_test
def test_broadcast_tensor_parallel():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, _NUM_GPUS, 1, 1)

    if fm_dist.get_tensor_parallel_rank() == 0:
        tensor_to_bcast = torch.ones((1)).cuda()
    else:
        tensor_to_bcast = torch.zeros((1,)).cuda()
    result = fm_dist.broadcast_tensor_parallel(tensor_to_bcast, fmps)
    assert torch.equal(result, torch.ones((1)).cuda())

    dist.destroy_process_group()


@register_mappings_test
def test_broadcast_data_parallel():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, 1, 1, _NUM_GPUS)

    if fm_dist.get_data_parallel_rank() == 0:
        tensor_to_bcast = torch.ones((1)).cuda()
    else:
        tensor_to_bcast = torch.zeros((1,)).cuda()
    result = fm_dist.broadcast_data_parallel(tensor_to_bcast, fmps)
    assert torch.equal(result, torch.ones((1)).cuda())

    dist.destroy_process_group()


@register_mappings_test
def test_all_gather_tensor_parallel():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, _NUM_GPUS, 1, 1)

    tensor_to_gather = (
        torch.ones((1)).cuda() * fm_dist.get_tensor_parallel_rank()
    )
    result = fm_dist.all_gather_tensor_parallel(tensor_to_gather, 0, fmps)
    assert torch.equal(
        result,
        torch.arange(fm_dist.get_tensor_parallel_world_size()).cuda(),
    )

    dist.destroy_process_group()


@register_mappings_test
def test_all_gather_data_parallel():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, 1, 1, _NUM_GPUS)

    tensor_to_gather = torch.ones((1)).cuda() * fm_dist.get_data_parallel_rank()
    result = fm_dist.all_gather_data_parallel(tensor_to_gather, 0, fmps)
    assert torch.equal(
        result,
        torch.arange(fm_dist.get_data_parallel_world_size()).cuda(),
    )

    dist.destroy_process_group()


@register_mappings_test
def test_scatter_tensor_parallel():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, _NUM_GPUS, 1, 1)

    tensor_to_scatter = torch.arange(
        fm_dist.get_tensor_parallel_world_size()
    ).cuda()
    result = fm_dist.scatter_tensor_parallel(tensor_to_scatter, 0, fmps)
    assert torch.equal(
        result,
        torch.ones((1)).cuda() * fm_dist.get_tensor_parallel_rank(),
    )

    dist.destroy_process_group()


@register_mappings_test
def test_scatter_data_parallel():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, 1, 1, _NUM_GPUS)

    tensor_to_scatter = torch.arange(
        fm_dist.get_data_parallel_world_size()
    ).cuda()
    result = fm_dist.scatter_data_parallel(tensor_to_scatter, 0, fmps)
    assert torch.equal(
        result,
        torch.ones((1)).cuda() * fm_dist.get_data_parallel_rank(),
    )

    dist.destroy_process_group()


@register_mappings_test
def test_batch_isend_irecv_pipeline_parallel():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, 1, _NUM_GPUS, 1)

    rank = fmps.get_pipeline_parallel_rank()
    world_size = fmps.get_pipeline_parallel_world_size()

    send_tensors = [torch.ones((1,)).cuda() * rank]
    send_to_ranks = [(rank + 1) % world_size]
    recv_tensors = [torch.empty((1,)).cuda()]
    recv_from_ranks = [(rank + 1) % world_size]

    fm_dist.batch_isend_irecv_pipeline_parallel(
        fmps,
        recv_tensors,
        recv_from_ranks,
        send_tensors,
        send_to_ranks,
    )

    for tensor in recv_tensors:
        assert torch.equal(
            tensor,
            torch.ones((1,)).cuda() * (rank + 1) % world_size,
        )

    dist.destroy_process_group()


@register_mappings_test
def test_gather_pipeline_parallel_base():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, 1, _NUM_GPUS, 1)

    rank = fmps.get_pipeline_parallel_rank()
    world_size = fmps.get_pipeline_parallel_world_size()

    # Test on empty data.
    tensor_dict = {}
    result = fm_dist.gather_pipeline_parallel_tensor_dicts(fmps, tensor_dict)
    assert len(result) == 0

    # Test on multiple tensors per rank.
    tensor_dict = {}
    tensors_per_rank = 4
    for i in range(tensors_per_rank):
        tensor_idx = rank * tensors_per_rank + i
        tensor_dict[f"tensor_{tensor_idx}"] = torch.ones((1,)) * tensor_idx

    result = fm_dist.gather_pipeline_parallel_tensor_dicts(fmps, tensor_dict)

    if rank == 0:
        assert len(result) == tensors_per_rank * world_size
        for tensor_idx in range(world_size * tensors_per_rank):
            assert torch.equal(
                result[f"tensor_{tensor_idx}"],
                torch.ones((1,)) * tensor_idx,
            )

    dist.destroy_process_group()


@register_mappings_test
def test_gather_pipeline_parallel_dtypes():
    dist.init_process_group("nccl")

    model = nn.Linear(2, 4)
    fmps = fm_dist.initialize_distributed_state(model, 1, _NUM_GPUS, 1)

    rank = fmps.get_pipeline_parallel_rank()
    world_size = fmps.get_pipeline_parallel_world_size()

    tensor_dict = {}
    tensors_per_rank = 4
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    for dtype in dtypes:
        for i in range(tensors_per_rank):
            tensor_idx = rank * tensors_per_rank + i
            name = f"tensor_{tensor_idx}_{dtype}"
            tensor = torch.ones((1,), dtype=dtype)
            tensor_dict[name] = tensor

    result = fm_dist.gather_pipeline_parallel_tensor_dicts(fmps, tensor_dict)

    if rank == 0:
        assert len(result) == tensors_per_rank * world_size * len(dtypes)
        for dtype in dtypes:
            for i in range(tensors_per_rank):
                tensor_idx = rank * tensors_per_rank + i
                name = f"tensor_{tensor_idx}_{dtype}"
                tensor = torch.ones((1,), dtype=dtype)
                assert torch.equal(
                    result[name],
                    tensor,
                )

    dist.init_process_group()
