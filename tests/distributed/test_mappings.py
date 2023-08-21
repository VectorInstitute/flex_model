import logging

import torch
import torch.distributed as dist
import fairscale.nn.model_parallel as mpu

from tests.test_utilities import Utils
import flex_model.distributed as fm_dist
from flex_model.utils import setup_logger
from flex_model.distributed.backends import GPUDeviceMesh


logger = logging.getLogger(__name__)


def test_broadcast_tensor_parallel():
    Utils.initialize_model_parallel(2, 1, 1)
    Utils.initialize_distributed_backend(2, 1, 1)
    Utils.initialize_activation_parallel()

    if fm_dist.get_activation_tensor_parallel_rank() == 0:
        tensor_to_bcast = torch.ones((1)).cuda()
    else:
        tensor_to_bcast = torch.zeros((1,)).cuda()
    result = fm_dist.broadcast_tensor_parallel(tensor_to_bcast)
    assert torch.equal(result, torch.ones((1)).cuda())

    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


def test_broadcast_data_parallel():
    Utils.initialize_model_parallel(1, 1, 2)
    Utils.initialize_distributed_backend(1, 1, 2)
    Utils.initialize_activation_parallel()

    if fm_dist.get_activation_data_parallel_rank() == 0:
        tensor_to_bcast = torch.ones((1)).cuda()
    else:
        tensor_to_bcast = torch.zeros((1,)).cuda()
    result = fm_dist.broadcast_data_parallel(tensor_to_bcast)
    assert torch.equal(result, torch.ones((1)).cuda())

    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


def test_all_gather_tensor_parallel():
    Utils.initialize_model_parallel(2, 1, 1)
    Utils.initialize_distributed_backend(2, 1, 1)
    Utils.initialize_activation_parallel()

    tensor_to_gather = (
        torch.ones((1)).cuda() * fm_dist.get_activation_tensor_parallel_rank()
    )
    result = fm_dist.all_gather_tensor_parallel(tensor_to_gather)
    assert torch.equal(
        result, torch.arange(fm_dist.get_activation_tensor_parallel_world_size()).cuda()
    )

    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


def test_all_gather_data_parallel():
    Utils.initialize_model_parallel(1, 1, 2)
    Utils.initialize_distributed_backend(1, 1, 2)
    Utils.initialize_activation_parallel()

    tensor_to_gather = (
        torch.ones((1)).cuda() * fm_dist.get_activation_data_parallel_rank()
    )
    result = fm_dist.all_gather_data_parallel(tensor_to_gather)
    assert torch.equal(
        result, torch.arange(fm_dist.get_activation_data_parallel_world_size()).cuda()
    )

    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


def test_all_reduce_tensor_parallel():
    Utils.initialize_model_parallel(2, 1, 1)
    Utils.initialize_distributed_backend(2, 1, 1)
    Utils.initialize_activation_parallel()

    tensor_to_reduce = torch.ones((1)).cuda()
    result = fm_dist.all_reduce_tensor_parallel(tensor_to_reduce)
    assert torch.equal(
        result,
        torch.ones((1)).cuda() * fm_dist.get_activation_tensor_parallel_world_size(),
    )

    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


def test_scatter_tensor_parallel():
    Utils.initialize_model_parallel(2, 1, 1)
    Utils.initialize_distributed_backend(2, 1, 1)
    Utils.initialize_activation_parallel()

    tensor_to_scatter = torch.arange(
        fm_dist.get_activation_tensor_parallel_world_size()
    ).cuda()
    result = fm_dist.scatter_tensor_parallel(tensor_to_scatter)
    assert torch.equal(
        result, torch.ones((1)).cuda() * fm_dist.get_activation_tensor_parallel_rank()
    )

    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


def test_scatter_data_parallel():
    Utils.initialize_model_parallel(1, 1, 2)
    Utils.initialize_distributed_backend(1, 1, 2)
    Utils.initialize_activation_parallel()

    tensor_to_scatter = torch.arange(
        fm_dist.get_activation_data_parallel_world_size()
    ).cuda()
    result = fm_dist.scatter_data_parallel(tensor_to_scatter)
    assert torch.equal(
        result, torch.ones((1)).cuda() * fm_dist.get_activation_data_parallel_rank()
    )

    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


def test_gather_pipeline_parallel():
    Utils.initialize_model_parallel(1, 2, 1)
    Utils.initialize_distributed_backend(1, 2, 1)
    Utils.initialize_activation_parallel()

    rank = fm_dist.get_activation_pipeline_parallel_rank()
    obj_to_gather = {f"obj{rank}": torch.ones((1)).cuda() * rank}
    result = fm_dist.gather_pipeline_parallel(obj_to_gather)

    if rank == 0:
        for i, obj in enumerate(result):
            assert isinstance(obj, dict)
            for k, v in obj.items():
                assert f"obj{i}" == k, f"Key mismatch: obj{i} - {k}"
                assert torch.equal(v.cuda(), torch.ones((1)).cuda() * i)

    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


setup_logger("debug")
test_broadcast_tensor_parallel()
test_broadcast_data_parallel()
test_all_gather_tensor_parallel()
test_all_gather_data_parallel()
test_all_reduce_tensor_parallel()
test_scatter_tensor_parallel()
test_scatter_data_parallel()
test_gather_pipeline_parallel()
