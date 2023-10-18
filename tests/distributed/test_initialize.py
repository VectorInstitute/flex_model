import logging

import fairscale.nn.model_parallel as mpu
import torch
import torch.distributed as dist

import flex_model.distributed as fm_dist
from flex_model.distributed.backends import GPUDeviceMesh
from flex_model.utils import setup_logger
from tests.test_utilities import Utils

logger = logging.getLogger(__name__)


def test_GPUDeviceMesh():
    cases = [
        (1, 1, 1),
        (4, 1, 1),
        (1, 4, 1),
        (1, 1, 4),
        (4, 4, 1),
        (1, 4, 4),
        (4, 1, 4),
        (2, 2, 2),
        (2, 4, 2),
    ]
    # (tp, pp, dp)
    solutions = [
        [[[0]], [[0]], [[0]],],
        [[[0, 1, 2, 3]], [[0]], [[0]],],
        [[[0], [1], [2], [3]], [[0, 1, 2, 3]], [[0], [1], [2], [3]],],
        [[[0], [1], [2], [3]], [[0]], [[0, 1, 2, 3]],],
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [[0, 4, 8, 12]],
            [[0], [4], [8], [12]],
        ],
        [
            [[i] for i in range(16)],
            # [[0, 1, 2, 3]],
            [[0, 4, 8, 12]],
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        ],
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [[0]],
            [[0, 4, 8, 12]],
        ],
        [[[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 4]], [[0, 2], [4, 6]],],
        [
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
            [[0, 4, 8, 12]],
            [[0, 2], [4, 6], [8, 10], [12, 14]],
        ],
    ]
    for case, solution in zip(cases, solutions):
        tp = case[0]
        pp = case[1]
        dp = case[2]
        world_size = tp * pp * dp
        gpu_device_mesh = GPUDeviceMesh.build(world_size, tp, pp, dp)
        assert gpu_device_mesh.tp_group_ranks == solution[0], f"{case}"
        assert gpu_device_mesh.pp_group_ranks == solution[1], f"{case}"
        assert gpu_device_mesh.dp_group_ranks == solution[2], f"{case}"


def test_initialize_and_destroy_activation_parallel():
    Utils.initialize_distributed()
    Utils.initialize_model_parallel(2, 1, 2)

    Utils.initialize_distributed_backend(2, 1, 2)
    assert fm_dist.distributed_backend_is_initialized()
    assert not fm_dist.activation_parallel_is_initialized()

    Utils.initialize_activation_parallel()
    assert fm_dist.distributed_backend_is_initialized()
    assert fm_dist.activation_parallel_is_initialized()

    Utils.destroy_activation_parallel()
    assert not fm_dist.activation_parallel_is_initialized()
    assert fm_dist.distributed_backend_is_initialized()

    Utils.destroy_distributed_backend()
    assert not fm_dist.distributed_backend_is_initialized()

    Utils.destroy_model_parallel()
    Utils.destroy_distributed()


test_GPUDeviceMesh()
test_initialize_and_destroy_activation_parallel()
