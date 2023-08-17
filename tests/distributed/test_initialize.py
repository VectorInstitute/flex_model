import logging

import torch
import torch.distributed as dist
import fairscale.nn.model_parallel as mpu

from tests.test_utilities import Utils
import flex_model.distributed as fm_dist
from flex_model.utils import setup_logger


logger = logging.getLogger(__name__)


def test_initialize_and_destroy_activation_parallel():
    Utils.initialize_distributed()
    Utils.initialize_model_parallel(2, 1, 2)
    Utils.initialize_distributed_backend(2, 1, 2)
    Utils.initialize_activation_parallel()

    tp_rank = fm_dist.get_activation_tensor_parallel_rank()
    tp_world_size = fm_dist.get_activation_tensor_parallel_world_size()

    # Data parallel groups only init on tp rank 0
    if tp_rank == 0:
        dp_rank = fm_dist.get_activation_data_parallel_rank()
        dp_world_size = fm_dist.get_activation_data_parallel_world_size()

        assert fm_dist.get_activation_data_parallel_group() is not None
        assert mpu.get_data_parallel_rank() == dp_rank, (
            f"Rank{dist.get_rank()} mismatch: DP - {mpu.get_data_parallel_rank()} "
            f"DP - {dp_rank}"
        )
        assert mpu.get_data_parallel_world_size() == dp_world_size, (
            f"Rank mismatch: DP - {mpu.get_data_parallel_world_size()} "
            f"DP - {dp_world_size}"
        )
        assert dp_world_size == 2, f"Got: {dp_world_size}"

    else:
        assert fm_dist.get_activation_data_parallel_group() is None

    assert fm_dist.activation_parallel_is_initialized()
    assert fm_dist.get_activation_tensor_parallel_group() is not None
    assert tp_world_size == 2, f"Got: {tp_world_size}"

    assert mpu.get_model_parallel_rank() == tp_rank, (
        f"Rank mismatch: MP - {mpu.get_model_parallel_rank()} "
        f"TP - {tp_rank}"
    )

    assert mpu.get_model_parallel_world_size() == tp_world_size, (
        f"Rank mismatch: MP - {mpu.get_model_parallel_world_size()} "
        f"TP - {tp_world_size}"
    )
    Utils.destroy_activation_parallel()
    assert not fm_dist.activation_parallel_is_initialized()
    assert fm_dist.distributed_backend_is_initialized()

    Utils.destroy_model_parallel()
    Utils.destroy_distributed_backend()
    assert not fm_dist.distributed_backend_is_initialized()
    Utils.destroy_distributed()


setup_logger("debug")
test_initialize_and_destroy_activation_parallel()
