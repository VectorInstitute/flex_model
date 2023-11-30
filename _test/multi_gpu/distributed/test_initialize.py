import torch.nn as nn
import torch.distributed as dist

from flex_model.distributed.distributed_state import (
    _global_state_is_initialized,
)
import flex_model.distributed as fm_dist
from _test.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry
import _test.multi_gpu.testing_utils as utils

register_initialize_test, get_initialize_test = make_test_registry(
    "initialize",
    SlurmJobResourceSpec(),
)


@register_initialize_test
def test_initialize_distributed_state():
    utils.init_process_group()

    # model = utils.TestModel().cuda()
    # model = utils.wrap_ddp(model, rank=dist.get_rank())
    model = nn.Linear(2, 4)

    fmps = fm_dist.initialize_distributed_state(
        model, 1, 1, dist.get_world_size()
    )
    assert _global_state_is_initialized()

    assert fmps.get_local_rank() == dist.get_rank()
    assert fmps.get_local_world_size() == dist.get_world_size()
    assert fmps.get_subset_rank() == dist.get_rank()
    assert fmps.get_subset_world_size() == dist.get_world_size()
    assert fmps.get_tensor_parallel_rank() == 0
    assert fmps.get_pipeline_parallel_rank() == 0
    assert fmps.get_data_parallel_rank() == dist.get_rank()
    assert fmps.get_tensor_parallel_world_size() == 1
    assert fmps.get_pipeline_parallel_world_size() == 1
    assert fmps.get_data_parallel_world_size() == dist.get_world_size()

    utils.print_success("test_initialize_distributed_state")


@register_initialize_test
def test_initialize_multiple_models():
    utils.init_process_group()

    model_1 = nn.Linear(2, 4)

    fmps_1 = fm_dist.initialize_distributed_state(
        model_1, 1, 1, dist.get_world_size()
    )

    model_2 = nn.Linear(3, 5)

    fmps_2 = fm_dist.initialize_distributed_state(
        model_2, dist.get_world_size(), 1, 1
    )

    assert fmps_1.get_local_rank() == dist.get_rank()
    assert fmps_1.get_local_world_size() == dist.get_world_size()
    assert fmps_1.get_subset_rank() == dist.get_rank()
    assert fmps_1.get_subset_world_size() == dist.get_world_size()
    assert fmps_1.get_tensor_parallel_rank() == 0
    assert fmps_1.get_pipeline_parallel_rank() == 0
    assert fmps_1.get_data_parallel_rank() == dist.get_rank()
    assert fmps_1.get_tensor_parallel_world_size() == 1
    assert fmps_1.get_pipeline_parallel_world_size() == 1
    assert fmps_1.get_data_parallel_world_size() == dist.get_world_size()

    assert fmps_2.get_local_rank() == dist.get_rank()
    assert fmps_2.get_local_world_size() == dist.get_world_size()
    assert fmps_2.get_subset_rank() == dist.get_rank()
    assert fmps_2.get_subset_world_size() == dist.get_world_size()
    assert fmps_2.get_tensor_parallel_rank() == dist.get_rank()
    assert fmps_2.get_pipeline_parallel_rank() == 0
    assert fmps_2.get_data_parallel_rank() == 0
    assert fmps_2.get_tensor_parallel_world_size() == dist.get_world_size()
    assert fmps_2.get_pipeline_parallel_world_size() == 1
    assert fmps_2.get_data_parallel_world_size() == 1

    utils.print_success("test_initialize_multiple_models")
