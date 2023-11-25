import torch.nn as nn
import torch.distributed as dist

from flex_model.utils import setup_logger
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
def test_initialize_and_destroy_distributed_state():
    setup_logger("debug")
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

    del fmps

    assert not _global_state_is_initialized()

    dist.destroy_process_group()


# TODO: Add tests for multiple models.
test_initialize_and_destroy_distributed_state()
