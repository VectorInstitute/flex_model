import logging

import torch
import torch.distributed as dist

from tests.test_utilities import Utils
import flex_model.distributed as fm_dist
from flex_model.distributed.initialize import(
    initialize_activation_parallel,
    get_world_size,
    destroy_activation_parallel,
)


def test_initialize_and_destroy_activation_parallel():
    Utils.initialize_distributed()
    initialize_activation_parallel(list(range(dist.get_world_size())))
    assert fm_dist.is_initialized()
    assert fm_dist.get_activation_parallel_group() is not None
    assert fm_dist.get_world_size() == dist.get_world_size()
    assert fm_dist.get_rank() == dist.get_rank()

    destroy_activation_parallel()
    assert not fm_dist.is_initialized()
    assert fm_dist.get_activation_parallel_group() is None


test_initialize_and_destroy_activation_parallel()
