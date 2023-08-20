import logging

import torch
import torch.distributed as dist

from tests.test_utilities import Utils
import flex_model.distributed as fm_dist
from flex_model.distributed.mappings import (
    broadcast_rank0_sync,
    all_gather_sync,
    all_reduce_sync,
    scatter_rank0_sync,
)


def test_broadcast_rank0_sync():
    Utils.initialize_activation_parallel()

    if fm_dist.get_rank() == 0:
        tensor_to_bcast = torch.ones((1)).cuda()
    else:
        tensor_to_bcast = torch.zeros((1,)).cuda()
    result = broadcast_rank0_sync(tensor_to_bcast)
    assert torch.equal(result, torch.ones((1)).cuda())

    Utils.destroy_activation_parallel()


def test_all_gather_sync():
    Utils.initialize_activation_parallel()

    tensor_to_gather = torch.ones((1)).cuda() * fm_dist.get_rank()
    result = all_gather_sync(tensor_to_gather)
    assert torch.equal(result, torch.arange(fm_dist.get_world_size()).cuda())

    Utils.destroy_activation_parallel()


def test_all_reduce_sync():
    Utils.initialize_activation_parallel()

    tensor_to_reduce = torch.ones((1)).cuda()
    result = all_reduce_sync(tensor_to_reduce)
    assert torch.equal(result, torch.ones((1)).cuda() * fm_dist.get_world_size())

    Utils.destroy_activation_parallel()


def test_scatter_rank0_sync():
    Utils.initialize_activation_parallel()

    tensor_to_scatter = torch.arange(fm_dist.get_world_size()).cuda()
    result = scatter_rank0_sync(tensor_to_scatter)
    assert torch.equal(result, torch.ones((1)).cuda() * fm_dist.get_rank())

    Utils.destroy_activation_parallel()


test_broadcast_rank0_sync()
test_all_gather_sync()
test_all_reduce_sync()
test_scatter_rank0_sync()
