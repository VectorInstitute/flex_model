import logging

import torch
import torch.distributed as dist

from tests.test_utilities import Utils
import flex_model.distributed as fm_dist
from flex_model.distributed.parse import (
    _get_different_dim,
    _autofill_expected_shape,
    parse_collect_and_distribute_from_tensor,
)


def test__get_different_dim():
    dim1, dim2 = (1, 2, 3), (1, 4, 3)
    result = _get_different_dim(dim1, dim2)
    assert result == 1
    dim1, dim2 = (5,), (6,)
    result = _get_different_dim(dim1, dim2)
    assert result == 0


def test__autofill_expected_shape():
    tensor = torch.randn(1, 2, 3)
    expected_shape = (None, None, None)
    result = _autofill_expected_shape(tensor, expected_shape)
    assert result == (1, 2, 3)


def test_parse_collect_and_distribute_from_tensor():
    tensor = torch.ones((1)).cuda()
    expected_shape = (1,)
    collect, distribute = parse_collect_and_distribute_from_tensor(
        tensor, expected_shape
    )
    assert collect == distribute == fm_dist.unity

    Utils.initialize_activation_parallel()

    # Sharded dim case, diff data
    tensor = torch.ones((1)).cuda() * fm_dist.get_rank()
    expected_shape = (fm_dist.get_world_size(),)
    collect, distribute = parse_collect_and_distribute_from_tensor(
        tensor, expected_shape
    )
    assert collect.func == fm_dist.all_gather_sync
    assert distribute.func == fm_dist.scatter_rank0_sync

    # Sharded dim case, same data
    tensor = torch.ones((1)).cuda() * fm_dist.get_rank()
    expected_shape = (fm_dist.get_world_size(),)
    collect, distribute = parse_collect_and_distribute_from_tensor(
        tensor, expected_shape
    )
    assert collect.func == fm_dist.all_gather_sync
    assert distribute.func == fm_dist.scatter_rank0_sync

    # Replicated case
    tensor = torch.ones((1)).cuda()
    expected_shape = (1,)
    collect, distribute = parse_collect_and_distribute_from_tensor(
        tensor, expected_shape
    )
    assert collect == fm_dist.unity
    assert distribute == fm_dist.broadcast_rank0_sync


test__get_different_dim()
test__autofill_expected_shape()
test_parse_collect_and_distribute_from_tensor()
