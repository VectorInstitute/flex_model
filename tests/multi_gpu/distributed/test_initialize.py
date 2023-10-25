import pytest

import flex_model.distributed as fm_dist
from tests.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry
from tests.multi_gpu.testing_utils import Utils

register_initialize_test, get_initialize_test = make_test_registry(
    "initialize", SlurmJobResourceSpec(),
)


@register_initialize_test
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
