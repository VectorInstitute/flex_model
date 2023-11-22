import flex_model.distributed as fm_dist
from _test.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry
from _test.multi_gpu.testing_utils import Utils

register_initialize_test, get_initialize_test = make_test_registry(
    "initialize",
    SlurmJobResourceSpec(),
)


@register_initialize_test
def test_initialize_and_destroy_distributed_state():
    Utils.initialize_flexmodel_distributed(2, 1, 2)
    assert fm_dist.distributed_state_is_initialized()

    Utils.destroy_flexmodel_distributed()
    assert not fm_dist.distributed_state_is_initialized()
