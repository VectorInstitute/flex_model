import copy
from functools import reduce
import logging

import torch
import torch.nn as nn
import torch.distributed as dist

from flex_model.utils import setup_logger
from flex_model.core import FlexModel, HookFunction
import _test.multi_gpu.testing_utils as utils
from _test.multi_gpu.testing_utils import Utils
from _test.multi_gpu.registry import SlurmJobResourceSpec, make_test_registry


logger = logging.getLogger(__name__)


register_hook_function_test, get_hook_function_test = make_test_registry(
    "hook_function",
    SlurmJobResourceSpec(),
)


MODEL_NAME = "test_model"
MODULE_NAME = "fc1"


# For getting submodule by name.
def rgetattr(module, attr):
    def _getattr(module, attr):
        return getattr(module, attr)

    return reduce(_getattr, [module] + attr.split("."))


def test_forward_hooks():
    Utils.initialize_torch_distributed()

    world_size = dist.get_world_size()

    fsdp_model = utils.test_model().cuda()
    regular_model = copy.deepcopy(fsdp_model).cuda()

    fsdp_model = utils.wrap_fsdp(fsdp_model, nn.Linear)

    inputs = torch.randn(4, 10).cuda()

    acc = {}

    def _edit(module, outputs, save_ctx, trainable_modules):
        return outputs * 2

    # Regular model.
    regular_flex_model = FlexModel(regular_model, acc)
    regular_flex_model.register_forward_hook(
        HookFunction(MODULE_NAME, editing_function=_edit)
    )
    regular_out = regular_flex_model(inputs)

    dist.barrier()

    del regular_flex_model

    # FSDP model.
    fsdp_flex_model = FlexModel(fsdp_model, acc, data_parallel_size=world_size)
    fsdp_flex_model.register_forward_hook(
        HookFunction(MODULE_NAME, editing_function=_edit)
    )
    fsdp_out = fsdp_flex_model(inputs)

    torch.testing.assert_close(regular_out, fsdp_out)
    torch.testing.assert_close(acc[MODULE_NAME][0], acc[MODULE_NAME][1])

    logger.info("Tests successful.")


setup_logger("debug")
test_forward_hooks()
