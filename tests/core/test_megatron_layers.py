import logging

import torch
import torch.distributed as dist

from flex_model.core import FlexModel, HookFunction
from flex_model.utils import setup_logger
from tests.test_utilities import Utils, MegatronLayers
from tests.registry import register_test


logger = logging.getLogger(__name__)


@register_test
def test_MegatronLayers():
    Utils.initialize_model_parallel()

    torch.manual_seed(42069)

    vocab_size = 512
    sequence_length = 128
    hidden_dim = 256
    batch_size = 4

    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
    ).cuda()
    logger.debug(inputs)

    model = MegatronLayers(vocab_size, sequence_length, hidden_dim)

    parallel_out, regular_out = model(inputs)
    assert torch.allclose(parallel_out, regular_out, atol=1e-7)
    Utils.destroy_model_parallel()


@register_test
def test_FlexModelMegatron():
    Utils.initialize_model_parallel(2, 1, 2)

    torch.manual_seed(42069)

    vocab_size = 512
    sequence_length = 128
    hidden_dim = 256
    batch_size = 4

    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
    ).cuda()
    logger.debug(inputs)

    model = MegatronLayers(vocab_size, sequence_length, hidden_dim)

    output_dict = {}
    model = FlexModel(
        model,
        output_dict,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=2,
    )
    print(model)
    hook_functions = {
        "vocab_parallel_embedding": (None, None, hidden_dim),
        "vocab_embedding": (None, None, None),
        "parallel_embedding": (None, None, hidden_dim),
        "embedding": (None, None, None),
        "column_parallel_linear": (None, None, hidden_dim),
        "col_linear": (None, None, None),
        "row_parallel_linear": (None, None, None),
        "row_linear": (None, None, None),
    }
    for module_name, expected_shape in hook_functions.items():
        model.register_hook_function(HookFunction(module_name, expected_shape))

    _, _ = model(inputs)

    if dist.get_rank() == 0:
        assert torch.allclose(
            output_dict["vocab_parallel_embedding"],
            output_dict["vocab_embedding"],
            atol=1e-7,
        )
        assert torch.allclose(
            output_dict["parallel_embedding"], output_dict["embedding"], atol=1e-7
        )
        assert torch.allclose(
            output_dict["column_parallel_linear"], output_dict["col_linear"], atol=1e-7
        )
        assert torch.allclose(
            output_dict["row_parallel_linear"], output_dict["row_linear"], atol=1e-7
        )
        logger.info("Tests successful.")
        
    Utils.destroy_activation_parallel()
    Utils.destroy_distributed_backend()
    Utils.destroy_model_parallel()


test_MegatronLayers()
test_FlexModelMegatron()
