import logging
from typing import Dict

import torch
import torch.distributed as dist
from torch import Tensor

from flex_model.tests.testing_utils import (
    get_llama_13b_megatron,
    apply_flex_model_fwd_hooks,
    compare_tensor_dicts,
)
from flex_model.tests.testing_constants import (
    _PROMPTS,
    _LLAMA_FSDP_MODULE_NAMES,
    _LLAMA_FSDP_MODULE_SHAPES,
)


logger = logging.getLogger(__name__)


def _llama_megatron_run() -> Dict[str, Tensor]:
    model, tokenize_fn = get_llama_13b_megatron()

    inputs = tokenize_fn(_PROMPTS).cuda()
    logger.info(f"Rank{torch.distributed.get_rank()} inputs: {inputs}")

    output_dict = apply_flex_model_fwd_hooks(
        model=model,
        inputs=inputs,
        module_names=_LLAMA_FSDP_MODULE_NAMES,
        shapes=_LLAMA_FSDP_MODULE_SHAPES,
        start_pos=0,
    )
    return output_dict


def test_distributed_flex_model_megatron():
    from flex_model.tests.test_distributed import _llama_vanilla_torch_run
    vanilla_torch_output = _llama_vanilla_torch_run()

    megatron_output = _llama_megatron_run()

    if dist.get_rank() == 0:
        assert compare_tensor_dicts(vanilla_torch_output, megatron_output)
        logger.info("Test successful")
    else:
        return


def main():
    test_distributed_flex_model_megatron()


if __name__ == "__main__":
    main()
