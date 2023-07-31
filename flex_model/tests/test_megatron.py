import logging
from typing import Dict

import torch
import torch.distributed as dist
from torch import Tensor

from flex_model.model_wrappers import FlexModel
from flex_model._distributed_utils import print_rank0
from flex_model.tests.testing_utils import (
    get_llama_13b_megatron,
    apply_flex_model_fwd_hooks,
    compare_tensor_dicts,
    print_named_modules,
    module_comparison_mapping,
    print_return_dict,
)
from flex_model.tests.testing_constants import (
    _PROMPTS,
    _LLAMA_VANILLA_TORCH_MODULES,
    _LLAMA_MEGATRON_MODULES,
)


logger = logging.getLogger(__name__)


def _llama_megatron_run() -> Dict[str, Tensor]:
    model, tokenize_fn = get_llama_13b_megatron()

    inputs = tokenize_fn(_PROMPTS).cuda()
    logger.info(f"Rank{torch.distributed.get_rank()} inputs: {inputs}")

    output_dict, outputs = apply_flex_model_fwd_hooks(
        model=model,
        inputs=inputs,
        module_names_with_shapes=_LLAMA_MEGATRON_MODULES,
        start_pos=0,
    )
    return output_dict, outputs


def test_distributed_flex_model_megatron():
    from flex_model.tests.test_distributed import _llama_vanilla_torch_run
    logger.info("Testing megatron distributed flex model")
    megatron_output, megatron_out = _llama_megatron_run()

    if dist.get_rank() == 0:
        vanilla_torch_output, vanilla_torch_out = _llama_vanilla_torch_run()

        mapping=module_comparison_mapping(
            _LLAMA_VANILLA_TORCH_MODULES,
            _LLAMA_MEGATRON_MODULES,
        )
        compare_tensor_dicts(
            vanilla_torch_output,
            megatron_output,
            mapping=mapping,
        )
        logger.info("Test complete")
    else:
        return


def test_weight_retrieval():
    logger.info("Testing sharded unembedding weight retrieval")
    model, _ = get_llama_13b_megatron()

    _output_dict = {}
    model = FlexModel(model, _output_dict)

    unembed_weight = model.get_module_parameter("output.weight", (32000, 5120))

    rank_chunk = torch.chunk(unembed_weight, 2, dim=0)[dist.get_rank()]

    assert torch.equal(rank_chunk, model.module.output.weight.detach().cpu())

    logger.info("Test complete")


def main():
    # TODO: Can't run both at once with *different* models
    test_distributed_flex_model_megatron()
    #test_weight_retrieval()


if __name__ == "__main__":
    main()
