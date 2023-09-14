import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from flex_model.core import FlexModel, HookFunction
import flex_model.distributed as dist
from flex_model.utils import setup_logger
from tests.test_utilities import Utils, make_model_and_tokenizer


logger = logging.getLogger(__name__)


def test_backward_hooks():
    accelerator = Accelerator()

    model, tokenizer = make_model_and_tokenzier()
    model = model.to(accelerator.device)

    prompts = [
        "It's a nice day we're having",
        "The capital of Canada is",
        "What should I eat for dinner tonight?",
        "There's about three people going to",
    ]

    inputs = tokenizer(
        prompts,
        padding="max_length",
        return_tensors="pt",
    )[
        "input_ids"
    ].to(accelerator.device)

    # Multi-gpu FSDP
    gradients = {}
    model = accelerator.prepare(model)

    flex_model = FlexModel(
        model,
        multi_gpu_activations,
        data_parallel_size=accelerator.num_processes,
    )
    for module_name, expected_shape in LLAMA_MODULES_FSDP.items():
        flex_model.register_hook_function(HookFunction(module_name, expected_shape))

    chunked_inputs = inputs.chunk(accelerator.num_processes, dim=0)

    _ = flex_model(chunked_inputs[accelerator.process_index])

    multi_gpu_activations_ = {
        k.replace("_fsdp_wrapped_module.", ""): v
        for k, v in multi_gpu_activations.items()
    }
