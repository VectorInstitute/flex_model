import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


@pytest.fixture
def llama_13b() -> nn.Module:
    """Helper function to construct a llama-2 model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "/model-weights/Llama-2-13b-hf",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    return model


@pytest.fixture
def llama_tokenizer() -> LlamaTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        "/model-weights/Llama-2-13b-hf",
        local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 128

    return tokenizer


@pytest.fixture
def make_opt_350m() -> nn.Module:
    def _make_opt_350m():
        model = AutoModelForCausalLM.from_pretrained(
            "/model-weights/opt-350m",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        return model

    yield _make_opt_350m


@pytest.fixture(scope="session")
def opt_350m_module_names():
    model = AutoModelForCausalLM.from_pretrained(
        "/model-weights/opt-350m",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    return [n for n, _ in model.named_modules()]


@pytest.fixture
def opt_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "/model-weights/opt-350m",
        local_files_only=True,
    )

    return tokenizer


# TODO: Parameterize models as a sweep.
@pytest.fixture
def model():
    raise NotImplementedError
