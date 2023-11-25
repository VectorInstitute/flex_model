import functools
import contextlib
import logging
import os

import fairscale.nn.model_parallel as mpu
import torch
import torch.distributed as dist
import torch.nn as nn
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP


import flex_model.distributed as fm_dist

logger = logging.getLogger(__name__)


# Model creation functions.
def llama_7b() -> nn.Module:
    """Helper function to construct a llama-2 model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "/model-weights/Llama-2-7b-hf",
        local_files_only=True,
    )

    return model


def llama_13b() -> nn.Module:
    """Helper function to construct a llama-2 model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "/model-weights/Llama-2-13b-hf",
        local_files_only=True,
    )

    return model


def llama_tokenizer() -> LlamaTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        "/model-weights/Llama-2-13b-hf",
        local_files_only=True,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 128

    return tokenizer


def opt_350m() -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        "/model-weights/opt-350m",
        local_files_only=True,
    )
    return model


def opt_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "/model-weights/opt-350m",
        local_files_only=True,
    )

    return tokenizer


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


def wrap_ddp(base_model, rank, pg=None):
    return DDP(
        base_model,
        # device_ids=[rank],
        process_group=pg,
    )


def wrap_fsdp(base_model, layer_to_wrap, pg=None):
    """Standard FSDP wrap in full-shard mode, CPU RAM efficient."""
    # Initialize fsdp options.
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # Shard model parameters, optimizer, grads over all GPUs.
    sharding_strategy = ShardingStrategy.FULL_SHARD

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_root_forward_inputs=True,
    )

    # Don't offload to CPU.
    cpu_offload = CPUOffload(offload_params=False)

    transformer_auto_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_to_wrap},
    )

    # Wrap model.
    model = FSDP(
        base_model,
        process_group=pg,  # default pg.
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        auto_wrap_policy=transformer_auto_wrapper_policy,
        backward_prefetch=backward_prefetch,
        mixed_precision=mixed_precision,
        ignored_modules=None,
        param_init_fn=None,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        forward_prefetch=True,
        limit_all_gathers=True,
        use_orig_params=False,
    )
    return model


def init_process_group():
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


class Utils:
    @staticmethod
    def initialize_torch_distributed():
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    @staticmethod
    def initialize_flexmodel_distributed(
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
    ):
        fm_dist.initialize_distributed_state(
            tp * pp * dp,
            tp,
            pp,
            dp,
        )

    @staticmethod
    def initialize_mpu_model_parallel(
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
    ):
        if not dist.is_initialized():
            Utils.initialize_torch_distributed()

        mpu.initialize_model_parallel(
            model_parallel_size_=tp,
            pipeline_length=pp,
        )

    @staticmethod
    def destroy_torch_distributed():
        dist.destroy_process_group()

    @staticmethod
    def destroy_flexmodel_distributed():
        fm_dist.destroy_distributed_state()

    @staticmethod
    def destroy_mpu_model_parallel():
        mpu.destroy_model_parallel()

    @staticmethod
    def contract_gpu_context():
        # Non-rank0 devices exit.
        # if dist.get_rank() != 0:
        #    return
        dist.destroy_process_group()

        # Torch distributed with only one device.
        dist.init_process_group(backend="nccl", world_size=1, rank=0)

    @staticmethod
    def expand_gpu_context():
        dist.destroy_process_group()

        # `torchrun` env vars still active, restore the full state w/ all devices.
        dist.init_process_group(backend="nccl")

    @staticmethod
    @contextlib.contextmanager
    def single_gpu_context():
        Utils.contract_gpu_context()
        try:
            yield
        finally:
            Utils.expand_gpu_context()


def gather_weight(param: Tensor, dim: int):
    mp_group = mpu.get_model_parallel_group()

    if mpu.get_model_parallel_world_size() == 1:
        return param

    tensor_list = [
        torch.empty_like(param)
        for _ in range(mpu.get_model_parallel_world_size())
    ]
    tensor_list[mpu.get_model_parallel_rank()] = param

    dist.all_gather(tensor_list, param, mp_group)

    output = torch.cat(tensor_list, dim=dim)

    return output


class FairscaleLayers(nn.Module):
    def __init__(
        self,
        vocab_size,
        sequence_length,
        hidden_dim,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim

        # Vocab parallel and regular embedding
        self.vocab_parallel_embedding = VocabParallelEmbedding(
            self.vocab_size, self.hidden_dim
        ).cuda()

        full_vocab_embedding_weight = gather_weight(
            self.vocab_parallel_embedding.weight.detach(),
            dim=0,
        )

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.vocab_embedding.weight = nn.Parameter(full_vocab_embedding_weight)

        # Parallel embedding and regular embedding
        self.parallel_embedding = ParallelEmbedding(
            self.vocab_size, self.hidden_dim
        ).cuda()

        full_embedding_weight = gather_weight(
            self.parallel_embedding.weight.detach(), dim=1
        )

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.embedding.weight = nn.Parameter(full_embedding_weight)

        # Column parallel linear and regular linear
        self.column_parallel_linear = ColumnParallelLinear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            gather_output=False,
        ).cuda()
        full_col_linear_weight = gather_weight(
            self.column_parallel_linear.weight.detach(), dim=0
        )
        self.col_linear = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.col_linear.weight = nn.Parameter(full_col_linear_weight)

        # Row parallel linear and regular linear
        self.row_parallel_linear = RowParallelLinear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            input_is_parallel=True,
        ).cuda()
        full_row_linear_weight = gather_weight(
            self.row_parallel_linear.weight.detach(), dim=1
        )
        self.row_linear = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.row_linear.weight = nn.Parameter(full_row_linear_weight)

    def parallel_forward(self, inputs):
        embed_0 = self.vocab_parallel_embedding(inputs)
        embed_1 = self.parallel_embedding(inputs)
        embed = embed_0 + embed_1

        col = self.column_parallel_linear(embed)

        row = self.row_parallel_linear(col)

        return row

    def regular_forward(self, inputs):
        embed_0 = self.vocab_embedding(inputs)
        embed_1 = self.embedding(inputs)
        embed = embed_0 + embed_1

        col = self.col_linear(embed)

        row = self.row_linear(col)

        return row

    def forward(self, inputs):
        parallel_out = self.parallel_forward(inputs)
        regular_out = self.regular_forward(inputs)

        return parallel_out, regular_out
