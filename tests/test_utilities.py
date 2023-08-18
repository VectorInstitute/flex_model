import logging
import os

import fairscale.nn.model_parallel as mpu
from fairscale.nn.model_parallel.layers import (
    VocabParallelEmbedding,
    ParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
)
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

import flex_model.distributed as fm_dist


logger = logging.getLogger(__name__)


class Utils:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    @staticmethod
    def initialize_distributed():
        dist.init_process_group(backend="nccl")
        print(f"Rank{dist.get_rank()}/{dist.get_world_size()}: "
              f"Distributed initialized")

    @staticmethod
    def destroy_distributed():
        dist.destroy_process_group()

    @staticmethod
    def initialize_model_parallel(
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
    ):
        if not dist.is_initialized():
            Utils.initialize_distributed()

        mpu.initialize_model_parallel(
            model_parallel_size_=tp,
        )

    @staticmethod
    def destroy_model_parallel():
        mpu.destroy_model_parallel()
        dist.barrier()

    @staticmethod
    def initialize_distributed_backend(
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
    ):
        if not dist.is_initialized():
            Utils.initialize_distributed()
        fm_dist.initialize_distributed_backend(dist.get_world_size(), tp, pp, dp)

    @staticmethod
    def destroy_distributed_backend():
        fm_dist.destroy_distributed_backend()
        dist.barrier()

    @staticmethod
    def initialize_activation_parallel():
        fm_dist.initialize_activation_parallel()

    @staticmethod
    def destroy_activation_parallel():
        fm_dist.destroy_activation_parallel()
        dist.barrier()


def gather_weight(param: Tensor, dim: int):
    mp_group = mpu.get_model_parallel_group()

    if dist.get_world_size() == 1:
        return param

    tensor_list = [torch.empty_like(param) for _ in range(mpu.get_model_parallel_world_size())]
    tensor_list[mpu.get_model_parallel_rank()] = param

    dist.all_gather(tensor_list, param, mp_group)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class MegatronLayers(nn.Module):
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
            self.vocab_parallel_embedding.weight.detach(), dim=0,
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
        self.col_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
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
        self.row_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
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
