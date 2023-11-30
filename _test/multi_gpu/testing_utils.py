import functools
import os
from typing import Dict, Tuple

import fairscale.nn.model_parallel as mpu
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from flex_model.core import HookFunction


def print_success(test_name: str):
    rank = dist.get_rank()
    print(f"Rank{rank}: [{test_name}] - Test successful")


def init_process_group():
    torch.manual_seed(0)
    if dist.is_initialized():
        return

    dist.init_process_group("nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not local_rank == torch.cuda.current_device():
        torch.cuda.set_device(local_rank)


def init_fairscale_mpu(tp_size, pp_size):
    if not dist.is_initialized():
        init_process_group()

    mpu.initialize_model_parallel(
        model_parallel_size_=tp_size,
        pipeline_length=pp_size,
    )


def destroy_fairscale_mpu():
    mpu.destroy_model_parallel()
    dist.barrier()


def destroy_process_group():
    dist.destroy_process_group()


def all_gather(tensor: Tensor, dim: int = 0, pg=None):
    world_size = dist.get_world_size(group=pg)

    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]

    dist.all_gather(tensor_list, tensor, group=pg)

    return torch.cat(tensor_list, dim=dim)


class TestModel(nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.fc1 = nn.Linear(10, 20, device=device, dtype=dtype)
        self.fc2 = nn.Linear(20, 10, device=device, dtype=dtype)

    def forward(self, inputs):
        return self.fc2(self.fc1(inputs))


class TestFairscaleModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        expansion=2,
        device="cuda",
        dtype=torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.expansion = expansion
        self.device = device
        self.dtyp = dtype

        # Vocab parallel and regular embedding
        self.vocab_parallel_embedding = (
            VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
            )
            .to(device)
            .to(dtype)
        )

        # Parallel embedding and regular embedding
        self.parallel_embedding = (
            ParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
            )
            .to(device)
            .to(dtype)
        )

        # Column parallel linear and regular linear
        self.column_parallel_linear = (
            ColumnParallelLinear(
                self.hidden_size,
                int(self.hidden_size * self.expansion),
                bias=False,
                gather_output=False,
            )
            .to(device)
            .to(dtype)
        )

        # Row parallel linear and regular linear
        self.row_parallel_linear = (
            RowParallelLinear(
                int(self.hidden_size * self.expansion),
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
            )
            .to(device)
            .to(dtype)
        )

    def get_unsharded_params_and_grads(
        self
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Get full parameter and gradient state tensors for each module weight."""
        tp_group = mpu.get_model_parallel_group()
        tp_world_size = mpu.get_model_parallel_world_size()

        layer_to_shard_dim_map = {
            VocabParallelEmbedding: 0,
            ColumnParallelLinear: 0,
            ParallelEmbedding: 1,  # Doesn't support negative indexing.
            RowParallelLinear: 1,
        }

        params = {}
        grads = {}
        for name, module in self.named_modules():
            if isinstance(module, TestFairscaleModel):
                continue

            sharded_dim = layer_to_shard_dim_map[type(module)]

            if tp_world_size > 1:
                unsharded_param = all_gather(
                    module.weight,
                    dim=sharded_dim,
                    pg=tp_group,
                )
                unsharded_grad = None
                if module.weight.grad is not None:
                    unsharded_grad = all_gather(
                        module.weight.grad,
                        dim=sharded_dim,
                        pg=tp_group,
                    )
            else:
                unsharded_param = module.weight
                unsharded_grad = module.weight.grad

            params[f"{name}.weight"] = unsharded_param
            grads[f"{name}.weight.grad"] = unsharded_grad

        return params, grads

    def copy_state_from_unsharded(
        self, other: nn.Module, other_tp_world_size: int = 1
    ):
        """Copy the parameter states from an unsharded version of this model."""
        tp_rank = mpu.get_model_parallel_rank()
        tp_world_size = mpu.get_model_parallel_world_size()

        # Guarantee local slice of param exists.
        assert other_tp_world_size < tp_world_size
        assert tp_world_size % other_tp_world_size == 0

        # Helpers
        def _resharded_param_dim(param, dim=-1):
            full_dim = param.shape[dim] * other_tp_world_size
            return full_dim // tp_world_size

        def _make_param_slice(start, end, shape, dim):
            slices = []
            for i in range(len(shape)):
                if i == dim:
                    slices.append(slice(start, end))
                else:
                    slices.append(slice(0, shape[i]))
            return tuple(slices)

        # Reshard each parameter.
        for (name, module), (other_name, other_module) in zip(
            self.named_modules(), other.named_modules()
        ):
            if isinstance(module, TestFairscaleModel):
                continue
            assert name == other_name
            assert type(module) == type(other_module)

            param = module.weight
            other_param = other_module.weight

            dim = (
                0
                if isinstance(
                    module, (VocabParallelEmbedding, ColumnParallelLinear)
                )
                else 1
            )
            resharded_param_dim = _resharded_param_dim(other_param, dim=dim)
            start = (
                tp_rank % other_tp_world_size + tp_rank
            ) * resharded_param_dim
            end = start + resharded_param_dim
            param_slice = _make_param_slice(start, end, param.shape, dim=dim)

            with torch.no_grad():
                param.copy_(other_param[param_slice])
                assert param.is_contiguous()

    def forward(self, inputs):
        embed_1 = self.vocab_parallel_embedding(inputs)
        embed_2 = self.parallel_embedding(inputs)
        embed = embed_1 + embed_2

        out_1 = self.column_parallel_linear(embed)
        out_2 = self.row_parallel_linear(out_1)

        return out_2


def assert_same_state(
    self_states: Dict[str, Tensor], other_states: Dict[str, Tensor]
) -> None:
    """Check if self and other have the same parameter and gradient states."""
    self_params, self_grads = self_states
    other_params, other_grads = other_states

    for (self_name, self_param), (other_name, other_param) in zip(
        self_params.items(), other_params.items()
    ):
        assert self_name == other_name
        torch.testing.assert_close(self_param, other_param)

    for (self_name, self_grad), (other_name, other_grad) in zip(
        self_grads.items(), other_grads.items()
    ):
        assert self_name == other_name
        torch.testing.assert_close(self_grad, other_grad)


def wrap_ddp(base_model, pg=None):
    return DDP(
        base_model,
        process_group=pg,
    )


def wrap_fsdp(base_model, layer_to_wrap, pg=None):
    """Standard FSDP wrap in full-shard mode, CPU RAM efficient."""
    # Initialize fsdp options.
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE

    # Shard model parameters, optimizer, grads over all GPUs.
    sharding_strategy = ShardingStrategy.FULL_SHARD

    # Test everying in fp32 default.
    mixed_precision = MixedPrecision(
        param_dtype=None,
        reduce_dtype=None,
        buffer_dtype=None,
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


def register_hook_functions(
    model, editing_function, hook_type, module_name_to_shape_map, module_prefix
):
    for name, expected_shape in module_name_to_shape_map.items():
        register_fn = getattr(model, hook_type, None)
        assert register_fn is not None, "Reg. fn {hook_type} couldn't be used."

        full_name = module_prefix + name
        register_fn(
            HookFunction(
                full_name,
                expected_shape=expected_shape,
                editing_function=editing_function,
            )
        )
