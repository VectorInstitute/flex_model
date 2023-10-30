import argparse
import contextlib
import functools
import os
import random
import time
from collections import defaultdict
from itertools import chain

import megatron.core.parallel_state as mpu
import numpy as np
import torch
import torch.nn as nn
import wandb
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    model_parallel_cuda_manual_seed,
)
from torch.profiler import profile

from flex_model.core import FlexModel, HookFunction


class TestNetwork(nn.Module):
    def __init__(self, model_dim, n_layers, is_distributed, config):
        super().__init__()
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.is_distributed = is_distributed
        self.config = config

        assert self.n_layers % 2 == 0

        if self.is_distributed:
            # Alternating column parallel - row parallel layers.
            init_method = nn.init.xavier_normal_
            layers = list(
                chain.from_iterable(
                    (
                        ColumnParallelLinear(
                            model_dim, model_dim, config=config, init_method=init_method
                        ),
                        RowParallelLinear(
                            model_dim,
                            model_dim,
                            input_is_parallel=True,
                            config=config,
                            init_method=init_method,
                        ),
                    )
                    for _ in range(self.n_layers // 2)
                )
            )
        else:
            layers = [
                nn.Linear(model_dim, model_dim, dtype=self.config.params_dtype)
                for _ in range(self.n_layers)
            ]
        self.layers = nn.ModuleList(layers)

        self.act_fn = nn.ReLU()

    def forward(self, inputs):
        rep = inputs
        for i in range(len(self.layers)):
            rep = self.layers[i](rep)

            # Column and row parallel layers return tuple(output, output_bias).
            if isinstance(rep, tuple):
                rep = rep[0]

            if i % 2 == 0:
                rep = self.act_fn(rep)

        return rep


def hook_function_identity(self, inputs, outputs, name):
    """Hook function that does nothing."""
    return outputs


def hook_function_cpu(self, inputs, outputs, acc, name):
    """Hook function that dumps to cpu."""
    rank = mpu.get_tensor_model_parallel_rank()

    # NOTE: See note about output of col and row parallel.
    _outputs = outputs[0] if isinstance(outputs, tuple) else outputs

    if rank == 0:
        acc[name] = _outputs.detach().cpu()
    return outputs


def hook_function_gpu(self, inputs, outputs, acc, name):
    """Hook function that dumps to GPU."""
    rank = mpu.get_tensor_model_parallel_rank()

    _outputs = outputs[0] if isinstance(outputs, tuple) else outputs

    if rank == 0:
        acc[name] = _outputs.detach()


def hook_function_cpu_gather_scatter(self, inputs, outputs, acc, name):
    """Hardcoded minimal hook function with gather/scatter.
    """
    rank = mpu.get_tensor_model_parallel_rank()

    _outputs = outputs[0] if isinstance(outputs, tuple) else outputs
    original_shape = _outputs.shape

    def all_gather(tensor, dim=-1):
        output_list = [
            torch.empty_like(tensor)
            for _ in range(mpu.get_tensor_model_parallel_world_size())
        ]
        torch.distributed.all_gather(
            output_list, tensor, group=mpu.get_tensor_model_parallel_group(),
        )
        return torch.cat(output_list, dim=dim)

    def scatter(tensor, dim=-1):
        scatter_list = list(
            torch.chunk(tensor, mpu.get_tensor_model_parallel_world_size(), dim=dim)
        )
        return scatter_list[rank]

    # Determine correct gather/scatter functions.
    if isinstance(self, ColumnParallelLinear):
        gather_fn = all_gather
        scatter_fn = scatter

    elif isinstance(self, RowParallelLinear):
        gather_fn = lambda x: x
        scatter_fn = lambda x: x

    else:
        gather_fn = lambda x: x
        scatter_fn = lambda x: x

    _outputs = gather_fn(_outputs)

    if rank == 0:
        acc[name] = _outputs.detach().cpu()

    _outputs = scatter_fn(_outputs)

    outputs = (_outputs, *outputs[1:])

    return outputs


class ExperimentNetworkManager:
    """Contains functions for creating the experiment networks.

    Also can cache networks to reduce initialization latency.
    """

    def __init__(self):
        self.cpu_acc = {}
        self.named_experiments = [
            "single_gpu_no_hooks",
            "single_gpu_unity_hooks",
            "single_gpu_cpu_hooks",
            "multi_gpu_no_hooks",
            "multi_gpu_unity_hooks",
            "multi_gpu_cpu_hooks",
            "multi_gpu_gpu_hooks",
            "multi_gpu_cpu_hooks_with_gather_scatter",
            "multi_gpu_flex_model",
        ]
        self.hook_handles = defaultdict(list)
        self.network_cache = None

    def cleanup(self):
        self.cpu_acc.clear()

        # NOTE: When the FlexModel goes out of scope, its pointers to the hook
        #       function handles trigger RemovableHandle.__exit__() which
        #       automatically calls `.remove()`. Hence we don't need to remove
        #       them manually here for the wrapped network.
        self.remove_hooks(self.network_cache[-1])

        for m in self.network_cache[-1].modules():
            assert len(m._forward_hooks) == 0

    def _hook_every_layer(self, network, hook_fn):
        module_names_to_hook = set(f"layers.{i}" for i in range(len(network.layers)))
        for n, m in network.named_modules():
            if n in module_names_to_hook:
                hook_fn = functools.partial(hook_fn, name=n)
                handle = m.register_forward_hook(hook_fn)
                self.hook_handles[network].append(handle)
                module_names_to_hook.remove(n)

        assert (
            len(module_names_to_hook) == 0
        ), f"Have left over modules to hook: {module_names_to_hook}"

    def remove_hooks(self, network):
        handles = self.hook_handles.get(network, [])
        for handle in handles:
            handle.remove()

    def check_network_cache(self, *args, **kwargs):
        if self.network_cache is None:
            return False

        cached_args, cached_kwargs = self.network_cache[:-2], self.network_cache[-2]
        for arg, c_arg in zip(args, cached_args):
            if arg != c_arg:
                return False

        if cached_kwargs != kwargs:
            return False

        return True

    def make_network(self, *args, **kwargs):
        if self.check_network_cache(*args, **kwargs):
            network = self.network_cache[-1]
        else:
            network = TestNetwork(*args, **kwargs)
            self.network_cache = [*args, kwargs, network]
        return network

    def single_gpu_no_hooks(self, model_dim, n_layers, config):
        network = self.make_network(
            model_dim, n_layers, is_distributed=False, config=config
        )
        return network

    def single_gpu_unity_hooks(self, model_dim, n_layers, config):
        network = self.make_network(
            model_dim, n_layers, is_distributed=False, config=config
        )

        hook_fn = hook_function_identity

        self._hook_every_layer(network, hook_fn)

        return network

    def single_gpu_cpu_hooks(self, model_dim, n_layers, config):
        network = self.make_network(
            model_dim, n_layers, is_distributed=False, config=config
        )

        hook_fn = functools.partial(hook_function_cpu, acc=self.cpu_acc)

        self._hook_every_layer(network, hook_fn)

        return network

    def multi_gpu_no_hooks(self, model_dim, n_layers, config):
        network = self.make_network(
            model_dim, n_layers, is_distributed=True, config=config
        )
        return network

    def multi_gpu_unity_hooks(self, model_dim, n_layers, config):
        network = self.make_network(
            model_dim, n_layers, is_distributed=True, config=config
        )

        hook_fn = hook_function_identity

        self._hook_every_layer(network, hook_fn)

        return network

    def multi_gpu_cpu_hooks(self, model_dim, n_layers, config):
        network = self.make_network(
            model_dim, n_layers, is_distributed=True, config=config
        )

        # NOTE: This will not accumulate full tensors since we don't gather.
        hook_fn = functools.partial(hook_function_cpu, acc=self.cpu_acc)

        self._hook_every_layer(network, hook_fn)

        return network

    def multi_gpu_gpu_hooks(self, model_dim, n_layers, config):
        network = self.make_network(
            model_dim, n_layers, is_distributed=True, config=config
        )

        # NOTE: This will not accumulate full tensors since we don't gather.
        hook_fn = functools.partial(hook_function_gpu, acc=self.cpu_acc)

        self._hook_every_layer(network, hook_fn)

        return network

    def multi_gpu_cpu_hooks_with_gather_scatter(self, model_dim, n_layers, config):
        network = self.make_network(
            model_dim, n_layers, is_distributed=True, config=config
        )
        hook_fn = functools.partial(hook_function_cpu_gather_scatter, acc=self.cpu_acc)

        self._hook_every_layer(network, hook_fn)

        return network

    def multi_gpu_flex_model(self, model_dim, n_layers, config):
        base_network = self.make_network(
            model_dim, n_layers, is_distributed=True, config=config
        )

        network = FlexModel(
            base_network,
            self.cpu_acc,
            tensor_parallel_size=mpu.get_tensor_model_parallel_world_size(),
            pipeline_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )

        module_names_to_hook = set(
            f"layers.{i}" for i in range(len(base_network.layers))
        )
        for n in module_names_to_hook:
            network.register_hook_function(
                HookFunction(module_name=n, expected_shape=(None, model_dim),)
            )
        return network


def init_megatron_dist(args):
    """Initialize Megatron-LM parallel state."""
    os.environ["NCCL_IB_DISABLE"] = "1"
    initialize_distributed()
    initialize_model_parallel(args.tp_size)

    # Taken from: https://github.com/NVIDIA/Megatron-LM/blob/feac76a79148622d8f2a45d46c08a972a24784a3/megatron/initialize.py#L236
    seed = 0
    seed += 100 * mpu.get_pipeline_model_parallel_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        model_parallel_cuda_manual_seed(0)


def initialize_distributed():
    """Initialize torch distributed state."""
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)

    torch.distributed.init_process_group(backend="nccl")


def spoof_megatron_config(dtype):
    """Spoof the megatron config to initialize megatron core layers."""
    config = argparse.Namespace()
    config.perform_initialization = True
    config.params_dtype = dtype
    config.async_tensor_model_parallel_allreduce = False
    config.sequence_parallel = False
    config.gradient_accumulation_fusion = False
    config.expert_model_parallel_size = 1
    config.use_cpu_initialization = False

    return config
