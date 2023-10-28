import argparse
import contextlib
import functools
import os
import time
from itertools import chain

import megatron.core.parallel_state as mpu
import torch
import torch.nn as nn
import wandb
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from torch.profiler import ProfilerActivity, profile, record_function

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
            layers = [nn.Linear(model_dim, model_dim) for _ in range(self.n_layers)]
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


class Benchmark:
    def __init__(self, network, inputs, warmup_iters):
        self.network = network
        self.inputs = inputs
        self.warmup_iters = warmup_iters

    def run_exp(self, num_steps):
        # Warmup iterations.
        N, B, M = self.inputs.shape
        warmup_data = torch.randn(
            self.warmup_iters, B, M, dtype=self.inputs.dtype
        ).cuda()

        with torch.no_grad():
            for i in range(self.warmup_iters):
                _ = self.network(warmup_data[i])

        # Benchmark run.
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            for i in range(num_steps):
                _ = self.network(self.inputs[i])

        torch.cuda.synchronize()
        end = time.time()

        wallclock = end - start

        iter_time = wallclock / num_steps

        return iter_time

    def run_profile(self):
        if len(self.inputs) > 1:
            inputs = self.inputs[0]
        else:
            inputs = self.inputs

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA,],
            record_shapes=True,
        ) as prof:
            with record_function("forward"):
                self.network(inputs)

        return prof


class WandbLogger:
    def __init__(self):
        wandb.init(project="flex_model_exps", config={})
        # Tuple (metric name, is step metric).
        self.metrics = set()

    def define_metric(self, metric_name, step_metric_name=None):
        wandb.define_metric(metric_name, step_metric=step_metric_name)

    def log(self, log_items):
        wandb.log(log_items)


def hook_function_identity(self, inputs, outputs, name):
    return outputs


def hook_function_cpu(self, inputs, outputs, acc, name):
    # NOTE: See note about output of col and row parallel.
    _outputs = outputs[0] if isinstance(outputs, tuple) else outputs

    cpu_ten = _outputs.detach().cpu()
    acc[name] = cpu_ten
    return outputs


class ExperimentNetworkFactory:
    def __init__(self):
        self.cpu_acc = {}
        self.named_experiments = [
            "single_gpu_no_hooks",
            "single_gpu_unity_hooks",
            "single_gpu_cpu_hooks",
            "multi_gpu_no_hooks",
            "multi_gpu_unity_hooks",
            "multi_gpu_cpu_hooks",
            "multi_gpu_flex_model",
        ]

    def clear(self):
        self.cpu_acc.clear()

    def _hook_every_layer(self, network, hook_fn):
        module_names_to_hook = set(f"layers.{i}" for i in range(len(network.layers)))
        for n, m in network.named_modules():
            if n in module_names_to_hook:
                hook_fn = functools.partial(hook_fn, name=n)
                m.register_forward_hook(hook_fn)
                module_names_to_hook.remove(n)

        assert (
            len(module_names_to_hook) == 0
        ), f"Have left over modules to hook: {module_names_to_hook}"

    def single_gpu_no_hooks(self, model_dim, n_layers, config):
        network = TestNetwork(model_dim, n_layers, is_distributed=False, config=config)
        return network

    def single_gpu_unity_hooks(self, model_dim, n_layers, config):
        network = TestNetwork(model_dim, n_layers, is_distributed=False, config=config)

        hook_fn = hook_function_identity

        self._hook_every_layer(network, hook_fn)

        return network

    def single_gpu_cpu_hooks(self, model_dim, n_layers, config):
        network = TestNetwork(model_dim, n_layers, is_distributed=False, config=config)

        hook_fn = functools.partial(hook_function_cpu, acc=self.cpu_acc)

        self._hook_every_layer(network, hook_fn)

        return network

    def multi_gpu_no_hooks(self, model_dim, n_layers, config):
        network = TestNetwork(model_dim, n_layers, is_distributed=True, config=config)
        return network

    def multi_gpu_unity_hooks(self, model_dim, n_layers, config):
        network = TestNetwork(model_dim, n_layers, is_distributed=True, config=config)

        hook_fn = hook_function_identity

        self._hook_every_layer(network, hook_fn)

        return network

    def multi_gpu_cpu_hooks(self, model_dim, n_layers, config):
        network = TestNetwork(model_dim, n_layers, is_distributed=True, config=config)

        # NOTE: This will not accumulate full tensors since we don't gather.
        hook_fn = functools.partial(hook_function_cpu, acc=self.cpu_acc)

        self._hook_every_layer(network, hook_fn)

        return network

    def multi_gpu_flex_model(self, model_dim, n_layers, config):
        base_network = TestNetwork(
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


def initialize_distributed():
    device = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(device)

    torch.distributed.init_process_group(backend="nccl")


def spoof_megatron_config(dtype):
    config = argparse.Namespace()
    config.perform_initialization = True
    config.params_dtype = dtype
    config.async_tensor_model_parallel_allreduce = False
    config.sequence_parallel = False
    config.gradient_accumulation_fusion = False
    config.expert_model_parallel_size = 1
    config.use_cpu_initialization = False

    return config
