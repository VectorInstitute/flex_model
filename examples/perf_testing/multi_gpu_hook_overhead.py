import argparse
import os
import random
from collections import defaultdict

import megatron.core.parallel_state as mpu
import numpy as np
import torch
import torch.nn as nn
import wandb
from megatron.core.parallel_state import initialize_model_parallel
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

from examples.perf_testing.utils import (
    Benchmark,
    ExperimentNetworkFactory,
    initialize_distributed,
    spoof_megatron_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_dtype", type=str, default="fp32")
    parser.add_argument("--profile_model_dim", type=int, default=4096)
    parser.add_argument("--profile_n_layers", type=int, default=4)
    parser.add_argument("--profile_exp", type=str, default="single_gpu_no_hooks")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def profile(args):
    factory = ExperimentNetworkFactory(args.profile_model_dim, args.profile_n_layers,)
    experiments = factory.named_experiments
    assert args.profile_exp in experiments

    dtype = DTYPES[args.dtype]
    network = getattr(factory, args.profile_exp)()
    network.to(dtype).cuda()

    inputs = torch.randn(1, args.batch_size, args.profile_model_dim, dtype=dtype).cuda()
    benchmark = Benchmark(network, inputs)

    profile = benchmark.run_profile()

    # TODO: Investigate best way to display profile.


def init_megatron_dist():
    os.environ["NCCL_IB_DISABLE"] = "1"
    initialize_distributed()
    initialize_model_parallel(torch.distributed.get_world_size())

    # Taken from: https://github.com/NVIDIA/Megatron-LM/blob/feac76a79148622d8f2a45d46c08a972a24784a3/megatron/initialize.py#L236
    seed = 0
    seed += 100 * mpu.get_pipeline_model_parallel_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        model_parallel_cuda_manual_seed(0)


def main(args):
    init_megatron_dist()
    rank = torch.distributed.get_rank()

    if args.profile:
        profile(args)
        return

    if args.debug:
        sweep_vars = {
            "dtype": [v for v in DTYPES.values()],
            "model_dim": [2 ** 5],
        }

    else:
        sweep_vars = {
            "dtype": [v for v in DTYPES.values()],
            "model_dim": [2 ** i for i in range(9, 15)],
        }
    metric_var_names = ["iter_time"]

    # Construct wandb metrics.
    if args.wandb and rank == 0:
        wandb.init(project="flex_model_exps", config={})
        wandb.define_metric("model_dim")

    # Construct experiment parameters for sweeps.
    exps = []
    for dtype in sweep_vars["dtype"]:
        for model_dim in sweep_vars["model_dim"]:
            exps.append((dtype, model_dim))

    # Run benchmarks for each experiment, sweeping over parameters.
    factory = ExperimentNetworkFactory()
    for exp_name in factory.named_experiments:
        for dtype in sweep_vars["dtype"]:
            if args.wandb and rank == 0:
                wandb.define_metric(
                    f"{exp_name}_{dtype}_iter_time", step_metric="model_dim"
                )
            for model_dim in sweep_vars["model_dim"]:
                spoof_config = spoof_megatron_config(dtype)

                # Get the network using current experiment params.
                network = getattr(factory, exp_name)(
                    model_dim=model_dim,
                    n_layers=32,  # TODO make this a sweep too.
                    config=spoof_config,
                )
                network.to(dtype).cuda()

                # Benchmark.
                inputs = torch.randn(
                    args.steps, args.batch_size, model_dim, dtype=dtype
                ).cuda()
                benchmark = Benchmark(network, inputs, warmup_iters=args.warmup_iters,)

                iter_time = benchmark.run_exp(args.steps)

                # Log benchmark results.
                if rank == 0:
                    print(f"{exp_name}_{model_dim}_{dtype}: {round(iter_time, 6)}s")
                    if args.wandb:
                        wandb.log(
                            {
                                "model_dim": model_dim,
                                f"{exp_name}_{dtype}_iter_time": iter_time,
                            }
                        )

                # Cleanup.
                factory.clear()


if __name__ == "__main__":
    args = parse_args()
    main(args)
