import argparse
import glob
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity

from examples.perf_testing.utils import (
    ExperimentNetworkManager,
    init_megatron_dist,
    spoof_megatron_config,
)
from flex_model.utils import setup_logger

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _add_profile_args(parser):
    group = parser.add_argument_group("profile")
    group.add_argument("--profile", action="store_true")
    group.add_argument("--profile_show", action="store_true")
    group.add_argument("--profile_warmup_steps", type=int, default=2)
    group.add_argument("--profile_active_steps", type=int, default=10)
    group.add_argument("--profile_wait_steps", type=int, default=1)
    group.add_argument("--profile_dir", type=str)
    group.add_argument("--profile_row_limit", type=int, default=5)
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group("distributed")
    group.add_argument("--tp_size", type=int)
    # parser.add_argument("--dp_size", type=int)
    # parser.add_argument("--pp_size", type=int)
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--dtypes", type=str, help="Comma-separated dtypes")
    parser.add_argument("--model_dim", type=int)
    parser.add_argument("--log_level", type=str, default="warning")
    parser.add_argument("--single_gpu_only", action="store_true")
    parser.add_argument("--multi_gpu_only", action="store_true")
    parser.add_argument("--n_layers", type=int, default=32)
    parser.add_argument("--debug", action="store_true")

    parser = _add_profile_args(parser)
    parser = _add_distributed_args(parser)

    args = parser.parse_args()

    args = validate_args(args)

    print_args(args)

    return args


def validate_args(args):
    # Manual override for one dtype.
    if args.dtypes is not None and args.dtypes in DTYPES:
        dtypes = []
        for d in args.dtypes.split(","):
            if d in DTYPES:
                dtypes.append(DTYPES[d])
            else:
                raise Exception(f"Unsupported dtype provided: {d}")
        args.dtype_sweep = dtypes
    else:
        args.dtype_sweep = list(DTYPES.values())

    # Manual override for one model dim.
    if args.model_dim is not None:
        args.model_dim_sweep = [args.model_dim]
    else:
        args.model_dim_sweep = [2 ** i for i in range(9, 15)]

    # Debug overrides both dtype and model dim to test all configurations.
    if args.debug:
        args.dtype_sweep = [v for v in DTYPES.values()]
        args.model_dim_sweep = [16, 32]

    # Determine which experiments to run.
    assert not (
        args.single_gpu_only and args.multi_gpu_only
    ), f"Cannot have both single gpu and multi gpu only flags both True."

    # TP is across all gpus by default.
    if args.tp_size is None:
        args.tp_size = int(os.environ["WORLD_SIZE"])

    # Make folder for profiling.
    if args.profile_dir is None:
        args.profile_dir = f"{os.getcwd()}/profiles/{os.environ['SLURM_JOB_ID']}"
    if not os.path.isdir(args.profile_dir):
        os.makedirs(args.profile_dir, exist_ok=True)

    return args


def print_args(args):
    # TODO: Pretty print.
    print(args)


def main(args):
    setup_logger(args.log_level)

    # Silence kineto warnings.
    os.environ["KINETO_LOG_LEVEL"] = "5"

    init_megatron_dist(args)
    rank = torch.distributed.get_rank()

    # Construct experiment setup functions.
    manager = ExperimentNetworkManager()
    experiments = manager.named_experiments
    prefix = ""
    if args.single_gpu_only:
        prefix = "single_gpu"
    elif args.multi_gpu_only:
        prefix = "multi_gpu"
    experiments = [getattr(manager, e) for e in experiments if e.startswith(prefix)]

    # Run benchmarks for each experiment, sweeping over parameters.
    for model_dim in args.model_dim_sweep:
        for dtype in args.dtype_sweep:
            for exp in experiments:
                exp_name = exp.__name__

                # Setup network.
                spoof_config = spoof_megatron_config(dtype)
                network = exp(
                    model_dim=model_dim, n_layers=args.n_layers, config=spoof_config,
                ).cuda()

                # Setup inputs.
                num_steps = (
                    args.profile_wait_steps
                    * args.profile_warmup_steps
                    * args.profile_active_steps
                )
                inputs = torch.randn(
                    num_steps, args.batch_size, model_dim, dtype=dtype
                ).cuda()

                # Run benchmark.
                schedule = torch.profiler.schedule(
                    wait=args.profile_wait_steps,
                    warmup=args.profile_warmup_steps,
                    active=args.profile_active_steps,
                    repeat=1,
                )
                with torch.profiler.profile(
                    schedule=schedule,
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        args.profile_dir
                    ),
                ) as prof:
                    for i in range(num_steps):
                        network(inputs[i])
                        prof.step()

                # Print profiles.
                if args.profile_show and rank == 0:
                    sort_variables = [
                        "self_cpu_time_total",
                        "self_cuda_time_total",
                        "self_cpu_memory_usage",
                        "self_cuda_memory_usage",
                    ]
                    print(" -> ".join(sort_variables))
                    key_avgs = prof.key_averages(group_by_input_shape=True)
                    print("=" * 160)
                    print(f"{exp_name}_{model_dim}_{dtype}")
                    for sort_var in sort_variables:
                        print(
                            key_avgs.table(
                                sort_by=sort_var, row_limit=args.profile_row_limit,
                            )
                        )

                # Cleanup.
                manager.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)