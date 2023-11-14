import argparse
import os
import pprint

import torch
from torch.profiler import ProfilerActivity

from flex_model.core import FlexModel
from flex_model.utils import setup_logger
from profiling.utils import (
    ExperimentNetworkManager,
    init_megatron_dist,
    spoof_megatron_config,
)

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _add_profile_args(parser):
    group = parser.add_argument_group("profile")
    group.add_argument("--profile", action="store_true")
    group.add_argument("--profile_show", action="store_true")
    group.add_argument("--profile_save_profile", action="store_true")
    group.add_argument("--profile_warmup_steps", type=int, default=2)
    group.add_argument("--profile_active_steps", type=int, default=10)
    group.add_argument("--profile_wait_steps", type=int, default=1)
    group.add_argument("--profile_dir", type=str)
    group.add_argument("--profile_row_limit", type=int, default=5)
    group.add_argument("--profile_force_exp", type=str)
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group("distributed")
    group.add_argument("--tp_size", type=int)
    # TODO: Full Megatron-LM port.
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
        args.model_dim_sweep = [2**i for i in range(9, 15)]

    # Debug overrides both dtype and model dim to test all configurations.
    if args.debug:
        args.dtype_sweep = [v for v in DTYPES.values()]
        args.model_dim_sweep = [16, 32]

    # Determine which experiments to run.
    assert not (
        args.single_gpu_only and args.multi_gpu_only
    ), "Cannot have both single gpu and multi gpu only flags both True."

    # TP is across all gpus by default.
    if args.tp_size is None:
        args.tp_size = int(os.environ["WORLD_SIZE"])

    # Make folder for profiling.
    if args.profile_save_profile:
        if args.profile_dir is None:
            args.profile_dir = (
                f"{os.getcwd()}/profiles/profile_{os.environ['SLURM_JOB_ID']}"
            )
        if not os.path.isdir(args.profile_dir):
            os.makedirs(args.profile_dir, exist_ok=True)

    args.exp_prefix = ""
    if args.single_gpu_only:
        args.exp_prefix = "single_gpu"
    elif args.multi_gpu_only:
        args.exp_prefix = "multi_gpu"

    return args


def print_args(args):
    pp = pprint.PrettyPrinter(width=80)
    pp.pprint(vars(args))


def main(args):
    setup_logger(args.log_level)

    # Silence kineto warnings.
    os.environ["KINETO_LOG_LEVEL"] = "5"

    # Initialize distributed and megatron-lm parallel state.
    init_megatron_dist(args)
    rank = torch.distributed.get_rank()
    torch.manual_seed(rank)
    if rank == 0:
        print_args(args)

    # Construct experiment setup functions and create profile folders.
    manager = ExperimentNetworkManager()
    experiments = manager.get_experiment_handles(args.exp_prefix)

    for exp in experiments:
        os.makedirs(f"{args.profile_dir}/{exp.__name__}", exist_ok=True)

    num_steps = (
        args.profile_wait_steps
        * args.profile_warmup_steps
        * args.profile_active_steps
    )

    # Profiler setup.
    schedule = torch.profiler.schedule(
        wait=args.profile_wait_steps,
        warmup=args.profile_warmup_steps,
        active=args.profile_active_steps,
        repeat=1,
    )

    # Run benchmarks for each experiment, sweeping over parameters.
    for model_dim in args.model_dim_sweep:
        for dtype in args.dtype_sweep:
            # Setup inputs and spoof config.
            inputs = torch.randn(
                num_steps, args.batch_size, model_dim, dtype=dtype
            ).cuda()

            # Need to spoof Megatron-LM config so we can use the Col and Row
            # parallel layers.
            spoof_config = spoof_megatron_config(dtype)

            for exp in experiments:
                exp_name = exp.__name__
                if (
                    args.profile_force_exp
                    and exp_name != args.profile_force_exp
                ):
                    continue

                if args.profile_save_profile:
                    trace_handler = torch.profiler.tensorboard_trace_handler(
                        f"{args.profile_dir}/{exp.__name__}"
                    )
                else:
                    trace_handler = None

                # Setup network.
                network = exp(
                    model_dim=model_dim,
                    n_layers=args.n_layers,
                    config=spoof_config,
                ).cuda()

                # Run benchmark.
                with torch.profiler.profile(
                    schedule=schedule,
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_flops=True,
                    profile_memory=True,
                    on_trace_ready=trace_handler,
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
                    key_avgs = prof.key_averages(group_by_input_shape=True)
                    print("=" * 160)
                    print(f"{exp_name}_{model_dim}_{dtype}")
                    print(" -> ".join(sort_variables))
                    for sort_var in sort_variables:
                        print(
                            key_avgs.table(
                                sort_by=sort_var,
                                row_limit=args.profile_row_limit,
                            )
                        )

                # Cleanup.
                if isinstance(network, FlexModel):
                    network.restore()
                else:
                    manager.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
