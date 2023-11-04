import os
from dataclasses import asdict, dataclass

import submitit
import torch

from tests.multi_gpu.registry import (
    SlurmJobResourceSpec,
    get_multigpu_resource_specs,
    get_multigpu_test_registry,
)


class TorchDistributedTestBatch:
    def __init__(self, test_functions):
        self.test_functions = test_functions

    def __repr__(self):
        repr_ = ""
        for name in self.test_functions.keys():
            repr_ = repr_ + "\n\t" + name
        return repr_

    def __call__(self):
        # Setup torch distributed environment.
        # NOTE: Allow each process to see all GPUs. Else we cannot properly call
        #       `torch.cuda.set_device(N)` when N > 0 since all processes will only
        #       have `CUDA_VISIBLE_DEVICES=0`.
        dist_env = submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )
        os.environ["NCCL_IB_DISABLE"] = "1"

        # Print distributed environment details.
        print(f"MASTER: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"RANK: {dist_env.rank}")
        print(f"WORLD_SIZE: {dist_env.world_size}")
        print(f"LOCAL_RANK: {dist_env.local_rank}")
        print(f"LOCAL_WORLD_SIZE: {dist_env.local_world_size}")

        # Run each test in batch.
        test_results = {}
        for name, fn in self.test_functions.items():
            try:
                print(f"Running test: {name}")
                res = fn()
                if dist_env.rank == 0:
                    test_results[name] = 0  # Success.

            # Test failure, record and continue.
            except AssertionError as assert_e:
                if dist_env.rank == 0:
                    test_results[name] = 1  # Failure.
                    print("AssertionError detected!")
                    print(assert_e)

            # Code or other error, crash.
            except Exception as e:
                print("Non-test exception detected!")
                print(e)
                raise SystemExit

        return test_results


class MultiGPUSlurmJob:
    def __init__(
        self,
        test_batch: TorchDistributedTestBatch,
        resource_spec: SlurmJobResourceSpec,
        log_dir: str = "multi_gpu_test_logs",
    ):
        self.res_spec = resource_spec
        self.test_batch = test_batch
        self.log_dir = log_dir

    def run(self):
        slurm_params = asdict(self.res_spec)
        python = slurm_params.pop("python", None)

        executor = submitit.SlurmExecutor(folder=self.log_dir, python=python)

        executor.update_parameters(**slurm_params)

        job = executor.submit(self.test_batch)

        submitit.helpers.monitor_jobs([job])

        results = job.results()[0]

        print("\n")
        for test_name, result in results.items():
            status = "SUCCESS" if result == 0 else "FAILURE"
            print(f"{test_name}: {result} ({status})")
        print("\n")

        return 0


def main():
    # Import folders to register tests.
    from tests.multi_gpu import core, distributed

    test_registries = get_multigpu_test_registry()

    # Construct one test batch per test file.
    # NOTE: Assumes that each test file has a single resource spec.
    test_batches = {}
    for test_reg_name, test_reg_fns in test_registries.items():
        test_batches[test_reg_name] = TorchDistributedTestBatch(test_reg_fns)

    print("Created test batches:")
    for name, batch in test_batches.items():
        print(f"{name}: {batch}")

    # Construct one resource spec per test file.
    test_resource_specs = get_multigpu_resource_specs()

    # Construct slurm job by combining each test batch with corresponding
    # resource spec.
    test_slurm_jobs = {}
    for test_batch_name in test_registries.keys():
        assert test_batch_name in test_resource_specs
        test_slurm_jobs[test_batch_name] = MultiGPUSlurmJob(
            test_batches[test_batch_name], test_resource_specs[test_batch_name],
        )

    # Run each job.
    # TODO: Launch all at once.
    for job in test_slurm_jobs.values():
        job.run()


if __name__ == "__main__":
    main()
