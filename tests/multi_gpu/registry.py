from dataclasses import dataclass
from typing import Optional

_MULTIGPU_TESTS_REGISTRY = None
_MULTIGPU_RESOURCE_SPECS = None


@dataclass
class SlurmJobResourceSpec:
    """Resource specification for a single slurm job."""

    partition: str = "a100"
    qos: str = "a100_mchoi"
    # python: str = "/h/mchoi/projects/dl_lc/dl_lc_env/bin/python3.9"
    time: int = 5
    mem: Optional[str] = None
    mem_per_gpu: str = "32G"
    nodes: int = 1
    gpus_per_node: int = 4
    ntasks_per_node: int = 4
    cpus_per_task: int = 6

    def __post_init__(self):
        # Providing mem overrides mem_per_gpu.
        if self.mem is not None:
            self.mem_per_gpu = None


def make_test_registry(
    registry_name,
    resource_spec: SlurmJobResourceSpec = None,
):
    global _MULTIGPU_TESTS_REGISTRY
    global _MULTIGPU_RESOURCE_SPECS
    if _MULTIGPU_TESTS_REGISTRY is None:
        _MULTIGPU_TESTS_REGISTRY = {}
    if _MULTIGPU_RESOURCE_SPECS is None:
        _MULTIGPU_RESOURCE_SPECS = {}

    # Defaults.
    if resource_spec is None:
        resource_spec = SlurmJobResourceSpec()

    _MULTIGPU_TESTS_REGISTRY[registry_name] = {}
    _MULTIGPU_RESOURCE_SPECS[registry_name] = resource_spec

    def _register_fn(fn):
        """Register a test to run in a multi-gpu setting."""
        fn_name = fn.__name__
        _MULTIGPU_TESTS_REGISTRY[registry_name][fn_name] = fn

        return fn

    def _get_fn():
        assert (
            _MULTIGPU_TESTS_REGISTRY is not None
        ), "Multi-gpu test registry is uninitialized or empty"
        return _MULTIGPU_TESTS_REGISTRY[registry_name]

    return _register_fn, _get_fn


def get_multigpu_test_registry():
    global _MULTIGPU_TESTS_REGISTRY
    assert _MULTIGPU_TESTS_REGISTRY is not None
    return _MULTIGPU_TESTS_REGISTRY


def get_multigpu_resource_specs():
    global _MULTIGPU_RESOURCE_SPECS
    assert _MULTIGPU_RESOURCE_SPECS is not None
    return _MULTIGPU_RESOURCE_SPECS
