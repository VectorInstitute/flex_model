from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as pt_dist
import accelerate
from accelerate import PartialState

from flex_model.distributed.backends import (
    DistributedBackend,
    TorchDistributedBackend,
    AccelerateDistributedBackend,
    GPUDeviceMesh,
)


"""Backend-agnostic distributed launch and teardown.

User has some model on 1+ GPUs, using some sort of distributed backed
(typically accelerate or torch distributed). First find out which
backend is being used in the __init__method of the core `FlexModel` class.

Note: We always assume distributed backend is initialized already, ie. torch
has already called `init_process_groups`.

Flow:
-> FlexModel.__init__: Call to initialize_distributed_backend(...)
-> initialize_distributed_backend(...): Call to parse_backend(...)
    -> Returns accelerate, torch or single-GPU
-> initialize_distributed_backend(...): Call to GPUDeviceMesh.build(...)
    -> Returns GPUDeviceMesh populated with torch dist groups
-> initialize_distributed_backend(...): Initialize DistributedBackend(mesh)
-> Set DistributedBackend public API methods as global distributed prims
    -> Some fn. should take the DistributedBackend public API methods and link
       them to the __init__.py publically exposed functions
    -> Ie. Fn. registers DistributedBackend as as active, and all the
       exposed functions access the current active backend.
    -> Ex. def init_act_parallel():
                global ACTIVE_BACKEND
                return ACTIVE_BACKEND.init_act_parallel()
"""

_SUPPORTED_BACKENDS = {
    "torch": TorchDistributedBackend,
    "accelerate": AccelerateDistributedBackend,
}
_ACTIVE_BACKEND = None


def initialize_distributed_backend(
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
):
    """Main entry point from `FlexModel` to initialize distributed backend."""
    assert world_size == tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    backend_cls = parse_backend()
    device_mesh = GPUDeviceMesh.build(
        world_size,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    backend = backend_cls(device_mesh)
    expose_distributed_backend(backend)


def parse_backend():
    global _SUPPORTED_BACKENDS

    ps = PartialState()
    if (
        ps.distributed_type == accelerate.DistributedType.MULTI_GPU or
        ps.distributed_type == accelerate.DistributedType.FSDP or
        ps.distributed_type == accelerate.DistributedType.MEGATRON_LM
    ):
        hf_distributed = True
    else:
        hf_distributed = False

    # Using huggingface accelerate with torch
    if torch.distributed.is_initialized() and hf_distributed():
        return _SUPPORTED_BACKENDS["accelerate"]

    # Using torch distributed only. Single-gpu case is covered by torch
    # backend.
    elif (torch.distributed.is_initialized() and not hf_distributed() or
          not torch.distributed.is_initialized()):
        return _SUPPORTED_BACKENDS["torch"]

    # Unsupported
    else:
        raise NotImplementedError("Distributed backend currently not supported.")


def expose_distributed_backend(backend: DistributedBackend):
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def initialize_activation_parallel() -> None:
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND.initialize_activation_parallel()


def activation_parallel_is_initialized() -> bool:
    global _ACTIVE_BACKEND
    return _ACTIVE_BACKEND.activation_parallel_is_initialized()


def get_activation_tensor_parallel_world_size() -> int:
    global _ACTIVE_BACKEND
    return _ACTIVE_BACKEND.get_activation_tensor_parallel_world_size()
