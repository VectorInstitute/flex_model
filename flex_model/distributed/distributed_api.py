"""Backend-agnostic distributed launch and teardown.

User has some model on 1+ GPUs, using some sort of distributed backed
(typically accelerate or torch distributed). First find out which
backend is being used in the __init__method of the core `FlexModel` class.

Notes:
- We always assume distributed backend is initialized already, ie. torch
  has already called `init_process_groups`.
- Additionally, we leave primitives like `torch.dsitributed.get_rank()` to
  torch instead of wrapping them.

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
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import List, Optional

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


_SUPPORTED_BACKENDS = {
    "torch": TorchDistributedBackend,
    "accelerate": AccelerateDistributedBackend,
}
_ACTIVE_BACKEND: Optional[DistributedBackend] = None

logger = logging.getLogger(__name__)


def initialize_distributed_backend(
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
):
    """Main entry point from `FlexModel` to initialize distributed backend."""
    assert world_size == tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    backend_cls = _parse_backend()
    logger.debug(f"Using DistributedBackend: {backend_cls.__name__}")

    device_mesh = GPUDeviceMesh.build(
        world_size,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )

    backend = backend_cls(device_mesh)
    _expose_distributed_backend(backend)


def distributed_backend_is_initialized():
    global _ACTIVE_BACKEND
    return _ACTIVE_BACKEND is not None


def destroy_distributed_backend():
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = None


def _parse_backend():
    """Figure out what distributed backend to use given current distributed state."""
    global _SUPPORTED_BACKENDS

    ps = PartialState()

    if (
        ps.distributed_type == accelerate.DistributedType.DEEPSPEED or
        ps.distributed_type == accelerate.DistributedType.FSDP or
        ps.distributed_type == accelerate.DistributedType.MEGATRON_LM
    ):
        hf_distributed = True
    else:
        hf_distributed = False

    # Using huggingface accelerate with torch
    if torch.distributed.is_initialized() and hf_distributed:
        return _SUPPORTED_BACKENDS["accelerate"]

    # Using torch distributed only. Single-gpu case is covered by torch
    # backend.
    elif (torch.distributed.is_initialized() and not hf_distributed or
          not torch.distributed.is_initialized()):
        return _SUPPORTED_BACKENDS["torch"]

    # Unsupported
    else:
        raise NotImplementedError("Distributed backend currently not supported.")


def _expose_distributed_backend(backend: DistributedBackend):
    """Set the global distributed backend."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def initialize_activation_parallel() -> None:
    """Initialize activation parallel distributed groups."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    _ACTIVE_BACKEND.initialize_activation_parallel()


def activation_parallel_is_initialized() -> bool:
    """Check if activation parallel distributed groups have been initialized."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.activation_parallel_is_initialized()


def in_tensor_parallel_group() -> bool:
    """Check if current worker belongs to a tensor parallel group."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.in_tensor_parallel_group()

def in_pipeline_parallel_group() -> bool:
    """Check if current worker belongs to a pipeline parallel group."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.in_pipeline_parallel_group()

def in_data_parallel_group() -> bool:
    """Check if current worker belongs to a data parallel group."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.in_data_parallel_group()


def get_activation_tensor_parallel_group() -> Optional[pt_dist.ProcessGroup]:
    """Get the activation parallel tp group."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_tensor_parallel_group()


def get_activation_data_parallel_group() -> Optional[pt_dist.ProcessGroup]:
    """Get the activation parallel dp group."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_data_parallel_group()


def get_activation_pipeline_parallel_group() -> Optional[pt_dist.ProcessGroup]:
    """Get the activation parallel dp group."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_pipeline_parallel_group()


def get_activation_tensor_parallel_world_size() -> int:
    """Get the activation parallel tp group world size."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_tensor_parallel_world_size()


def get_activation_data_parallel_world_size() -> int:
    """Get the activation parallel dp group world size."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_data_parallel_world_size()


def get_activation_pipeline_parallel_world_size() -> int:
    """Get the activation parallel dp group world size."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_pipeline_parallel_world_size()


def get_activation_tensor_parallel_rank() -> int:
    """Get the activation parallel tp group world size."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_tensor_parallel_rank()


def get_activation_data_parallel_rank() -> int:
    """Get the data parallel dp group world size."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_data_parallel_rank()


def get_activation_pipeline_parallel_rank() -> int:
    """Get the data parallel dp group world size."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    return _ACTIVE_BACKEND.get_activation_pipeline_parallel_rank()


def destroy_activation_parallel() -> None:
    """Destroy the activation parallel groups."""
    global _ACTIVE_BACKEND
    assert _ACTIVE_BACKEND is not None
    _ACTIVE_BACKEND.destroy_activation_parallel()
