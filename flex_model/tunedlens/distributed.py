import logging
from typing import List, Dict, Tuple, Callable, Optional

import torch
import torch.distributed as dist

from flex_model.core import FlexModel, HookFunction


logger = logging.getLogger(__name__)
LENS_MODEL_PARALLEL_GROUP = None


def initialize_lens_model_parallel(lens_model_parallel_size: int = 1) -> None:
    """Initialize lens model parallel group."""
    global LENS_MODEL_PARALLEL_GROUP
    assert LENS_MODEL_PARALLEL_GROUP is None

    ranks = list(range(lens_model_parallel_size))
    assert dist.get_world_size() >= len(ranks)

    LENS_MODEL_PARALLEL_GROUP = dist.new_group(
        ranks=ranks,
        backend="nccl",
    )


def is_initialized():
    global LENS_MODEL_PARALLEL_GROUP
    return LENS_MODEL_PARALLEL_GROUP is not None


def get_lens_model_parallel_group() -> dist.ProcessGroup:
    """Return the lens model parallel group."""
    global LENS_MODEL_PARALLEL_GROUP
    return LENS_MODEL_PARALLEL_GROUP


def get_lens_model_parallel_world_size() -> int:
    """Return the lens model parallel group (world) size."""
    return dist.get_world_size(group=get_lens_model_parallel_group())


def get_lens_model_parallel_rank() -> int:
    """Return the lens model parallel rank."""
    return dist.get_rank(group=get_lens_model_parallel_group())


def destroy_lens_model_parallel():
    global LENS_MODEL_PARALLEL_GROUP
    LENS_MODEL_PARALLEL_GROUP = None
