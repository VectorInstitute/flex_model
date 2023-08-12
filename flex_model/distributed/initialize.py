from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor


_ACTIVATION_PARALLEL_GROUP = None


def initialize_activation_parallel(ranks: List[int]) -> None:
    """Given a list of ranks, initialize a new `ProcessGroup`"""
    global _ACTIVATION_PARALLEL_GROUP

    assert torch.distributed.get_world_size() >= len(ranks)
    assert _ACTIVATION_PARALLEL_GROUP is None

    act_proc_group = torch.distributed.new_group(
        ranks=ranks,
        backend="nccl",
    )

    _ACTIVATION_PARALLEL_GROUP = act_proc_group


def get_activation_parallel_group() -> torch.distributed.ProcessGroup:
    """Return global activation processes group handle."""
    global _ACTIVATION_PARALLEL_GROUP
    return _ACTIVATION_PARALLEL_GROUP


def is_initialized() -> bool:
    return _ACTIVATION_PARALLEL_GROUP is not None


def get_world_size() -> int:
    """Return global activation processes world size."""
    if not torch.distributed.is_initialized():
        return 1

    return torch.distributed.get_world_size(
        group=get_activation_parallel_group(),
    )


def get_rank() -> int:
    if not is_initialized():
        return 0

    return torch.distributed.get_rank(
        group=get_activation_parallel_group(),
    )


def destroy_activation_parallel() -> None:
    """Set the groups to None."""
    global _ACTIVATION_PARALLEL_GROUP
    _ACTIVATION_PARALLEL_GROUP = None
