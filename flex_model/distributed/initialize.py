from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor


_GLOBAL_ACTIVATION_GROUP = None


def init_activation_parallel_group(ranks: List[int]) -> None:
    """Given a list of ranks, initialize a new `ProcessGroup`"""
    global _GLOBAL_ACTIVATION_GROUP

    assert torch.distributed.get_world_size() >= len(ranks)
    assert _GLOBAL_ACTIVATION_GROUP is None

    act_proc_group = torch.distributed.new_group(
        ranks=ranks,
        backend="nccl",
    )

    _GLOBAL_ACTIVATION_GROUP = act_proc_group


def get_activation_parallel_group() -> torch.distributed.ProcessGroup:
    """Return global activation processes group handle."""
    global _GLOBAL_ACTIVATION_GROUP
    assert _GLOBAL_ACTIVATION_GROUP is not None
    return _GLOBAL_ACTIVATION_GROUP


def is_initialized() -> bool:
    return torch.distributed.is_initialized()


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
