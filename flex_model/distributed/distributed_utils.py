from functools import partial, reduce
import logging
from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor

from flex_model.distributed.initialize import (
    is_initialized,
    get_rank,
)

logger = logging.getLogger(__name__)


def accelerate_distributed_is_initialized():
    ps = accelerate.PartialState()
    if (
        ps.distributed_type == accelerate.DistributedType.MULTI_GPU
        or ps.distributed_type == accelerate.DistributedType.FSDP
        or ps.distributed_type == accelerate.DistributedType.MEGATRON_LM
    ):
        return True
    else:
        return False


def print_rank0(msg: str) -> None:
    """Print to rank 0 worker."""
    if is_initialized() and get_rank() == 0:
        print(msg, flush=True)
