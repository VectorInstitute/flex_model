from functools import partial, reduce
import logging
from typing import List, Tuple, Callable, Optional

import accelerate
import torch
from torch import Tensor

from flex_model.distributed.initialize import (
    is_initialized,
    get_rank,
)

logger = logging.getLogger(__name__)


def print_rank0(msg: str) -> None:
    """Print to rank 0 worker."""
    if is_initialized() and get_rank() == 0:
        print(msg, flush=True)
