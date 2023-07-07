from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import logging
from typing import (
    List,
    Any,
    Dict,
    Optional,
    Callable,
    Tuple,
    Generator,
    Union,
)
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor


from flex_model.hook_functions import (
    HookFunctionTriple,
    _HookFunction, 
    _DistributedHookFunction,
)
from flex_model.utils import _recursively_find_first_tensor, _flatten, _unflatten
from flex_model.distributed_utils import (
    _set_activation_group,
)

_LayerInputs = Any
_LayerOutputs = Any
_HookHandle = Any

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(funcName)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FlexModel(nn.Module):
    """Wraps a PyTorch model and applies forward and backward hooks.

    Class which wraps a singular PyTorch model and installs/uninstalls
    forward and backward hooks. The use provides a list of modules and
    associated hook functions for each as input. To generate activations
    as output, the user calls the forward function of the wrapped model
    or optionally from the FlexModel. Contains no additional state
    besides metadata and hook functions.

    NOTE: See `torch.distributed.fsdp` for pytorch model wrapper example.
    """

    def __init__(
        self,
        module: nn.Module,
        output_ptr: Dict[str, Any],
    ):
        super().__init__()

        self.module = module
        self.hook_fns: Dict[str, Callable] = {}
        self.output_ptr = output_ptr

        self._hook_fn_handles: Dict[str, _HookHandle] = {}

        self._hook_fn_impl = _HookFunction

    def register_hook_function_triple(
        self,
        hook_fn_triple: HookFunctionTriple,
    ) -> None:
        """Given user hook reqest, generate hook function and store it."""
        # Instantiate _HookFunction with user-provided hook data, then run
        # gen to create pytorch-facing hook function
        pytorch_hook_fn = self._hook_fn_impl(
            hook_fn_triple,
            self.output_ptr,
        ).gen_hook_function()
        self.hook_fns[hook_fn_triple.module_name] = pytorch_hook_fn

    @contextmanager
    def _hook(self) -> Generator:
        for name, module in self.module.named_modules():
            if name in self.hook_fns:
                hook_handle = module.register_forward_hook(self.hook_fns[name])
                self._hook_fn_handles[name] = hook_handle

                logger.info(f"Installing module: {name} forward hook")

        try:
            yield
        finally:
            for hook in self._hook_fn_handles.values():
                hook.remove()
        self._hook_fn_handles.clear()

    def forward(self, *args, **kwargs) -> Any:
        """Run forward of wrapped model."""

        with self._hook():
            logger.info("Running forward")
            outputs = self.module(*args, **kwargs)
        return outputs


class DistributedFlexModel(FlexModel):
    def __init__(self, ranks: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranks = ranks

        _set_activation_group(self.ranks)
        self._hook_fn_impl = _DistributedHookFunction

        logger.info(f"Initialzed DistributedFlexModel on "
                    f"rank{torch.distributed.get_rank()}")