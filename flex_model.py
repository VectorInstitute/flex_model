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

from utils import _recursively_find_first_tensor, _flatten, _unflatten

_LayerInputs = Any
_LayerOutputs = Any
_HookHandle = Any

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(funcName)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# TODO:
#   1. Larger test coverage
#   2. Test with different huggingface models
#   3. Implement multi-gpu single-node functionality (decouple)
#   4. Implement multi-gpu multi-node functionality


@dataclass
class HookFunctionTriple:
    module_name: str
    shape: Optional[Tuple[int]] = None
    editing_fn: Optional[Callable] = lambda x: x


class _HookFunction:
    """Object that assembles and exposes a forward/backward hook.

    Receives expected activation shape and optionally an activation
    editing function from the user. Defines the general function which
    will be exposed to the FlexModel and subsequently installed as a
    hook.
    """

    def __init__(
        self,
        hook_function_triple: HookFunctionTriple,
        output_ptr: Dict[str, Any],
    ):
        self._module_name = hook_function_triple.module_name
        self._shape = hook_function_triple.shape
        self._editing_fn = hook_function_triple.editing_fn
        self._hook_function_triple = hook_function_triple
        self._output_ptr = output_ptr

    def _parse(
        self,
        outputs: Union[_LayerOutputs, Tensor],
    ) -> Tuple[Tensor, Callable]:
        # Get container treedef and tensor leaf nodes
        treedef, leaves = _flatten(outputs)

        # Get the target tensor
        # TODO: Let user bias which leaf tensor to retrieve
        tensor, other_leaves = leaves[0], leaves[1:]

        # Define undo function to re-pack the edited activation tensor
        def _repack(
            _treedef,
            _leaves,
            _edited_tensor,
        ):
            return _unflatten(_treedef, [_edited_tensor] + _leaves)

        return tensor, partial(_repack, treedef, other_leaves)

    def _rearrange(
        self,
        activation: Tensor,
    ) -> Tuple[Tensor, Callable]:
        original_shape = activation.shape
        # Given current shape or shape not specified
        if original_shape == self._shape or self._shape is None:
            return activation, lambda y: y

        def _undo(_tensor, _shape):
            _tensor = _tensor.reshape(*_shape)
            return _tensor

        activation = activation.reshape(*self._shape)
        undo_fn = partial(_undo, _shape=original_shape)
        return activation, undo_fn

    def _edit(
        self,
        activation: Tensor,
    ) -> Tensor:
        if self._editing_fn is None:
            return activation
        else:
            return self._editing_fn(activation)

    def _bind(
        self,
        activation: Tensor,
    ) -> None:
        self._output_ptr[self._module_name] = activation.detach().cpu()

    def gen_hook_function(self):
        """Returns the hook function to be passed to PyTorch.

        Note: Needs to partial out the `self` arg in `_hook_function` to
        preserve the pointer to HookFunction class instance.
        """
        return partial(_hook_function, self)


def _hook_function(
    func: Any,
    module: nn.Module,
    inputs: Union[_LayerInputs, Tensor],
    outputs: Union[_LayerOutputs, Tensor],
) -> Optional[_LayerOutputs]:
    """Hook function implementation passed to PyTorch."""
    logger.info(f"Module {func._module_name} - Hook function activated")

    # Parse layer outputs
    tensor, repack_fn = func._parse(outputs)

    # Rearrange
    logger.info(f"Module {func._module_name} - initial shape: {tensor.shape}")
    tensor, undo_rearrange_fn = func._rearrange(tensor)
    logger.info(f"Module {func._module_name} - bind/edit shape: {tensor.shape}")

    # Edit
    tensor = func._edit(tensor)

    # Bind copy to output dict
    func._bind(tensor)

    # Undo rearrange
    tensor = undo_rearrange_fn(tensor)
    logger.info(f"Module {func._module_name} - return shape: {tensor.shape}")

    # Undo parse
    outputs = repack_fn(tensor)

    return outputs


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

    def register_hook_function_triple(
        self,
        hook_fn_triple: HookFunctionTriple,
    ) -> None:
        """Given user hook reqest, generate hook function and store it."""
        # Instantiate _HookFunction with user-provided hook data, then run
        # gen to create pytorch-facing hook function
        pytorch_hook_fn = _HookFunction(
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
