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
from distributed_utils import _infer_collective

_LayerInputs = Any
_LayerOutputs = Any
_HookHandle = Any

logger = logging.getLogger(__name__)


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
        # TODO: If editing fn = unity(), enable async collectives
        self._editing_fn = hook_function_triple.editing_fn
        self._hook_function_triple = hook_function_triple
        self._output_ptr = output_ptr

    def _parse(
        self,
        outputs: Union[_LayerOutputs, Tensor],
    ) -> Tuple[Tensor, Callable]:
        """Parse out the activation tensor."""
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
        """
        Reshape a tensor, and return it alongside the inverse of the reshape.
        """
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

        NOTE: We want to pass the function without running it, but
        binding of `self` happens implicitly on call.
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


class _DistributedHookFunction(_HookFunction):
    """Implements distributed communication functionality to hook functions."""

    def __init__(
        self,
        hook_function_triple: HookFunctionTriple,
        output_ptr: Dict[str, Any],
    ) -> None:
        super().__init__(hook_function_triple, output_ptr)
        assert self._shape is not None
        assert torch.distributed.is_initialized()

        # TODO: After call to _infer_collective, we can cache the corresponding
        #       collective functions
        self._collect_fn_cache = None
        self._distribute_fn_cache = None

    def _collect(self, tensor: Tensor) -> Tensor:
        """Applies appropriate comm. collective to put tensor on rank0."""
        if self._collect_fn_cache is None:
            collect_fn, distribute_fn = _infer_collective(
                tensor,
                self._shape,
            )
            self._collect_fn_cache = collect_fn
            self._distribute_fn_cache = distribute_fn

        logger.info(
            f"Rank{torch.distributed.get_rank()}: Collecting using {self._collect_fn_cache.__name__}"
        )
        return self._collect_fn_cache(tensor)

    def _distribute(self, tensor: Tensor) -> Tensor:
        """Applies appropriate comm. collective to redistribute the collected
        tensor.
        """
        assert self._distribute_fn_cache is not None

        logger.info(
            f"Rank{torch.distributed.get_rank()}: Distributing using {self._distribute_fn_cache.__name__}"
        )
        return self._distribute_fn_cache(tensor)
