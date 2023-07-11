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

from flex_model._distributed_utils import (
    _set_activation_group,
    _infer_collective
)
from flex_model._common_utils import (
    _FlexModelState,
    _HookFunctionState,
    _init_distributed_model_state,
    _init_core_model_state,
    _init_runtime_model_state,
    _init_distributed_function_state,
    _init_core_function_state,
    _init_runtime_function_state
)
from flex_model._traverse_utils import (
    InternalObject,
    LeafObject,
    ScalarObject,
    _flatten,
    _unflatten,
)

LayerOutputs = Union[InternalObject, LeafObject, ScalarObject]
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(funcName)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# TODO: Implement hook functions which can add grad-requiring layers
class HookFunction(_HookFunctionState):
    def __init__(
        self,
        module_name: str,
        expected_shape: Tuple[int, ...],
        editing_function: Optional[Callable] = lambda x: x,
    ) -> None:
        _init_distributed_function_state(self)

        _init_core_function_state(
            self,
            module_name,
            expected_shape,
            editing_function,
        )

        _init_runtime_function_state(self)
        
    def _distributed_collect(self, tensor: Tensor) -> Tensor:
        """Applies appropriate comm. collective to put tensor on rank0."""
        if self._collect_function is None:
            collect_fn, distribute_fn = _infer_collective(
                tensor,
                self.expected_shape,
            )
            self._collect_fn_cache = collect_fn
            self._distribute_fn_cache = distribute_fn

            if isinstance(self._collect_fn_cache, partial):
                msg = (f"Rank{torch.distributed.get_rank()}: Collecting using "
                       f"{self._collect_fn_cache.func.__name__}")
            else:
                msg = (f"Rank{torch.distributed.get_rank()}: Collecting using "
                f"{self._collect_fn_cache.__name__}")
            logger.info(msg)

        return self._collect_fn_cache(tensor)

    def _distributed_redistribute(self, tensor: Tensor) -> Tensor:
        """Applies appropriate comm. collective to redistribute the collected
        tensor.
        """
        assert self._distribute_fn_cache is not None

        logger.info(
            f"Rank{torch.distributed.get_rank()}: Distributing using "
            f"{self._distribute_fn_cache.__name__}"
        )
        return self._distribute_fn_cache(tensor)

    def _parse_out_tensor(
        self,
        outputs: Union[LayerOutputs, Tensor],
    ) -> Tuple[Tensor, partial]:
        """Parse out the activation tensor."""
        # Get container treedef and tensor leaf nodes
        treedef, leaves = _flatten(outputs)

        # Get the target tensor
        # TODO: Let user bias which leaf tensor to retrieve
        tensor, other_leaves = leaves[0], leaves[1:]
        assert tensor is not None

        if self._using_torch_distributed:
            tensor = self._distributed_collect(tensor)

        # Define undo function to re-pack the edited activation tensor
        # TODO: Typecheck
        def _repack(
            _treedef,
            _leaves,
            _edited_tensor,
        ) -> Union[LayerOutputs, Tensor]:
            """Pack activation tensor back into layer output container."""
            if self._using_torch_distributed:
                _edited_tensor = self._distributed_redistribute(_edited_tensor)
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
        if (original_shape == self.expected_shape or
            self.expected_shape is None):
            return activation, lambda y: y

        def _undo(_tensor, _shape):
            _tensor = _tensor.reshape(*_shape)
            return _tensor

        activation = activation.reshape(*self.expected_shape)
        undo_fn = partial(_undo, _shape=original_shape)
        return activation, undo_fn

    def _edit(
        self,
        activation: Tensor,
    ) -> Tensor:
        """Apply editing function to activation tensor."""
        if self.editing_function is None:
            return activation
        else:
            return self.editing_function(activation)

    def _bind_tensor_to_output(
        self,
        activation: Tensor,
    ) -> None:
        """Bind the activation tensor to the output dict."""
        assert self._output_ptr is not None
        self._output_ptr[self.module_name] = activation.detach().cpu()

    def hook_function(
        self,
        module: nn.Module,
        inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
    ) -> Optional[LayerOutputs]:
        """Hook function implementation passed to PyTorch."""
        logger.info(f"Module {self.module_name} - Hook selftion activated")

        # Parse layer outputs
        tensor, repack_fn = self._parse_out_tensor(outputs)

        # Only rank0 gets the full activation tensor. Other ranks get their
        # corresponding tensor shards. This is useful when it comes to broadcast/
        # scatter outputs.
        if (torch.distributed.is_initialized() and
             torch.distributed.get_rank() == 0 or
             not torch.distributed.is_initialized()):
            # Rearrange
            logger.info(f"Module {self.module_name} - initial shape: "
                        f"{tensor.shape}")
            tensor, undo_rearrange_fn = self._rearrange(tensor)
            logger.info(f"Module {self.module_name} - bind/edit shape: "
                        f"{tensor.shape}")

            # Edit
            tensor = self._edit(tensor)

            # Bind copy to output dict
            self._bind_tensor_to_output(tensor)

            # Undo rearrange
            tensor = undo_rearrange_fn(tensor)
            logger.info(f"Module {self.module_name} - return shape: "
                        f"{tensor.shape}")

        # All ranks participate in repacking (broadcast/scatter included), implicit
        # barrier here.
        outputs = repack_fn(tensor)

        return outputs


class FlexModel(nn.Module, _FlexModelState):
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

        _init_distributed_model_state(self)

        _init_core_model_state(self, output_ptr)

        _init_runtime_model_state(self)

    def register_hook_function(
        self,
        hook_function: HookFunction,
    ) -> None:
        """Given user hook reqest, generate hook function and store it."""
        hook_function._output_ptr = self.output_ptr
        self.hook_fns[hook_function.module_name] = hook_function

    @contextmanager
    def _hook(self) -> Generator:
        for name, module in self.module.named_modules():
            if name in self.hook_fns:
                hook_handle = module.register_forward_hook(
                    self.hook_fns[name].hook_function
                )
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
