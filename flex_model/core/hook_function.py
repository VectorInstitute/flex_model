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

import flex_model.distributed as dist
from flex_model.traverse import (
    InternalObject,
    LeafObject,
    ScalarObject,
    flatten,
    unflatten,
)


LayerOutputs = Union[InternalObject, LeafObject, ScalarObject]
logger = logging.getLogger(__name__)


def _parse_edit_from_function(edit_function):
    """Parse the user-provided editing function."""
    if edit_function is None:
        parsed_edit_function = _unity
    else:
        parsed_edit_function = edit_function
    return parsed_edit_function


def _parse_dump_from_function(dump_function):
    """Parse the provided dump function."""
    return dump_function


class HookFunction:
    """Respondible for frontend hook function via templating or codegen.

    Args:
        module_name (str):
            The module name to be hooked into.
        expected_shape (Tuple[int, ...]):
            Expected shape of the activation tensor.
        editing_function (Optional[Callable]):
            Optional editing function which will be applied to the activation
            tensor.
    """

    def __init__(
        self,
        module_name: str,
        expected_shape: Tuple[Optional[int], ...],
        editing_function: Optional[Callable] = lambda x: x,
    ) -> None:
        self.module_name = module_name
        self.expected_shape = expected_shape
        self.editing_function = editing_function
        self.trainable_modules: Dict[str, nn.Module] = {}

        self._collect: Optional[Callable] = None
        self._disperse: Optional[Callable] = None
        self._edit: Optional[Callable] = None
        self._dump: Optional[Callable] = None
        self._output_ptr: Optional[Dict[str, Tensor]] = None

    def _unpack_layer_outputs(
        self,
        outputs: Union[LayerOutputs, Tensor],
    ) -> Tuple[Tensor, partial]:
        """Parse out the activation tensor."""
        # Get container treedef and tensor leaf nodes
        # TODO: Let user bias which leaf tensor to retrieve
        treedef, leaves = flatten(outputs)
        tensor, other_leaves = leaves[0], leaves[1:]
        assert tensor is not None

        # TODO: Typecheck
        def _repack(
            _treedef,
            _leaves,
            _edited_tensor,
        ) -> Union[LayerOutputs]:
            """Pack activation tensor back into layer output container."""
            layer_outputs = unflatten(_treedef, [_edited_tensor] + _leaves)
            return layer_outputs

        return tensor, partial(_repack, treedef, other_leaves)

    def _bind_tensor_to_cpu_output(self, activation: Tensor) -> None:
        """Bind the activation tensor to the output dict."""
        assert self._output_ptr is not None
        dumped_tensor = activation.detach().cpu()
        self._output_ptr[self.module_name] = dumped_tensor

    def _parse_tensor(self, tensor: Tensor) -> None:
        self._collect, self._disperse = dist.parse_collect_and_distribute_from_tensor(
            tensor,
            self.expected_shape,
        )

        self._edit = _parse_edit_from_function(self.editing_function)
        self._dump = _parse_dump_from_function(self._bind_tensor_to_cpu_output)

    def _hook_function_template(
        self,
        module: nn.Module,
        inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
    ) -> Optional[LayerOutputs]:
        """Internal template for hook function."""
        logger.debug(f"*{self.module_name}: Hook function activated*")
        tensor, _repack_layer_outputs = self._unpack_layer_outputs(
            outputs,
        )
        # Debugging
        start_shape = tensor.shape

        if self._collect is None and self._disperse is None:
            self._parse_tensor(tensor)

        tensor = self._collect(tensor)

        if not dist.is_initialized() or (
            dist.is_initialized() and dist.get_rank() == 0
        ):
            # Dump then edit: See V-composition analysis algo
            self._dump(tensor)

        tensor = self._edit(tensor)

        tensor = self._disperse(tensor)
        end_shape = tensor.shape

        outputs = _repack_layer_outputs(tensor)

        assert start_shape == end_shape

        return outputs

    def register_trainable_module(self, name: str, module: nn.Module):
        self.trainable_modules[name] = module

    def requires_grad_(self, level: bool):
        for module in self.trainable_modules.values():
            module.requires_grad_(level)

    def __call__(
        self,
        module: nn.Module,
        inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
    ) -> LayerOutputs:
        outputs = self._hook_function_template(module, inputs, outputs)
        return outputs
