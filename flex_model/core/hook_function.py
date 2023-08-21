from argparse import Namespace
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
from flex_model.distributed.mappings import unity


LayerOutputs = Union[InternalObject, LeafObject, ScalarObject]
logger = logging.getLogger(__name__)


def _parse_edit_from_function(edit_function):
    """Parse the user-provided editing function."""
    if edit_function is None:
        parsed_edit_function = default_editing_function
    else:
        parsed_edit_function = edit_function
    return parsed_edit_function


def _parse_dump_from_function(dump_function):
    """Parse the provided dump function."""
    return dump_function


def default_editing_function(
    current_module: nn.Module,
    inputs: Tensor,
    save_ctx: Namespace,
    modules: nn.ModuleDict,
) -> Tensor:
    logger.debug(f"Running default editing function on tensor: {inputs.shape}")
    return inputs


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

            Has signature (current_module, inputs, save_ctx, modules).
                current_module (nn.Module): Module hooked into.
                inputs (Tensor): Full activation tensor.
                save_ctx (Namespace): Global namespace to save state visible
                    to all hook (and editing) functions.
                modules (nn.ModuleDict): Global trainable pytorch layers
                    visible to all hook (and editing) functions.
    """

    def __init__(
        self,
        module_name: str,
        expected_shape: Tuple[Optional[int], ...],
        editing_function: Optional[Callable] = None,
    ) -> None:
        self.module_name = module_name
        self.expected_shape = expected_shape
        self.editing_function = editing_function

        self._collect: Optional[Callable] = None
        self._disperse: Optional[Callable] = None
        self._edit: Optional[Callable] = None
        self._dump: Optional[Callable] = None

        self._output_ptr: Optional[Dict[str, Tensor]] = None
        self.save_ctx: Optional[Namespace] = None
        self.modules: Optional[nn.ModuleDict] = None

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

    def _default_bind_tensor_to_cpu_output(self, activation: Tensor) -> None:
        """Bind the activation tensor to the output dict."""
        assert self._output_ptr is not None
        dumped_tensor = activation.detach().cpu()
        self._output_ptr[self.module_name] = dumped_tensor

    def _parse_tensor(self, tensor: Tensor) -> None:
        """Populate collect, disperse, edit and dump functions at runtime.

        The collection, dispersion, edit and dump functions are parsed from
        known quantities like the tensors in question across workers and the
        expected shape of the full activation. The collection and dispersion
        functions require communication to figure out the correct strategy,
        but this should only run once in the programs lifetime as hook
        functions retain state during the life of a `FlexModel`. This should
        also be the only state that is maintained by a hook function.
        """
        if torch.distributed.is_initialized():
            tp_world_size = dist.get_activation_tensor_parallel_world_size()
            dp_world_size = dist.get_activation_data_parallel_world_size()
        else:
            tp_world_size = 1
            dp_world_size = 1

        self._collect, self._disperse = dist.parse_collect_and_distribute_from_tensor(
            tensor,
            self.expected_shape,
        )

        self._edit = _parse_edit_from_function(self.editing_function)
        self._dump = _parse_dump_from_function(self._default_bind_tensor_to_cpu_output)

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

        # Need PP group members to send layer activations to head rank0
        if not torch.distributed.is_initialized() or (
            dist.activation_parallel_is_initialized()
            and dist.in_pipeline_parallel_group()
        ):
            # Dump then edit: See V-composition analysis algo
            self._dump(tensor)

            tensor = self._edit(module, tensor, self.save_ctx, self.modules)

        tensor = self._disperse(tensor)
        end_shape = tensor.shape

        outputs = _repack_layer_outputs(tensor)

        assert start_shape == end_shape

        return outputs

    def __call__(
        self,
        module: nn.Module,
        inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
    ) -> LayerOutputs:
        outputs = self._hook_function_template(module, inputs, outputs)
        return outputs
