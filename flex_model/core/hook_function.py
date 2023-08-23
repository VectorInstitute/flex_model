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
    """Function which retrieves/edits activations in a Pytorch `nn.Module`.

    Wraps a user-provided `editing_function` which must have the same function
    signature as `default_editing_function`. The runtime within the editing
    function is single-threaded, so the user does not have to think about
    how the model is sharded across processes. An instance of this class is
    registered into a submodule with the same `module_name`. The full
    activation tensor is materialized and exposed to the `editing_function`
    runtime by parsing the `expected_shape`.

    Activations are processed in a fixed way, templated out in
    `_hook_function_template`. The local activation tensor is parsed according
    to the expected shape and distributed device mesh to figure out what
    collective communication functions are needed to gather/disperse it. The
    collection function is run to materialize the full activation tensor. Then
    the pipeline parallel rank0 processes dump the tensor to their CPU. Next,
    the user-provided `editing_function` is run on the full activation tensor.
    Finally, the activation tensor is dispersed back to the necessary ranks
    for further propagation of the forward pass. Also note that layer outputs
    can be arbitrary python objects, so unpacking/repacking is done using
    the `FlexModel.traversal` library.

    Attributes:
        module_name: Name of the `nn.Module` submodule to hook into.
        expected_shape: Shape of the full activation tensor. Only the
            dimensions which are sharded need to be provided. Other dimensions
            can be annotated as `None` and will be auto-completed.
        editing_function: Function which is run on the full activation tensor
            and returns some edited function. Global contexts like the
            save context and trainable modules are available for use in the
            editing function runtime.
        _collect: Collective communication function which gathers the sharded
            activation tensors into a full activation tensor.
        _disperse: Collective communication function which disperses the
            full activation tensor into sharded activation tensors on their
            respective ranks.
        _edit: See `editing_function`.
        _dump: Dump function used to save full activation tensors to CPU.
        _output_ptr: Output dictionary keyed by module name to save the
            full activations to.
        save_ctx: Global save context that is exposed to the
            `editing_function`.
        modules: Global trainable modules that are exposed to the
            `editing_function`.
    """

    def __init__(
        self,
        module_name: str,
        expected_shape: Tuple[Optional[int], ...],
        editing_function: Optional[Callable] = None,
    ) -> None:
        """Initializes the instance by wrapping the `editing_function`.

        Args:
            See class docstring.
        """
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
        """Converts layer output object into an activation tensor and def.

        The output of model layers can be arbitrary python objects, so this
        function unpacks this object and separates out Pytorch tensors using
        the `FlexModel.traverse` library. Outputs are sorted into `treedef`s
        and `leaves`. The `treedef`s define the structure of the object, and
        the `leaves` correspond to a list of the found tensors. When the
        activation tensor needs to be sent to the next layer at the end of
        the `HookFunction` execution, the returned `_repack` function
        reconstructs the layer output.

        Args:
            outputs: The current module's layer outputs.

        Returns:
            The (potentially sharded) activation tensor and a function to
            reverse the unpacking operation.

        Raises:
            AssertionError: Occurs if no tensor is found at all in the layer
                outputs.
        """
        treedef, leaves = flatten(outputs)

        # TODO: Current behaviour is always taking the first leaf as the
        #       activation tensor. But this should be exposed as configurable
        #       to the user.

        tensor, other_leaves = leaves[0], leaves[1:]
        assert tensor is not None

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
        """Binds the given activation tensor to the local CPU.

        Default function for binding an activation to the user-provided
        activation dictionary.

        Args:
            activation: Full activation tensor to save.

        Raises:
            AssertionError: Occurs if there is no activation dictionary that
                is bound the `HookFunction` instance to save to.
        """
        assert self._output_ptr is not None
        dumped_tensor = activation.detach().cpu()
        self._output_ptr[self.module_name] = dumped_tensor

    def _parse_tensor(self, tensor: Tensor) -> None:
        """Runs parsers for collection/dispersion, editing and dumping.

        Populates the `_collect`, `_disperse`, `_edit` and `_dump` functions
        during runtime depending on:
            1. The distributed device mesh
            2. The shape of the (potentially sharded) activation tensor
            3. The expected shape of the full activation tensor.

        Args:
            tensor: (Potentially sharded) activation tensor to parse.

        Returns:
            Nothing, but populates the attributes in the `HookFunction`
            instance.
        """
        self._collect, self._disperse = dist.parse_collect_and_distribute_from_tensor(
            tensor,
            self.expected_shape,
        )
        self._edit = _parse_edit_from_function(self.editing_function)
        self._dump = _parse_dump_from_function(
            self._default_bind_tensor_to_cpu_output,
        )

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
