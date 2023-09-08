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


def _parse_edit_from_function(edit_function: Callable) -> Callable:
    """Parse the user-provided editing function.

    :note: This is the default parser for editing functions.

    :param Callable edit_function: User-defined editing function.

    :returns: Parsed editing function.
    :rtype: Callable
    """
    # TODO: Move to `distributed.parse`
    if edit_function is None:
        parsed_edit_function = default_editing_function
    else:
        parsed_edit_function = edit_function
    return parsed_edit_function


def _parse_dump_from_function(dump_function: Callable):
    """Parse the provided dump function.

    :note: This is the default parser for dump functions.

    :param Callable dump_function: Dump function.

    :returns: Parsed dump function.
    :rtype: Callable
    """
    # TODO: Move to `distributed.parse`
    return dump_function


def default_editing_function(
    current_module: nn.Module,
    inputs: Tensor,
    save_ctx: Namespace,
    modules: nn.ModuleDict,
) -> Tensor:
    """No-op editing function for logging and debug purposes.

    :note: This editing function showcases the expected function signature for
        custom editing functions.

    :note: If no editing function is provided for a :code:`HookFunction`, then
        this is the default editing function.

    :param nn.Module current_module: Submodule instance hooked into.
    :param Tensor inputs: Activation tensor produced during the forward pass of
        the :code:`current_module`.
    :param Namespace save_ctx: Save context pointer where cached data can be
        accessed or stored.
    :param nn.ModuleDict: Pointer to trainable modules globally exposed to all
        :class:`HookFunction` instances.

    :returns: Edited (or not) activation tensor
    :rtype: Tensor
    """
    logger.debug(f"Running default editing function on tensor: {inputs.shape}")
    return inputs


class HookFunction:
    """Function which retrieves/edits activations in a Pytorch `nn.Module`.

    The user provides the :code:`module_name` of the target submodule. The user
    can optionally pass in an :code:`editing_function` containing arbitrarily complex
    python code, which will be used to edit the full submodule activation
    tensor. If certain dimensions of the activation tensor are expected to be
    sharded over distributed workers, the user must also provide an
    :code:`expected_shape` hint so the activation tensor can be assembled.

    :var str module_name: Name of the :code:`nn.Module` submodule to hook into.
    :var expected_shape: Shape of the full activation tensor. Only the
        dimensions which are sharded need to be provided. Other dimensions
        can be annotated as :code:`None` and will be auto-completed.
    :type expected_shape: Tuple[Optional[int], ...]
    :var editing_function: Function which is run on the full activation tensor
        and returns some edited function. Global contexts like the
        save context and trainable modules are available for use in the
        editing function runtime.
    :type editing_function: Optional[Callable]
    :var save_ctx: Global save context that is exposed to the
        :code:`editing_function`.
    :type save_ctx: Optional[Namespace]
    :var modules: Global trainable modules that are exposed to the
        :code:`editing_function`.
    :type modules: Optional[nn.ModuleDict]

    :note: :code:`save_ctx` and :code:`modules` are populated when the :class:`HookFunction`
        is registered with a :class:`FlexModel` instance.

    Example:

    .. highlight:: python
    .. code-block:: python

        # Define editing function to be run on an activation tensor.
        def my_editing_function(current_module,
                                inputs,
                                save_ctx,
                                modules) -> Tensor:

            # Cache data for later.
            _, s, _ = torch.svd(inputs)
            save_ctx.activation_singular_values = s

            # Edit the activation tensor.
            inputs = torch.where(inputs > 1.0, inputs, 0.0)

            # Apply a torch layer to the activation tensor.
            outputs = modules["linear_projection"](inputs)
            
            # Pass edited activation tensor to next layer.
            return outputs

        # Instantiate registration-ready hook function.
        my_hook_function = HookFunction(
            "my_model.layers.16.self_attention",
            expected_shape=(4, 512, 5120),
            editing_function=my_editing_function,
        )
    """

    def __init__(
        self,
        module_name: str,
        expected_shape: Tuple[Optional[int], ...],
        editing_function: Optional[Callable] = None,
    ) -> None:
        """Initializes the instance by wrapping the :code:`editing_function`.

        :param str module_name: Name of the :code:`nn.Module` submodule to hook
            into.
        :param expected_shape: Shape of the full activation tensor.
        :type expected_shape: Tuple[Optional[int], ...]
        :param editing_function: Function which edits the activation
            tensor.
        :type editing_function: Optional[Callable]
        """
        self.module_name = module_name
        self.expected_shape = expected_shape

        if editing_function is None:
            self.editing_function = default_editing_function
        else:
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
        the :code:`FlexModel.traverse` library. Outputs are sorted into :code:`treedef`
        and :code:`leaves`. The :code:`treedef` define the structure of the object, and
        the :code:`leaves` correspond to a list of the found tensors. When the
        activation tensor needs to be sent to the next layer at the end of
        the :class:`HookFunction` execution, the returned :code:`_repack` function
        reconstructs the layer output.

        :param outputs: The current module's layer outputs.
        :type outputs: Union[LayerOutputs, Tensor]

        :returns: The (potentially sharded) activation
            tensor and a function to undo the unpacking operation.
        :rtype: Tuple[Tensor, partial]

        :raises AssertionError: Occurs if no tensor is found at all in the
            layer outputs.
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
        ) -> LayerOutputs:
            """Pack activation tensor back into layer output container."""
            layer_outputs = unflatten(_treedef, [_edited_tensor] + _leaves)
            return layer_outputs

        return tensor, partial(_repack, treedef, other_leaves)

    def _default_bind_tensor_to_cpu_output(self, activation: Tensor) -> None:
        """Binds the given activation tensor to the local CPU.

        Default function for binding an activation to the user-provided
        activation dictionary.

        :param Tensor activation: Full activation tensor to save.

        :raises AssertionError: Occurs if there is no activation dictionary that
            is bound the :class:`HookFunction` instance to save to.
        """
        assert self._output_ptr is not None
        dumped_tensor = activation.detach().cpu()
        self._output_ptr[self.module_name] = dumped_tensor

    def _parse_tensor(self, tensor: Tensor) -> None:
        """Runs parsers for collection/dispersion, editing and dumping.

        Populates the :code:`_collect`, :code:`_disperse`, :code:`_edit` and
        :code:`_dump` functions during runtime. The results of which depend on:
            1. The distributed device mesh
            2. The shape of the (potentially sharded) activation tensor
            3. The expected shape of the full activation tensor.

        :param Tensor tensor: (Potentially sharded) activation tensor to parse.
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
    ) -> Union[LayerOutputs, Tensor]:
        """Hook function implementation called by Pytorch after registration.

        Template function which contains the implementation of the hook
        function. See the :class:`HookFunction` docstring for details.

        :param nn.Module module: Current hooked module.
        :param inputs: Input to the current module.
        :type inputs: Union[LayerOutputs, Tensor]
        :param outputs: Output of the current module.
        :type outputs: Union[LayerOutputs, Tensor]

        :returns: Potentially edited layer outputs.
            These outputs are sent as input to the next layer.
        :rtype: Union[LayerOutputs, Tensor]

        :raise AssertionError: Start activation tensor shape is not the same as
            the ending activation tensor shape.
        """
        logger.debug(f"*{self.module_name}: Hook function activated*")

        # Separate local activation tensor from rest of layer outputs.
        tensor, _repack_layer_outputs = self._unpack_layer_outputs(
            outputs,
        )
        # Debugging
        start_shape = tensor.shape

        # Poplate the collection and dispersion functions.
        if self._collect is None and self._disperse is None:
            self._parse_tensor(tensor)

        # Collect the activation tensor across workers.
        tensor = self._collect(tensor)

        # Rank0 of each pipeline parallel group dumps to local output
        # dictionaries and runs editing functions.
        if not torch.distributed.is_initialized() or (
            dist.activation_parallel_is_initialized()
            and dist.in_pipeline_parallel_group()
        ):
            # Dump then edit: See V-composition analysis algo.
            self._dump(tensor)

            tensor = self._edit(module, tensor, self.save_ctx, self.modules)

        # Disperse the activation tensor back to respective workers.
        tensor = self._disperse(tensor)
        end_shape = tensor.shape

        # Repack the local activation tensor back into the rest of the layer
        # outputs.
        outputs = _repack_layer_outputs(tensor)

        assert start_shape == end_shape

        return outputs

    def __call__(
        self,
        module: nn.Module,
        inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
    ) -> LayerOutputs:
        """Entrypoint to forward hook implementation.

        Allows us to bind the entire :class:`HookFunction` to an :code:`nn.Module`
        using Pytorch hook registration.

        :param nn.Module module: Hooked submodule.
        :param inputs: Hooked submodule inputs.
        :type inputs: Union[LayerOutputs, Tensor]
        :param outputs: Hooked submodule outputs.
        :type outputs: Union[LayerOutputs, Tensor]

        :note: Inputs do not matter since hooks are run after the submodule has
            completed its forward pass.
        """
        outputs = self._hook_function_template(module, inputs, outputs)
        return outputs
