import logging
from argparse import Namespace
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

import flex_model.distributed as dist
from flex_model.distributed.mappings import unity
from flex_model.traverse import (
    InternalObject,
    LeafObject,
    ScalarObject,
    flatten,
    unflatten,
)

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


# Map: hook type -> nn.Module hook registry function.
_PYTORCH_HOOK_MAP = {
    "forward": "register_forward_hook",
    "backward": "register_full_backward_hook",
    "tensor_backward": "register_hook",
    "pre_forward": "register_forward_pre_hook",
    "pre_backward": "register_full_backward_pre_hook",
}


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
        hook_type: str = "forward",
    ) -> None:
        """Initializes the instance by wrapping the :code:`editing_function`.

        :param str module_name: Name of the :code:`nn.Module` submodule to hook
            into.
        :param expected_shape: Shape of the full activation tensor.
        :type expected_shape: Tuple[Optional[int], ...]
        :param editing_function: Function which edits the activation
            tensor.
        :type editing_function: Optional[Callable]
        :param str hook_type: Type of hook to register, eg. forward, backward,
            etc.
        """
        self.module_name = module_name
        self.expected_shape = expected_shape

        # TODO (mchoi): If editing function not passed (ie. just doing
        #               retrieval), then we can fire async collectives instead
        #               since there's no data dependency.
        if editing_function is None:
            self.editing_function = default_editing_function
        else:
            self.editing_function = editing_function

        assert hook_type in _PYTORCH_HOOK_MAP, f"Couldn't find hook type: {hook_type}"
        self.hook_type = hook_type
        self._hook_registry_function = _PYTORCH_HOOK_MAP[self.hook_type]

        self._collect: Optional[Callable] = None
        self._disperse: Optional[Callable] = None
        self._edit: Optional[Callable] = None
        self._dump: Optional[Callable] = None
        self._output_ptr: Optional[Dict[str, Tensor]] = None

        self.save_ctx: Optional[Namespace] = None
        self.modules: Optional[nn.ModuleDict] = None

        self._hook_type_to_handle_map = {
            "forward": self._handle_forward,
            "backward": self._handle_backward,
            "tensor_backward": self._handle_tensor_backward,
            "pre_forward": self._handle_pre_forward,
            "pre_backward": self._handle_pre_backward,
        }

    def _unpack_layer_outputs(
        self, outputs: Union[LayerOutputs, Tensor],
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

        def _repack(_treedef, _leaves, _edited_tensor,) -> LayerOutputs:
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
            tensor, self.expected_shape,
        )
        self._edit = _parse_edit_from_function(self.editing_function)
        self._dump = _parse_dump_from_function(self._default_bind_tensor_to_cpu_output,)

    def _dispatch_hook_function(
        self, hook_function_args: Tuple[Any, ...],
    ) -> Union[LayerOutputs, Tensor]:
        """Dispatches the correct handling function depending on the hook type.

        There are many different types of Pytorch hooks with varying function
        signatures. This function unpacks the Pytorch hook function input
        arguments depending on the hook type and dispatches the corresponding
        handling function.

        :note: The unpacking here is in constrast to the unpacking of layer
            outputs, which is done in the next step if needed.

        :returns: Potentially edited layer outputs.
            These outputs are sent as input to the next layer.
        :rtype: Union[LayerOutputs, Tensor]

        :raise NotImplementedError: The requested hook type isn't yet
            supported.
        """
        # TODO: Change hook granularity
        logger.debug(f"*{self.module_name}: Hook function activated*")

        handle_fn = self._hook_type_to_handle_map[self.hook_type]
        retval = handle_fn(*hook_function_args)

        return retval

    def _handle_forward(
        self,
        module: nn.Module,
        _inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Runs a hook function for editing forward module outputs."""
        outputs = self._template_for_input_output_editing(module, outputs)
        return outputs

    def _handle_backward(
        self,
        module: nn.Module,
        grad_inputs: Union[LayerOutputs, Tensor],
        _grad_outputs: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Runs a hook function for editing backward module input gradients."""
        outputs = self._template_for_input_output_editing(module, grad_inputs)
        return outputs

    def _handle_pre_forward(
        self, module: nn.Module, args: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Runs a hook function for editing forward module inputs."""
        # Same procedure as `_handle_forward`, just that we operate on args.
        outputs = self._template_for_input_output_editing(module, args)
        return outputs

    def _handle_pre_backward(
        self, module: nn.Module, grad_outputs: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Runs a hook function for editing backward module output gradients."""
        outputs = self._template_for_input_output_editing(module, grad_outputs)
        return outputs

    def _handle_tensor_backward(self, grad: Tensor,) -> Tensor:
        """Runs a hook function for editing tensor gradients."""
        # No module since this is tensor-level.
        outputs = self._template_for_point_tensor_editing(None, grad)
        return outputs

    def _template_for_point_tensor_editing(
        self, module: nn.Module, tensor: Tensor,
    ) -> Tensor:
        """Template function for editing a sharded activation tensor.

        This function is used alone in cases where hook functions operate
        directly on a tensor, and not an entire module.
        """
        start_shape = tensor.shape

        # Poplate the collection and dispersion functions.
        if self._collect is None or self._disperse is None:
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

        assert start_shape == end_shape, (
            f"Input tensor and output tensor shape mismatch: {start_shape} -> "
            f"{end_shape}. The tensor returned by the editing function must "
            f"not change in shape at the output."
        )

        return tensor

    def _template_for_input_output_editing(
        self, module: nn.Module, inputs_or_outputs: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Template function for editing layer input or output activation tensors.

        Given arbitary layer outputs, this function does unpacking of layer
        outputs and repacking of the potentially edited layer outputs.

        :param nn.Module module: Module which was hooked into.
        :param inputs_or_outputs: Layer inputs or outputs, depending on if
            it's hooked into a backward or forward hook respectively.
        :type inputs_or_outputs: Union[LayerOutputs, Tensor]

        :returns: The edited layer outputs.
        :rtype: Union[LayerOutputs, Tensor]
        """
        # Separate local activation tensor from rest of layer outputs.
        tensor, _repack_layer_outputs = self._unpack_layer_outputs(inputs_or_outputs)

        # Run editing logic on activation tensor.
        tensor = self._template_for_point_tensor_editing(module, tensor)

        # Repack the local activation tensor back into the rest of the layer
        # outputs.
        edited_inputs_or_outputs = _repack_layer_outputs(tensor)

        return edited_inputs_or_outputs

    def __call__(self, *args, **kwargs,) -> LayerOutputs:
        """Entrypoint called by Pytorch hook logic.

        Allows us to bind the entire :class:`HookFunction` to an :code:`nn.Module`
        using Pytorch hook registration.

        :note: Doesn't currently support accepting keyword argments passed into
            :code:`nn.Module`s.
        """
        if len(kwargs) != 0:
            raise NotImplementedError("HookFunction doesn't support kwargs.")

        outputs = self._dispatch_hook_function(args)
        return outputs
