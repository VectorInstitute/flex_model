import logging
from argparse import Namespace
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

import flex_model.distributed as fm_dist
from flex_model.traverse import (
    InternalObject,
    LeafObject,
    ScalarObject,
    flatten,
    unflatten,
)

LayerOutputs = Union[InternalObject, LeafObject, ScalarObject]
logger = logging.getLogger(__name__)


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
    :type expected_shape: Optional[Tuple[Optional[int], ...]]
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
        expected_shape: Optional[Tuple[Optional[int], ...]] = None,
        editing_function: Optional[Callable] = None,
        unpack_idx: int = 0,
    ) -> None:
        """Initializes the instance by wrapping the :code:`editing_function`.

        :param str module_name: Name of the :code:`nn.Module` submodule to hook
            into.
        :param expected_shape: Shape of the full activation tensor.
        :type expected_shape: Optional[Tuple[Optional[int], ...]]
        :param editing_function: Function which edits the activation
            tensor.
        :type editing_function: Optional[Callable]
        :param str hook_type: Type of hook to register, eg. forward, backward,
            etc.
        :param int unpack_idx: Index of the tensor in the unpacked layer output
            list. When layer outputs are pre-processed before editing function
            execution, valid `torch.Tensor` objects are extracted into a list
            by recursive unpacking. Hence the `unpack_idx` parameter allows
            for specification of which tensor to consider the activation
            tensor for downstream processing in the `HookFunction`.
        """
        # User-provided state.
        self.module_name = module_name
        self.expected_shape = expected_shape
        # TODO (mchoi): If editing function not passed (ie. just doing
        #               retrieval), then we can fire async collectives instead
        #               since there's no data dependency.
        if editing_function is None:
            self.editing_function = default_editing_function
        else:
            self.editing_function = editing_function
        self.unpack_idx = unpack_idx

        # FM instance registry-provided state and other runtime state.
        self._shared_state = None
        self._hook_type = None
        self.module = None  # Safe to cache this state, never changes.

        # Default strategies, initialized once at first runtime.
        # TODO: These should be bound when self is registered to a
        #       FlexModel instance.
        self.routing_strategy = fm_dist.ActivationTensorAllToAllRoutingStrategy
        self.offload_strategy = fm_dist.CPUPinnedMemoryOffloadStrategy
        self.function_strategy = fm_dist.NonValidatedFunctionStrategy

        # Valid hook function implementations.
        self.hook_type_to_impl_fn = {
            "forward": self._forward_hook_impl,
            "full_backward": self._full_backward_hook_impl,
            "tensor": self._tensor_hook_impl,
            "forward_pre": self._forward_pre_hook_impl,
            "full_backward_pre": self._full_backward_pre_hook_impl,
        }

    def _forward_hook_impl(
        self,
        module: nn.Module,
        _inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Runs a hook function for editing forward module outputs."""
        if self.module is None:
            self.module = module
        outputs = self._peel_and_apply(outputs)
        return outputs

    def _full_backward_hook_impl(
        self,
        module: nn.Module,
        grad_inputs: Union[LayerOutputs, Tensor],
        _grad_outputs: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Runs a hook function for editing backward module input gradients."""
        if self.module is None:
            self.module = module
        outputs = self._peel_and_apply(grad_inputs)
        return outputs

    def _tensor_hook_impl(
        self,
        grad: Tensor,
    ) -> Tensor:
        """Runs a hook function for editing tensor gradients."""
        # No module since this is tensor-level.
        outputs = self._apply(grad)
        return outputs

    def _forward_pre_hook_impl(
        self,
        module: nn.Module,
        args: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Runs a hook function for editing forward module inputs."""
        if self.module is None:
            self.module = module
        outputs = self._peel_and_apply(args)
        return outputs

    def _full_backward_pre_hook_impl(
        self,
        module: nn.Module,
        grad_outputs: Union[LayerOutputs, Tensor],
    ) -> Union[LayerOutputs, Tensor]:
        """Runs a hook function for editing backward module output gradients."""
        if self.module is None:
            self.module = module
        outputs = self._peel_and_apply(grad_outputs)
        return outputs

    def _apply(self, tensor: Optional[Tensor]) -> Tensor:
        """Template function for editing a sharded activation tensor.

        This function is used alone in cases where hook functions operate
        directly on a tensor, and not an entire module.
        """
        # Runtime initialization of strategies.
        # TODO: Only routing strategies need to be init at first iteration.
        if not isinstance(self.routing_strategy, fm_dist.BaseRoutingStrategy):
            self.routing_strategy = self.routing_strategy.initialize(
                self._shared_state.fmps,
                tensor,
                self.expected_shape,
            )
            self.offload_strategy = self.offload_strategy.initialize(
                self.module_name, self._shared_state.output_ptr
            )
            self.function_strategy = self.function_strategy.initialize(
                self.editing_function
            )

        if tensor is None:
            return

        start_shape = tensor.shape
        tensor = self.routing_strategy.execute_prologue(tensor)

        # Need to pre-divide activation grads by dp world size, see:
        # https://yi-wang-2005.medium.com/pytorch-distributeddataparallel-internals-c01c30a41192.
        if self._hook_type in ["full_backward", "full_backward_pre"]:
            tensor = (
                tensor / self._shared_state.fmps.get_data_parallel_world_size()
            )

        self.offload_strategy.execute(tensor)
        tensor = self.function_strategy.execute(
            self.module,
            tensor,
            self._shared_state.save_ctx,
            self._shared_state.modules,
        )

        if self._hook_type in ["full_backward", "full_backward_pre"]:
            tensor = (
                tensor * self._shared_state.fmps.get_data_parallel_world_size()
            )

        tensor = self.routing_strategy.execute_epilogue(tensor)
        end_shape = tensor.shape

        assert start_shape == end_shape, (
            f"Input tensor and output tensor shape mismatch: {start_shape} -> "
            f"{end_shape}. The tensor returned by the editing function must "
            f"not change in shape at the output."
        )

        return tensor

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

        if len(leaves) == 0:
            logger.debug(
                "Unpacked tensor is None, nothing to operate on "
                "(input activation grad is likely None)"
            )
            left_leaves, tensor, right_leaves = [], None, []
        else:
            left_leaves, tensor, right_leaves = (
                leaves[: self.unpack_idx],
                leaves[self.unpack_idx],
                leaves[self.unpack_idx + 1 :],
            )

        def _repack(_edited_tensor) -> LayerOutputs:
            """Pack activation tensor back into layer output container."""
            layer_outputs = unflatten(
                treedef,
                left_leaves + [_edited_tensor] + right_leaves,
            )
            return layer_outputs

        return tensor, _repack

    def _peel_and_apply(
        self,
        inputs_or_outputs: Union[LayerOutputs, Tensor],
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
        tensor, repack_fn = self._unpack_layer_outputs(inputs_or_outputs)

        tensor = self._apply(tensor)

        edited_inputs_or_outputs = repack_fn(tensor)

        return edited_inputs_or_outputs

    def _dispatch_hook_function(
        self,
        hook_function_args: Tuple[Any, ...],
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
        logger.debug(f"*{self.module_name}: Hook function activated*")

        handle_fn = self.hook_type_to_impl_fn[self._hook_type]
        retval = handle_fn(*hook_function_args)

        return retval

    def __call__(self, *args, **kwargs) -> LayerOutputs:
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
