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

import flex_model._distributed_utils as dist
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

    def _unpack_layer_outputs(
        self,
        outputs: Union[LayerOutputs, Tensor],
    ) -> Tuple[Tensor, partial, torch.Size]:
        """Parse out the activation tensor."""
        # Get container treedef and tensor leaf nodes
        # TODO: Let user bias which leaf tensor to retrieve
        treedef, leaves = _flatten(outputs)
        tensor, other_leaves = leaves[0], leaves[1:]
        assert tensor is not None

        start_shape = tensor.size()

        # TODO: Typecheck
        def _repack(
            _treedef, _leaves, _edited_tensor,
        ) -> Union[LayerOutputs, torch.Size]:
            """Pack activation tensor back into layer output container."""
            end_shape = _edited_tensor.size()
            layer_outputs = _unflatten(_treedef, [_edited_tensor] + _leaves)
            return layer_outputs, end_shape

        return tensor, partial(_repack, treedef, other_leaves), start_shape

    def _bind_tensor_to_cpu_output(self, activation: Tensor) -> None:
        """Bind the activation tensor to the output dict."""
        assert self._output_ptr is not None
        dumped_tensor = activation.detach().cpu() #.reshape(*self.expected_shape)
        self._output_ptr[self.module_name] = dumped_tensor

    def _parse_tensor(self, tensor: Tensor) -> None:
        self._collect, self._disperse = dist._parse_collect_and_distribute_from_tensor(
            tensor, self.expected_shape,
        )
        self._edit = dist._parse_edit_from_function(self.editing_function)
        self._dump = dist._parse_dump_from_function(self._bind_tensor_to_cpu_output)
        
    def _hook_function_template(
        self,
        module: nn.Module,
        inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
        _hook_trainable_modules: nn.ModuleDict,
    ) -> Optional[LayerOutputs]:
        """Internal template for hook function."""
        dist.print_rank0(f"*{self.module_name}: Hook function activated*")
        tensor, _repack_layer_outputs, start_shape = self._unpack_layer_outputs(
            outputs,
        )

        if self._collect is None and self._disperse is None:
            self._parse_tensor(tensor)

        tensor = self._collect(tensor)

        if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
            # Dump then edit: See V-composition analysis algo
            self._dump(tensor)

        # Edit function API: Tensor edit_fn (Tensor, hook_trainable_modules)
        tensor = self._edit(tensor, _hook_trainable_modules)

        tensor = self._disperse(tensor)

        outputs, end_shape = _repack_layer_outputs(tensor)

        assert start_shape == end_shape

        return outputs

    def hook_function(
        self,
        module: nn.Module,
        inputs: Union[LayerOutputs, Tensor],
        outputs: Union[LayerOutputs, Tensor],
        _hook_trainable_modules: nn.ModuleDict,
    ) -> Optional[LayerOutputs]:
        """Public API for hook function.

        Hook function that the `FlexModel` will register and apply. Underscored
        parameters are not included in the public API for torch hooks and are
        partial'd out.
        """
        outputs = self._hook_function_template(
            module,
            inputs,
            outputs,
            _hook_trainable_modules,
        )
        return outputs


class FlexModel(nn.Module, _FlexModelState):
    """Wraps a PyTorch model and applies forward and backward hooks.

    Class which wraps a singular PyTorch model and installs/uninstalls
    forward and backward hooks. The use provides a list of modules and
    associated hook functions for each as input. To generate activations
    as output, the user calls the forward function of the wrapped model
    or optionally from the FlexModel. Contains no additional state
    besides metadata and hook functions.

    Note: See `torch.distributed.fsdp` for pytorch model wrapper example.

    Args:
        module (nn.Module):
            PyTorch module to be wrapped.
        output_ptr (Dict[str, Tensor]):
            Object where retrieved activation are bound to during execution of
            the hook function.
    """
    def __init__(
        self,
        module: nn.Module,
        output_ptr: Dict[str, Tensor],
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
        hook_fn_to_reg = partial(
            hook_function.hook_function,
            _hook_trainable_modules=self.hook_trainable_modules,
        )
        self.hook_fns[hook_function.module_name] = hook_fn_to_reg

    def register_hook_trainable_module(
        self,
        name: str,
        submodule: nn.Module
    ) -> None:
        self.hook_trainable_modules[name] = submodule

    def set_hooks(self) -> None:
        for name, module in self.module.named_modules():
            if name in self.hook_fns:
                _handle = module.register_forward_hook(
                    self.hook_fns[name]
                )
                self._hook_fn_handles[name] = _handle

                dist.print_rank0(f"Installing module: {name} forward hook")

    def remove_hooks(self) -> None:
        for hook in self._hook_fn_handles.values():
            hook.remove()

        self._hook_fn_handles.clear()

    @contextmanager
    def _hook(self) -> Generator:
        """Context manager for applying forward hooks."""
        # TODO: Calling module.forward doesn't run forward hooks. Is there a
        #       way to force the call to module.__call__ instead?
        self.set_hooks()
        try:
            yield

        finally:
            self.remove_hooks()

    def freeze_wrapped_model(self) -> None:
        self.module.requires_grad_(False)

    def unfreeze_wrapped_model(self) -> None:
        self.module.requires_grad_(True)

    def module_names(self) -> List[str]:
        return [n for n, _ in self.module.named_modules()]

    def parameter_names(self) -> List[str]:
        return [n for n, _ in self.module.named_parameters()]

    def get_module_parameter(self, parameter_name: str, expected_shape: Tuple[int, ...]):
        """Convenience function to retrieve a full weight.

        Similar workflow to activation retrieval/editing in the hook functions
        in that we must potentially gather the weight across ranks beforehand.
        Hence we will use the same utility to decide on the collective, but we
        don't need to use the dispersion function.
        """
        local_param = self.module.get_parameter(parameter_name).detach()
        collect_fn, _ = dist._parse_collect_and_distribute_from_tensor(
            local_param,
            expected_shape,
        )
        full_param = collect_fn(local_param).cpu()
        return full_param

    def forward(self, *args, **kwargs) -> Any:
        """Run forward of wrapped model."""
        with self._hook():
            logger.info("Running forward")
            outputs = self.module(*args, **kwargs)
        return outputs


class DummyModule(nn.Module):
    """Identity module used to expose activations.

    Can be placed in any `nn.Module` to artificially create an activation to
    be hooked onto. For instance, explicitly calling a module's `.forward()`
    method will not run forward hooks and therefore will not generate an
    activation. However, applying this module to the output of that will
    generate an activation which can be hooked onto.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs
