from contextlib import contextmanager
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
from .hook_function import HookFunction


logger = logging.getLogger(__name__)


class FlexModel(nn.Module):
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
        self.output_ptr = output_ptr
        self.hook_functions: Dict[str, HookFunction] = {}
        self._hook_function_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hooks_active: bool = False

    def register_hook_function(
        self,
        hook_function: HookFunction,
    ) -> None:
        """Given user hook reqest, generate hook function and store it."""
        hook_function._output_ptr = self.output_ptr
        self.hook_functions[hook_function.module_name] = hook_function

    def set_hooks(self) -> None:
        if not self._hooks_active:
            for name, module in self.module.named_modules():
                if name in self.hook_functions:
                    self._hook_function_handles[name] = module.register_forward_hook(
                        self.hook_functions[name]
                    )
                    logger.debug(f"Installing module: {name} forward hook")
            self._hooks_active = True

    def remove_hooks(self) -> None:
        if self._hooks_active:
            for hook in self._hook_function_handles.values():
                hook.remove()
            self._hook_function_handles.clear()

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

    def freeze(self):
        """Helper function to disable gradients of wrapped model."""
        self.requires_grad_(False)

    def unfreeze(self):
        """Helper function to enable gradients of wrapped model."""
        self.requires_grad_(True)

    def module_names(self) -> List[str]:
        """Helper function to get a list module names in the wrapped module."""
        return [n for n, _ in self.module.named_modules()]

    def parameter_names(self) -> List[str]:
        """Helper function to get a list param names in the wrapped module."""
        return [n for n, _ in self.module.named_parameters()]

    def get_module_parameter(
        self, parameter_name: str, expected_shape: Tuple[int, ...]
    ):
        """Convenience function to retrieve a full weight.

        Similar workflow to activation retrieval/editing in the hook functions
        in that we must potentially gather the weight across ranks beforehand.
        Hence we will use the same utility to decide on the collective, but we
        don't need to use the dispersion function.
        """
        local_param = self.module.get_parameter(parameter_name).detach()
        collect_fn, _ = dist.parse_collect_and_distribute_from_tensor(
            local_param,
            expected_shape,
        )
        full_param = collect_fn(local_param).cpu()
        return full_param

    def forward(self, *args, **kwargs) -> Any:
        """Run forward of wrapped model."""
        if self._hooks_active:
            outputs = self.module(*args, **kwargs)
        else:
            with self._hook():
                logger.debug("Rank{dist.get_rank()}: Running forward")
                outputs = self.module(*args, **kwargs)

        return outputs


