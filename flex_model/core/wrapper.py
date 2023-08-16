from argparse import Namespace
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
        _override_world_size: int = 0,
    ):
        super().__init__()
        self.module = module
        self.hook_functions: Dict[str, HookFunction] = {}
        self._hook_function_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hooks_active: bool = False

        # Globals accessible in hook functions (rank0 only!)
        self.output_ptr = output_ptr
        self.save_ctx: Namespace = Namespace()  # dumb Namespace for now
        self.trainable_modules = nn.ModuleDict()

        # TODO: Make this configurable, especially for TP+PP use cases
        if torch.distributed.is_initialized():
            parallel_size = list(range(torch.distributed.get_world_size()))
            dist.initialize_activation_parallel(parallel_size)

    def clear_all_state_(self) -> None:
        """Clear all state aside from wrapped module and output pointer."""
        self.hook_functions.clear()
        self._hook_function_handles.clear()
        self._hooks_active = False
        self.save_ctx.clear()
        self.trainable_modules.clear()

    def register_hook_function(
        self,
        hook_function: HookFunction,
    ) -> None:
        """Given user hook reqest, generate hook function and store it."""
        hook_function._output_ptr = self.output_ptr
        hook_function.save_ctx = self.save_ctx
        hook_function.modules = self.trainable_modules
        self.hook_functions[hook_function.module_name] = hook_function

    def register_trainable_module(self, name: str, module: nn.Module):
        """Register some trainable layers for access by all hook functions."""
        self.trainable_modules[name] = module

    def wrapped_requires_grad(self, level: bool):
        """Recursive set requires_grad field on wrapped model parameters."""
        self.module.requires_grad_(level)

    def trainable_modules_requires_grad(self, level: bool):
        """Recursive set requires_grad field on trainable layer parameters."""
        self.trainable_modules.requires_grad_(level)

    def set_hooks(self) -> None:
        """Enable hooks forever."""
        if not self._hooks_active:
            for name, module in self.module.named_modules():
                if name in self.hook_functions:
                    self._hook_function_handles[name] = module.register_forward_hook(
                        self.hook_functions[name]
                    )
                    logger.debug(f"Installing module: {name} forward hook")
            self._hooks_active = True

    def remove_hooks(self) -> None:
        """Disable all hooks."""
        if self._hooks_active:
            for hook in self._hook_function_handles.values():
                hook.remove()
            self._hook_function_handles.clear()
            self._hooks_active = False

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
                logger.debug(f"Rank{dist.get_rank()}: Running forward")
                outputs = self.module(*args, **kwargs)

        return outputs
