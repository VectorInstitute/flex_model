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
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
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

        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size
        self.dp_size = data_parallel_size

        # Distributed valid states:
        #   1. Single-gpu no torch distributed
        #   2. Single-gpu with torch distributed
        #   3. Multi-gpu with torch distributed
        if torch.distributed.is_initialized():
            # Initialize the proper distributed backend (ie. torch, accelerate,
            # etc.)
            dist.initialize_distributed_backend(
                torch.distributed.get_world_size(),
                self.tp_size,
                self.pp_size,
                self.dp_size,
            )

            # Initialize the activation parallel distributed process groups
            dist.initialize_activation_parallel()

    def clear_all_state_(self) -> None:
        """Clear all state aside from wrapped module and output pointer."""
        self.hook_functions.clear()
        self._hook_function_handles.clear()
        self._hooks_active = False
        self.save_ctx.clear()
        self.trainable_modules.clear()
        dist.destroy_activation_parallel()
        dist.destroy_distributed_backend()

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
        collect_fn = dist.parse_collect_from_parameter_tensor(
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
                outputs = self.module(*args, **kwargs)

        # Gather from all pp ranks to pp rank0 (ie. head rank0)
        # NOTE: If rank not in pp group, torch dist gets world size for default
        #       group!
        if (
            torch.distributed.is_initialized()
            and dist.in_pipeline_parallel_group()
            and dist.get_activation_pipeline_parallel_world_size() > 1
        ):
            gathered_acts = dist.gather_pipeline_parallel(self.output_ptr)
            if gathered_acts is not None:
                self.output_ptr = {
                    layer_name: act
                    for act_dict in gathered_acts
                    for layer_name, act in act_dict.items()
                }

            # Reset back to empty dict for next forward pass
            else:
                self.output_ptr = {}

        if torch.distributed.is_initialized():
            logger.debug(f"Rank{torch.distributed.get_rank()} Finished forward")
        return outputs
