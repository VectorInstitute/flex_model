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
    Iterator,
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
    """Wraps a Pytorch :code:`nn.Module` to provide an interface for various
    model-surgery techniques. Most importantly, allows registration of
    user-instantiated :class:`HookFunction` classes which perform
    model-surgery.

    :note: Supported features include:

        * Registry, enabling and disabling of :class:`HookFunction` instances.
        * Exposing global states to all :class:`Hookfunction` runtimes.
        * Distributed orchestration of 1-D to 3-D parallelisms.
        * Providing convenience functions for various attributes.

    :note: :code:`output_dict` is populated in-place. So running a subsequent
        forward pass with the same hooks in will delete the previous
        activations.

    :var nn.Module module: The wrapped Pytorch :code:`nn.Module` to hook into.
    :var Dict[str, HookFunction] hook_functions: Collection of :class:`HookFunction`
        instances keyed by the module name to hook into.
    :var Dict[str, Tensor] output_ptr: Pointer to output dictionary provided by
        the user. Activations will be streamed here on the rank0 process only.
        The returned tensors will all be on CPU.
    :var Namespace save_ctx: Context for caching activations or other metadata
        to be accessed later within the same or a later forward pass.
    :var nn.ModuleDict trainable_modules: Collection of named Pytorch
        modules/layers globally accessible to all :class:`HookFunction`
        runtimes. Can be trained using calls to :code:`.backward()`.
    :var int tp_size: Tensor parallel dimension size.
    :var int pp_size: Pipeline parallel dimension size.
    :var int dp_size: Data parallel dimension size.

    :note: Calls to `.backward()` should consider calling :code:`wrapped_module_requires_grad(False)`,
        else the gradient will be generated for the entire wrapped model and
        :code:`trainable_modules`.

    Example:

    .. highlight:: python
    .. code-block:: python

        ## Code block being run by 4 GPUs ##

        # Load model.
        model = MyModel.from_pretrained(...)

        # Distribute model over many workers using fully-sharded data parallel.
        model = FSDP(model)

        # Create output dictionary where activations will stream to.
        output_dict = {}

        # Wrap the model.
        flex_model = FlexModel(
            model,
            output_dict,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=4,
        )

        # Create hook function for post-mlp.
        my_hook_function = HookFunction(
            "my_model.layers.15.mlp",
            expected_shape=(16, 1024, 4096),
            editing_function=None,
        )

        # Register the hook function.
        flex_model.register_hook_function(my_hook_function)

        # Run forward pass. Output dictionary will become populated.
        outputs = flex_model(inputs)

    """
    # TODO: Backward hook refactor
    # TODO: Tests for each function independently

    def __init__(
        self,
        module: nn.Module,
        output_ptr: Dict[str, Tensor],
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
    ):
        """Initialize the instance by wrapping the Pytorch module.

        :param nn.Module module: :code:`nn.Module` to wrap and apply hooks to.
        :param output_ptr: Output dictionary to dump activations to.
        :type output_ptr: Dict[str, Tensor]
        :param int tensor_parallel_size: Number of workers in each tensor
            parallel group.
        :param int pipeline_parallel_size: Number of workers in each pipeline
            parallel group.
        :param int data_parallel_size: Number of processes in each data
            parallel group.
        """
        super().__init__()
        self.module = module
        self.hook_functions: Dict[str, HookFunction] = {}
        self._hook_function_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hooks_active: bool = False
        self.output_ptr = output_ptr
        self.save_ctx: Namespace = Namespace()  # dumb Namespace for now
        self.trainable_modules = nn.ModuleDict()
        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size
        self.dp_size = data_parallel_size

        if torch.distributed.is_initialized():
            world_size = self.tp_size * self.pp_size * self.dp_size
            dist.initialize_distributed_backend(
                world_size,
                self.tp_size,
                self.pp_size,
                self.dp_size,
            )
            dist.initialize_activation_parallel()

    def register_hook_function(self, hook_function: HookFunction) -> None:
        """Register a user-defined `HookFunction` instance.

        Given a :class:`HookFunction` attach any necessary global context such as
        the activation output dictionary. Save it into the :class:`FlexModel`s
        :class:`HookFunction` collection keyed by the module name to hook into.

        :param HookFunction hook_function: User-defined :class:`HookFunction` instance
            to register.
        """
        hook_function._output_ptr = self.output_ptr
        hook_function.save_ctx = self.save_ctx
        hook_function.modules = self.trainable_modules
        self.hook_functions[hook_function.module_name] = hook_function

    def register_trainable_module(self, name: str, module: nn.Module) -> None:
        """Register trainable module accessible to all :class:`HookFunction` instances.

        Given an :code:`nn.Module`, add it to the :code:`nn.ModuleDict` which is exposed
        to all :class:`HookFunction` runtimes.

        :param str name: Name of the module/layer.
        :param nn.Module module: :code:`nn.Module` to register.
        """
        self.trainable_modules[name] = module

    def enable_forward_hooks(self) -> None:
        """Set forward hooks into the wrapped module indefinitely.

        Takes all registered `HookFunction` instances and registers them into
        the wrapped module to run during the model forward pass.

        :raises NameError: Module name associated with a :class:`HookFunction` was not
            found in the wrapped module, hence the :class:`HookFunction` would be
            unused.
        """
        if not self._hooks_active:

            submodules = {n: p for n, p in self.module.named_modules()}
            for module_name, hook_function in self.hook_functions.items():
                submod = submodules.get(module_name, None)

                if submod is None:
                    raise NameError(
                        f"Hook function module name: {module_name} could not "
                        f"be found in the wrapped model."
                    )

                handle = submod.register_forward_hook(
                    hook_function,
                )
                self._hook_function_handles[module_name] = handle
                logger.debug(
                    f"Intalling hook function on module {module_name}"
                )

            self._hooks_active = True

    def disable_forward_hooks(self) -> None:
        """Un-set forward hooks in the wrapped module indefinitely.

        Takes all registered and active forward :class:`HookFunction` handles and
        deletes them.
        """
        if self._hooks_active:

            for hook_function_handle in self._hook_function_handles.values():
                hook_function_handle.remove()
            self._hook_function_handles.clear()

            self._hooks_active = False

    @contextmanager
    def hooks(self) -> Iterator[None]:
        """Context manager for applying forward hooks.

        Enables hooks within the context. When the context is exited, the
        hooks are disabled. State like the :code:`save_ctx` and :code:`trainable_modules`
        is persistent between entrance and exit of this context manager. Hence
        this context manager mainly controls when activations are retrieved
        and/or edited.

        :returns: Yields nothing as it just manages setup and
            teardown of hooks.
        :rtype: Iterator[None]
        """
        self.enable_forward_hooks()
        try:
            yield
        finally:
            self.disable_forward_hooks()

    def get_module_parameter(
        self,
        parameter_name: str,
        expected_shape: Tuple[int, ...],
    ) -> Tensor:
        """Retrieves unsharded parameter from wrapped module.

        Given the name of the wrapped module submodule parameter gather it
        across the relevant process group if necessary and return it to the
        user on CPU.

        :param str parameter_name: Name of the wrapped module submodule parameter to
            retrieve.
        :param expected_shape: Shape of the full parameter tensor. Only the
            dimensions which are sharded need to be provided. Other dimensions
            can be annotated as `None` and will be auto-completed.
        :type expected_shape: Tuple[int, ...]

        :returns: The requested unsharded parameter tensor detached from
            the computation graph and on CPU.
        :rtype: Tensor
        """
        local_param = self.module.get_parameter(parameter_name).detach()
        collect_fn = dist.parse_collect_from_parameter_tensor(
            local_param,
            expected_shape,
        )
        full_param = collect_fn(local_param).cpu()
        return full_param

    def _gather_pipeline_parallel(self) -> None:
        """Gathers output dicts across the pipeline parallel workers to global
        rank0.

        :note: If :code:`pipeline_parallel_size == 1`, then this function
            is a no-op.
        """
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

    def _forward_with_hooks(self, *args, **kwargs) -> Any:
        """Run the forward pass with hook functions enabled.

        If hooks are active, then we just run the forward pass. If the hooks
        are disabled, then we briefly enable them with the context manager
        which disables them again automatically after the forward pass has
        completed.

        :returns: The output of the wrapped model.
        :rtype: Any
        """
        if self._hooks_active:
            outputs = self.module(*args, **kwargs)
        else:
            with self.hooks():
                outputs = self.module(*args, **kwargs)

        return outputs

    def _forward_no_hooks(self, *args, **kwargs) -> Any:
        """Run forward pass with hook functions disabled.

        If hooks are active, then we briefly disable them before running the
        forward pass, then enable them again after the forward pass is
        completed. If the hooks are disabled, then we simply run the forward
        pass.

        :returns: The output of the wrapped model.
        :rtype: Any
        """
        if self._hooks_active:
            self.disable_forward_hooks()
            outputs = self.module(*args, **kwargs)
            self.enable_forward_hooks()
        
        else:
            outputs = self.module(*args, **kwargs)
        return outputs
            

    def forward(self, *args, with_hooks: bool = True, **kwargs) -> Any:
        """Run forward pass of the wrapped module with arbitrary arguments.

        Primary entrypoint where activations are generated and potentially
        retrieved and/or edited.

        :param bool with_hooks: Boolean flag which can temporarily disable
            hooks. The default behaviour is to always run with hooks.

        :returns: Output of the wrapped module.
        :rtype: Any
        """
        if with_hooks:
            outputs = self._forward_with_hooks(*args, **kwargs)
        else:
            outputs = self._forward_no_hooks(*args, **kwargs)

        self._gather_pipeline_parallel()

        return outputs

    def wrapped_module_requires_grad(self, requires_grad: bool) -> None:
        """Recursively enable/disable gradient on wrapped module submodules.

        Sets the :code:`requires_grad` field recursively on all submodules of the
        wrapped model.

        :param bool requires_grad: True or False value for gradient tensor
            calculation.
        """
        self.module.requires_grad_(requires_grad)

    def trainable_modules_requires_grad(self, requires_grad: bool) -> None:
        """Recursively enable/disable gradient on trainable modules.

        Sets the :code:`reqires_grad` field on all modules in the main
        `nn.ModuleDict` collection.

        :param bool requires_grad: True or False value for gradient tensor
            calculation.
        """
        self.trainable_modules.requires_grad_(requires_grad)

    def all_modules_requires_grad(self, requires_grad: bool) -> None:
        """Recursively enable/disable gradient computation on all submodules.

        This is basically a combination of :code:`wrapped_requires_grad` and
        :code:`trainable_modules_requires_grad`.

        :param bool requires_grad: True or False value for gradient tensor
            calculation.
        """
        self.requires_grad_(requires_grad)

    @property
    def wrapped_module_names(self) -> List[str]:
        """Names of wrapped module submodules.

        :returns: List of module names.
        :rtype: List[str]
        """
        return [n for n, _ in self.module.named_modules()]

    @property
    def trainable_modules_names(self) -> List[str]:
        """Names of trainable modules.

        :returns: List of module names.
        :rtype: List[str]
        """
        return [n for n in self.trainable_modules.keys()]

    @property
    def all_modules_names(self) -> List[str]:
        """Names of all submodules.

        :returns: List of module names.
        :rtype: List[str]
        """
        return self.wrapped_module_names + self.trainable_module_names

    def _clear_all_state_(self) -> None:
        """Destroy all `HookFunction` and distributed state.

        Deletes `HookFunction` states, caching contexts, trainable modules
        and distributed backend/groups.
        """
        self.hook_functions.clear()
        self._hook_function_handles.clear()
        self._hooks_active = False
        self.save_ctx = Namespace()
        self.trainable_modules.clear()
        
        if torch.distributed.is_initialized():
            dist.destroy_activation_parallel()
            dist.destroy_distributed_backend()

    def destroy(self) -> nn.Module:
        """Destroys `FlexModel` state and returns the wrapped module untouched.

        Explicit destruction of `FlexModel` state while leaving the wrapped
        model invariant.

        :returns: The original wrapped module.
        :rtype: nn.Module
        """
        self._clear_all_state_()
        return self.module
