import functools
import logging
import time
import weakref
from argparse import Namespace
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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


def _get_module(root_module: nn.Module, tgt_module_name: str) -> nn.Module:
    submodule_names = tgt_module_name.split(".")
    # Can access submodules by `a.b`, but some modules require `a[b]`.
    return functools.reduce(
        lambda carry, n: getattr(carry, n)
        if not isinstance(carry, (nn.ModuleList, nn.Sequential))
        else carry[int(n)],
        submodule_names,
        root_module,
    )


@dataclass
class _SharedState:
    output_ptr: Dict[str, List[Tensor]]
    save_ctx: Namespace
    modules: nn.ModuleDict
    offload_mode: str


def _finalize_dangling_state(hook_functions) -> None:
    """Clear persistent state when FlexModel is garbage collected."""
    # Remove hook functions from model.
    for group_name, module_to_hf in hook_functions.items():
        for m, hf_to_handle in module_to_hf.items():
            for hf, handle in hf_to_handle.items():
                handle.remove()
    hook_functions.clear()

    # Clear distributed states.
    if dist.distributed_backend_is_initialized():
        if dist.activation_parallel_is_initialized():
            dist.destroy_activation_parallel()
        dist.destroy_distributed_backend()


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
    :var int offload_mode: Selected device which activation tensors are
        offloaded to.

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

        # Register the hook function (same as PyTorch API).
        flex_model.register_forward_hook(my_hook_function)

        # Run forward pass. Output dictionary will become populated.
        outputs = flex_model(inputs)

    """

    def __init__(
        self,
        module: nn.Module,
        output_ptr: Dict[str, List[Tensor]],
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        offload_mode: str = "CPU",
    ):
        """Initialize the instance by wrapping the Pytorch module.

        :param nn.Module module: :code:`nn.Module` to wrap and apply hooks to.
        :param output_ptr: Output dictionary to dump activations to.
        :type output_ptr: Dict[str, List[Tensor]]
        :param int tensor_parallel_size: Number of workers in each tensor
            parallel group.
        :param int pipeline_parallel_size: Number of workers in each pipeline
            parallel group.
        :param int data_parallel_size: Number of processes in each data
            parallel group.
        :param str offload_mode: Device which activation tensors are offloaded
            to. Valid modes are currently "CPU" and "GPU".
        """
        super().__init__()
        self.module = module
        self.hook_functions: Dict[
            nn.Module, Dict[HookFunction, torch.utils.hooks.RemovableHandle]
        ] = {"all": defaultdict(dict)}
        self.output_ptr = output_ptr
        self.save_ctx: Namespace = Namespace()  # dumb Namespace for now
        self.trainable_modules = nn.ModuleDict()
        self.tp_size = tensor_parallel_size
        self.pp_size = pipeline_parallel_size
        self.dp_size = data_parallel_size

        # Initialize FM distributed.
        if torch.distributed.is_initialized():
            world_size = self.tp_size * self.pp_size * self.dp_size
            dist.initialize_distributed_backend(
                world_size, self.tp_size, self.pp_size, self.dp_size,
            )
            dist.initialize_activation_parallel()

        self._offload_modes = {"CPU", "GPU"}
        assert offload_mode in self._offload_modes
        self.offload_mode = offload_mode

        # Create shared state between FM instance and HF instances.
        self._shared_state = _SharedState(
            self.output_ptr, self.save_ctx, self.trainable_modules, self.offload_mode,
        )

        # Initialize mappings.
        self._hook_type_to_pt_attr = {
            "forward": "register_forward_hook",
            "full_backward": "register_full_backward_hook",
            "tensor": "register_hook",
            "forward_pre": "register_forward_pre_hook",
            "backward_pre": "register_full_backward_pre_hook",
        }

        self._finalizer = weakref.finalize(
            self, _finalize_dangling_state, self.hook_functions,
        )

    def restore(self) -> nn.Module:
        """Cleans up dangling states and modifications to wrapped module."""
        self._finalizer()
        return self.module

    @functools.singledispatchmethod
    def _make_group(self, group_constructor):
        raise NotImplementedError(f"Cannot make group using: {group_constructor}")

    @_make_group.register
    def _(self, group_constructor: str):
        # TODO: Pattern match module names for already-registered hook functions.
        raise NotImplementedError

    @_make_group.register
    def _(self, group_constructor: list, group_alias: str):
        # TODO: Construct group explicitly using hook functions.
        raise NotImplementedError

    def _register_hook_impl(self, hook_function: HookFunction) -> None:
        """Registers hook to underlying pytorch module."""
        module = _get_module(self.module, hook_function.module_name)

        # Register hook function to pt module.
        register_fn = getattr(
            module, self._hook_type_to_pt_attr[hook_function._hook_type], None
        )
        handle = register_fn(hook_function)
        self.hook_functions["all"][module][hook_function] = handle

    def _hook_registration_prologue(
        self, hook_function: HookFunction, hook_type: str
    ) -> None:
        """Validate hook function and set state."""
        # Validate hook function.
        if (
            "_fsdp_wrapped_module" in hook_function.module_name
            and hook_type == "tensor"
        ):
            raise NotImplementedError(
                "Pytorch FSDP is currently not supported for parameter/grad "
                "level hooks yet."
            )
        assert (
            hook_type in self._hook_type_to_pt_attr
        ), "Invalid hook type provided: {hook_type}"

        # Set private HF state.
        assert hook_function._shared_state is None
        assert hook_function._hook_type is None
        hook_function._shared_state = self._shared_state
        hook_function._hook_type = hook_type

        # Pass to registration impl.
        self._register_hook_impl(hook_function)

    def register_forward_hook(self, hook_function: HookFunction) -> None:
        """Register a forward hook function.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._hook_registration_prologue(hook_function, "forward")

    def register_full_backward_hook(self, hook_function: HookFunction) -> None:
        """Register a backward hook function.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._hook_registration_prologue(hook_function, "full_backward")

    def register_hook(self, hook_function: HookFunction) -> None:
        """Register a backward hook function on a tensor.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._hook_registration_prologue(hook_function, "tensor")

    def register_forward_pre_hook(self, hook_function: HookFunction) -> None:
        """Register a pre-forward hook function.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._hook_registration_prologue(hook_function, "forward_pre")

    def register_full_backward_pre_hook(self, hook_function: HookFunction) -> None:
        """Register a pre-backward hook function.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._hook_registration_prologue(hook_function, "backward_pre")

    def _enable_group_hooks(self, group: str):
        for m, hook_fn_to_handle in self.hook_functions[group].items():
            for hook_fn, handle in hook_fn_to_handle.items():
                if handle is None:
                    self._register_hook_impl(m, hook_fn)

    def _disable_group_hooks(self, group: str):
        for m, hook_fn_to_handle in self.hook_functions[group].items():
            for hook_fn, handle in hook_fn_to_handle.items():
                if handle is not None:
                    handle.remove()
                    self.hook_functions[group][m][hook_fn] = None

    def _flush_pipeline(self) -> None:
        """Gather tensors from all pipeline stages to the first stage."""
        if (
            torch.distributed.is_initialized()
            and dist.in_pipeline_parallel_group()
            and dist.get_activation_pipeline_parallel_world_size() > 1
        ):
            gathered_acts = dist.gather_pipeline_parallel_tensor_dicts(self.output_ptr)

            # Rank 0 accumulates the activation tensors.
            if dist.get_activation_pipeline_parallel_rank() == 0:
                self.output_ptr.update(gathered_acts)

            # Other ranks reset their collections for the next forward pass.
            else:
                self.output_ptr = {}

    def forward(self, *args, _group: str = "all", **kwargs) -> Any:
        """Run a forward pass of the model with hooks enabled by default."""
        # TODO: Extend for group pattern matching.
        self._enable_group_hooks(_group)

        outputs = self.module(*args, **kwargs)

        # Post-forward cleanup.
        self._flush_pipeline()

        return outputs

    def register_trainable_module(self, name: str, module: nn.Module) -> None:
        """Register trainable module accessible to all :class:`HookFunction` instances.

        Given an :code:`nn.Module`, add it to the :code:`nn.ModuleDict` which is exposed
        to all :class:`HookFunction` runtimes.

        :param str name: Name of the module/layer.
        :param nn.Module module: :code:`nn.Module` to register.
        """
        self.trainable_modules[name] = module

    def get_module_parameter(
        self, parameter_name: str, expected_shape: Tuple[int, ...],
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
        # Need to specially handle cases where parameters/gradients are sharded
        # in complex ways.
        if isinstance(self.module, FSDP):
            raise NotImplementedError(
                "Pytorch FSDP is currently not supported for parameter "
                "retrieval yet."
            )

        local_param = self.module.get_parameter(parameter_name).detach()
        collect_fn = dist.parse_collect_from_parameter_tensor(
            local_param, expected_shape,
        )
        full_param = collect_fn(local_param).cpu()
        return full_param

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
