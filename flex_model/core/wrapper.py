import functools
import logging
import weakref
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import flex_model.distributed as dist

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


class _HookFunctionGroupManager:
    def __init__(self):
        self.hook_fn_to_groups_map = defaultdict(set)
        self.groups = set()

    def get_hook_fn_groups(self, hook_fn: HookFunction) -> Set[str]:
        return self.hook_fn_to_groups_map[hook_fn]

    def get_group_hook_fns(self, group_name: str) -> List[HookFunction]:
        hook_fns = []
        for hook_fn, groups in self.hook_fn_to_groups_map.items():
            if group_name in groups:
                hook_fns.append(hook_fn)
        return hook_fns

    def create(
        self,
        group_name: str,
        group_constructor: str,
        expected_shape: Tuple[Optional[int], ...],
        all_names: List[str],
        editing_function: Optional[Callable] = None,
        unpack_idx: int = 0,
    ) -> List[HookFunction]:
        """Instantiate and group hook functions based on module name pattern matching."""
        matching_modules = []
        for module_name in all_names:
            if group_constructor in module_name:
                matching_modules.append(module_name)

        hook_fns = []
        for mod_name in matching_modules:
            hf = HookFunction(
                module_name=mod_name,
                expected_shape=expected_shape,
                editing_function=editing_function,
                unpack_idx=unpack_idx,
            )
            self.hook_fn_to_groups_map[hf].add(group_name)
            hook_fns.append(hf)

        if group_name not in self.groups:
            self.groups.add(group_name)

        return hook_fns

    @functools.singledispatchmethod
    def update(self, group_constructor, group_name) -> None:
        raise NotImplementedError(
            f"Cannot make group using: {type(group_constructor)}"
        )

    @update.register
    def _update_by_list_hook_fns(
        self, group_constructor: list, group_name: str
    ) -> None:
        """Add group tag to a collection of hook functions."""
        are_hook_fns = map(
            lambda x: isinstance(x, HookFunction), group_constructor
        )
        assert all(
            are_hook_fns
        ), "set_group takes a collection of only HookFunctions"

        # Associate the hook functions with this group.
        for hook_fn in group_constructor:
            self.hook_fn_to_groups_map[hook_fn].add(group_name)

        if group_name not in self.groups:
            self.groups.add(group_name)

    @update.register
    def _update_by_hook_fn(
        self, group_constructor: HookFunction, group_name: str
    ) -> None:
        """Add a group tag to a single hook function."""
        self.hook_fn_to_groups_map[group_constructor].add(group_name)

        if group_name not in self.groups:
            self.groups.add(group_name)

    @update.register
    def _update_by_string(
        self, group_constructor: str, group_name: str
    ) -> None:
        """Pattern-match existing hook functions using group constrcutor."""
        hook_functions_to_add = []
        for hook_fn, groups in self.hook_fn_to_groups_map.items():
            if group_constructor in hook_fn.module_name:
                hook_functions_to_add.append(hook_fn)

        # Associate the hook functions with this group.
        for hook_fn in hook_functions_to_add:
            self.hook_fn_to_groups_map[hook_fn].add(group_name)

        if group_name not in self.groups:
            self.groups.add(group_name)

    def _is_group_alive(self, group_name: str) -> bool:
        """A group is alive if it has one or more associated hook functions."""
        return any(
            map(
                lambda gs: group_name in gs,
                list(self.hook_fn_to_groups_map.values()),
            )
        )

    @functools.singledispatchmethod
    def remove(self, group_constructor, group_name) -> None:
        raise NotImplementedError

    @remove.register
    def _remove_by_list_hook_fns(
        self, group_constructor: list, group_name: str
    ) -> None:
        are_hook_fns = map(
            lambda x: isinstance(x, HookFunction), group_constructor
        )
        assert all(
            are_hook_fns
        ), "remove (list) takes a collection of only HookFunctions"
        assert group_name != "all", "Can't remove 'all' group references"

        for hook_fn in group_constructor:
            self.hook_fn_to_groups_map[hook_fn].remove(group_name)

        if not self._is_group_alive(group_name):
            self.groups.remove(group_name)

    @remove.register
    def _remove_by_hook_fn(
        self, group_constructor: HookFunction, group_name: str
    ) -> None:
        assert group_name != "all", "Can't remove 'all' group references"
        self.hook_fn_to_groups_map[group_constructor].remove(group_name)

        if not self._is_group_alive(group_name):
            self.groups.remove(group_name)

    @remove.register
    def _remove_by_string(
        self, group_constructor: str, group_name: str
    ) -> None:
        assert group_name != "all", "Can't remove 'all' group references"
        hook_functions_to_remove = []
        for hook_fn, groups in self.hook_fn_to_groups_map.items():
            if group_constructor in hook_fn.module_name:
                hook_functions_to_remove.append(hook_fn)

        for hook_fn in hook_functions_to_remove:
            self.hook_fn_to_groups_map[hook_fn].remove(group_name)

        if not self._is_group_alive(group_name):
            self.groups.remove(group_name)

    def bisect(
        self, active_group_names: Union[str, List[str]]
    ) -> Set[HookFunction]:
        """Separate all hook functions into active/inactive sets."""
        if isinstance(active_group_names, str):
            active_group_names = [active_group_names]
        active_group_names = set(active_group_names)

        # A hook function is active if its intersection with the requested
        # active groups is nonzero.
        active_hook_fns = set()
        inactive_hook_fns = set()
        for hook_fn, groups in self.hook_fn_to_groups_map.items():
            if not groups.isdisjoint(active_group_names):
                active_hook_fns.add(hook_fn)
            else:
                inactive_hook_fns.add(hook_fn)

        # Sanity check: Hook functions belong to only one group.
        assert active_hook_fns.isdisjoint(inactive_hook_fns)

        return active_hook_fns, inactive_hook_fns


def _finalize_dangling_state(
    hook_functions: Dict[
        Union[nn.Module, Tensor],
        Dict[HookFunction, torch.utils.hooks.RemovableHandle],
    ],
) -> None:
    """Clear persistent state when FlexModel is garbage collected."""
    # Remove hook functions from model.
    for m, hf_to_handle in hook_functions.items():
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
        * Creation of :class:`HookFunction` groups, which may be selectively
            activated during model forward passes.
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
                world_size,
                self.tp_size,
                self.pp_size,
                self.dp_size,
            )
            dist.initialize_activation_parallel()

        self._offload_modes = {"CPU", "GPU"}
        assert offload_mode in self._offload_modes
        self.offload_mode = offload_mode

        # Create shared state between FM instance and HF instances.
        self._shared_state = _SharedState(
            self.output_ptr,
            self.save_ctx,
            self.trainable_modules,
            self.offload_mode,
        )
        self._hook_fn_group_manager = _HookFunctionGroupManager()

        # Initialize mappings.
        self._hook_type_to_pt_attr = {
            "forward": "register_forward_hook",
            "full_backward": "register_full_backward_hook",
            "tensor": "register_hook",
            "forward_pre": "register_forward_pre_hook",
            "backward_pre": "register_full_backward_pre_hook",
        }
        # Map: submodule -> hook functions -> hook handle.
        self._module_to_hook_fns_map: Dict[
            Union[nn.Module, Tensor],
            Dict[HookFunction, torch.utils.hooks.RemovableHandle],
        ] = defaultdict(dict)

        # Setup finalizer for cleanup.
        self._finalizer = weakref.finalize(
            self,
            _finalize_dangling_state,
            self._module_to_hook_fns_map,
        )

    def _enable_hooks(self, active_hooks: Set[HookFunction]) -> None:
        """Sink selected hooks into associated submodules."""
        for module, hook_fn_to_handle in self._module_to_hook_fns_map.items():
            for hook_fn, handle in hook_fn_to_handle.items():
                if handle is None and hook_fn in active_hooks:
                    self._register_hook_impl(module, hook_fn)

    def _disable_hooks(self, hooks: Set[HookFunction]) -> None:
        """Pull selected hooks out of associated submodules."""
        for module, hook_fn_to_handle in self._module_to_hook_fns_map.items():
            for hook_fn, handle in hook_fn_to_handle.items():
                if handle is not None and hook_fn in hooks:
                    handle.remove()
                    self._module_to_hook_fns_map[module][hook_fn] = None

    def _flush_pipeline(self) -> None:
        """Gather tensors from all pipeline stages to the first stage."""
        if (
            torch.distributed.is_initialized()
            and dist.in_pipeline_parallel_group()
            and dist.get_activation_pipeline_parallel_world_size() > 1
        ):
            gathered_acts = dist.gather_pipeline_parallel_tensor_dicts(
                self.output_ptr
            )

            # Rank 0 accumulates the activation tensors.
            if dist.get_activation_pipeline_parallel_rank() == 0:
                self.output_ptr.update(gathered_acts)

            # Other ranks reset their collections for the next forward pass.
            else:
                self.output_ptr = {}

    def forward(
        self,
        *args,
        groups: Union[str, List[str]] = "all",
        complement: bool = False,
        **kwargs,
    ) -> Any:
        """Run a forward pass of the model with all hooks active by default.

        :param groups: `HookFunction` groups to activate during the forward
            pass.
        :type groups: Union[str, List[str]]
        :param bool complement: If `True`, then the `HookFunctions` groups
            passed in the `group` argument are *not* active.
        """
        active, inactive = self._hook_fn_group_manager.bisect(groups)

        if not complement:
            self._enable_hooks(active)
            self._disable_hooks(inactive)
        else:
            self._enable_hooks(inactive)
            self._disable_hooks(active)

        outputs = self.module(*args, **kwargs)

        # Post-forward cleanup.
        self._flush_pipeline()

        return outputs

    def _register_hook_impl(self, hook_function: HookFunction) -> None:
        """Pytorch `nn.Module` hook function registration implementation."""
        # TODO: `get_module` should work with the parameter tensors in the case
        #       of `register_hook(Tensor)`. Needs verification.
        module = _get_module(self.module, hook_function.module_name)

        # Register hook function to pt module.
        register_fn = getattr(
            module, self._hook_type_to_pt_attr[hook_function._hook_type], None
        )
        handle = register_fn(hook_function)

        # All registered hooks are members of the "all" hook function group.
        self._module_to_hook_fns_map[module][hook_function] = handle
        self._hook_fn_group_manager.update(hook_function, "all")

    def _register_hook_prologue(
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

    def get_hook_function_groups(self, hook_function: HookFunction) -> Set[str]:
        """Get a collection of groups that the `hook_function` belongs to.

        :param HookFunction hook_function: `HookFunction` to fetch related
            groups.
        """
        return self._hook_fn_group_manager.get_hook_fn_groups(hook_function)

    def get_group_hook_functions(self, group_name: str) -> List[HookFunction]:
        """Get a collection of `HookFunction`s that belong in the given group.

        :param str group_name: Group to fetch related `HookFunction`s from.
        """
        return self._hook_fn_group_manager.get_group_hook_fns(group_name)

    def update_hook_groups(
        self,
        group_constructor: Union[List[HookFunction], HookFunction, str],
        group_name: str,
    ) -> None:
        """Adds a group reference to a set of `HookFunction`s.

        `group_constructor` can be one of three things:
            1. List of `HookFunction`s to add the group to.
            2. A single `HookFunction` to add the group to.
            3. A string pattern to match against `HookFunction`s `module_name`
                attributes. The matching `HookFunction`s will have the group
                reference added.

        :param group_constructor: See above note.
        :type group_constructor: Union[List[HookFunction], HookFunction, str]
        :param str group_name: Name of the group to add.
        """
        self._hook_fn_group_manager.update(group_constructor, group_name)

    def remove_hook_groups(
        self,
        group_constructor: Union[List[HookFunction], HookFunction, str],
        group_name: str,
    ) -> None:
        """Removes a group reference from a set of `HookFunction`s.

        `group_constructor` can be one of three things:
            1. List of `HookFunction`s to remove the group from.
            2. A single `HookFunction` to remove the group from.
            3. A string pattern to match against `HookFunction`s `module_name`
                attributes. The matching `HookFunction`s will have the group
                reference removed.

        :param group_constructor: See above note.
        :type group_constructor: Union[List[HookFunction], HookFunction, str]
        :param str group_name: Name of the group to remove.
        """
        self._hook_fn_group_manager.remove(group_constructor, group_name)

    def create_hook_group(
        self,
        group_name: str,
        group_constructor: str,
        expected_shape: Tuple[Optional[int], ...],
        editing_function: Optional[Callable] = None,
        unpack_idx: Optional[int] = 0,
        hook_type: str = "forward",
    ) -> None:
        """Create a group of HookFunctions.

        Instantiates a collection of `HookFunction`s according to the provided
        arguments (broadcast). Adds the instantiated `HookFunction`s to a
        group.

        :param str group_name: Group name to assign.
        :param str group_constructor: String pattern to match module/parameter
            names as the `module_name` parameter for creating the
            `HookFunction`s.
        :param expected_shape: Expected shape of the activations.
        :param Callable editing_function: Editing function to apply on each
            `HookFunction`.
        :param int unpack_idx: Index of tensor in module outputs.
        :param str hook_type: Type of pytorch hook to use.
        """
        if hook_type == "tensor":
            all_names = [n for n, _ in self.module.named_parameters()]
        else:
            all_names = ([n for n, _ in self.module.named_modules()],)
        hook_fns = self._hook_fn_group_manager.create(
            group_name,
            group_constructor,
            expected_shape,
            all_names,
            editing_function,
            unpack_idx,
        )

        for hf in hook_fns:
            self._register_hook_prologue(hf, hook_type)

    def register_forward_hook(self, hook_function: HookFunction) -> None:
        """Register a forward hook function.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._register_hook_prologue(hook_function, "forward")

    def register_full_backward_hook(self, hook_function: HookFunction) -> None:
        """Register a backward hook function.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._register_hook_prologue(hook_function, "full_backward")

    def register_hook(self, hook_function: HookFunction) -> None:
        """Register a backward hook function on a tensor.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._register_hook_prologue(hook_function, "tensor")

    def register_forward_pre_hook(self, hook_function: HookFunction) -> None:
        """Register a pre-forward hook function.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._register_hook_prologue(hook_function, "forward_pre")

    def register_full_backward_pre_hook(
        self, hook_function: HookFunction
    ) -> None:
        """Register a pre-backward hook function.

        :param HookFunction hook_function: `HookFunction` instance to register.
        """
        self._register_hook_prologue(hook_function, "backward_pre")

    def register_trainable_module(self, name: str, module: nn.Module) -> None:
        """Register trainable module accessible to all :class:`HookFunction` instances.

        Given an :code:`nn.Module`, add it to the :code:`nn.ModuleDict` which is exposed
        to all :class:`HookFunction` runtimes.

        :param str name: Name of the module/layer.
        :param nn.Module module: :code:`nn.Module` to register.
        """
        self.trainable_modules[name] = module

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
        # Need to specially handle cases where parameters/gradients are sharded
        # in complex ways.
        if isinstance(self.module, FSDP):
            raise NotImplementedError(
                "Pytorch FSDP is currently not supported for parameter "
                "retrieval yet."
            )

        local_param = self.module.get_parameter(parameter_name).detach()
        collect_fn = dist.parse_collect_from_parameter_tensor(
            local_param,
            expected_shape,
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

    def restore(self) -> nn.Module:
        """Cleans up dangling states and modifications to wrapped module."""
        self._finalizer()
        return self.module
