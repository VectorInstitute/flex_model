import logging
from typing import (
    Tuple,
    Callable,
    Dict,
    Optional,
)

import accelerate
import torch
from torch import Tensor

from flex_model._distributed_utils import (
    _set_activation_group,
    _get_activation_parallel_group,
)

logger = logging.getLogger(__name__)


class _State:
    pass


class _FlexModelState(_State):
    """State object for a `FlexModel`.

    Args:
        output_ptr (Dict[str, Tensor]):
            Object where retrieved activations are bound to during execution of
            the hook function.
        rank (int):
            Worker rank in the activation process group.
        world_size (int):
            World size of the activation process group.
        process_group (Optional[torch.distributed.ProcessGroup]):
            Activation process group.
        compute_device (Optional[torch.device]):
            Local compute device (GPU).
        hook_fns (Dict[str, HookFunction]):
            Collection of hook functions, keyed by hooked module name.
        _hook_fn_handles (Dict[str, torch.utils.hooks.RemovableHandle]):
            Collection of hook function handles which are used to remove the
            hooks when the forward pass has completed.
    """
    def __init__(self) -> None:
        from flex_model.model_wrappers import HookFunction # For typing
        self.output_ptr: Dict[str, Tensor]
        self.rank: int = -1
        self.world_size: int = -1
        self.process_group: Optional[torch.distributed.ProcessGroup] = None
        self.compute_device: Optional[torch.device] = None
        self.hook_fns: Dict[str, HookFunction]
        self._hook_fn_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}


class _HookFunctionState(_State):
    def __init__(self) -> None:
        """State object for a `HookFunction`.

        Args:
            module_name (str):
                Name of the module to be hooked into.
            expected_shape (Tuple[int, ...]):
                Expected shape of the activation tensor.
            editing_function (Optional[Callable]):
                Optional editing function which will be applied to the
                acitvation tensor during execution of the hook function.
            _using_torch_distributed (bool):
                Flag to indicate if using torch distributed.
            _collect_function (Optional[Callable]):
            _distribute_function (Optional[Callable]):
            _output_ptr (Optional[Dict[str, Tensor]]):
                Object where retrieved activations are bound to during
                execution of the hook function.
        """
        self.module_name: str
        self.expected_shape: Tuple[int, ...]
        self.editing_function: Optional[Callable]
        self._using_torch_distributed: bool
        self._using_accelerate_distributed: bool
        self._collect: Optional[Callable]
        self._disperse: Optional[Callable]
        self._edit: Optional[Callable]
        self._dump: Optional[Callable]
        self._output_ptr: Optional[Dict[str, Tensor]]


def _accelerate_distributed_is_initialized():
    ps = accelerate.PartialState()
    if (
        ps.distributed_type == accelerate.DistributedType.MULTI_GPU or
        ps.distributed_type == accelerate.DistributedType.FSDP or
        ps.distributed_type == accelerate.DistributedType.MEGATRON_LM
    ):
        return True
    else:
        return False


def _init_distributed_function_state(
    state: _HookFunctionState,
) -> _HookFunctionState:
    state._using_torch_distributed = True if torch.distributed.is_initialized() else False
    state._using_accelerate_distributed = True if _accelerate_distributed_is_initialized() else False
    return state


def _init_core_function_state(
    state: _HookFunctionState,
    module_name: str,
    expected_shape: Tuple[int, ...],
    editing_function: Optional[Callable] = lambda x: x,
) -> _HookFunctionState:
    state.module_name = module_name
    state.expected_shape = expected_shape
    
    # TODO: Can add editing fn parse util here
    state.editing_function = editing_function
    return state


def _init_runtime_function_state(
    state: _HookFunctionState,
) -> _HookFunctionState:
    state._collect = None
    state._disperse = None
    state._edit = None
    state._dump = None
    state._output_ptr = None
    return state


def _init_distributed_model_state(
    state: _FlexModelState,
) -> _FlexModelState:
    state.compute_device = torch.device(
        "cuda", torch.cuda.current_device(),
    )

    if (
            not torch.distributed.is_initialized() and 
            not _accelerate_distributed_is_initialized()
    ):
        logger.info(("*" * 10) + "NOT USING DISTRIBUTED FEATURES" + ("*" * 10))
        state.rank = 0
        state.world_size = 1
        return state

    logger.info(("*" * 10) + "DISTRIBUTED FEATURES ENABLED" + ("*" * 10))

    state.rank = torch.distributed.get_rank()
    state.world_size = torch.distributed.get_world_size()

    # Default to all workers for activation group
    _set_activation_group(list(range(state.world_size)))

    state.process_group = _get_activation_parallel_group()

    return state


def _init_core_model_state(
    state: _FlexModelState,
    output_ptr: Dict[str, Tensor],
) -> _FlexModelState:
    state.output_ptr = output_ptr

    return state


def _init_runtime_model_state(
    state: _FlexModelState,
) -> _FlexModelState:
    state._hook_fn_handles = {}
    state.hook_fns = {}
    return state
