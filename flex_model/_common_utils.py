from typing import (
    Tuple,
    Callable,
    Dict,
    Optional,
)

import torch
from torch import Tensor

from flex_model._distributed_utils import (
    _set_activation_group,
    _get_activation_parallel_group,
)

class _State:
    pass


class _FlexModelState(_State):
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
        self.module_name: str
        self.expected_shape: Tuple[int, ...]
        self.editing_function: Optional[Callable]
        self._using_torch_distributed: bool
        self._collect_function: Optional[Callable]
        self._distribute_function: Optional[Callable]
        self._output_ptr: Optional[Dict[str, Tensor]]


def _init_distributed_function_state(
    state: _HookFunctionState,
) -> _HookFunctionState:
    if torch.distributed.is_initialized():     
        state._using_torch_distributed = True
    else:
        state._using_torch_distributed = False
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
    state._collect_function = None
    state._distribute_function = None
    state._output_ptr = None
    return state


def _init_distributed_model_state(
    state: _FlexModelState,
) -> _FlexModelState:
    state.compute_device = torch.device(
        "cuda", torch.cuda.current_device(),
    )

    if not torch.distributed.is_initialized():
        state.rank = 0
        state.world_size = 1
        return state

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
