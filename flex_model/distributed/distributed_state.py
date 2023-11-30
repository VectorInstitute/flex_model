import logging
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict
import weakref

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


_GLOBAL_PARALLEL_STATE = None


@dataclass
class _BaseGPUDeviceMesh:
    """Device mesh containing torch distributed groups.

    Contains metadata on torch distributed groups facilitating tensor,
    pipeline and data parallelism.

    :var tp_group_ranks: Tensor parallel group ranks.
    :type tp_group_ranks: List[List[int]]
    :var pp_group_ranks: Pipeline parallel group ranks.
    :type pp_group_ranks: List[List[int]]
    :var dp_group_ranks: Data parallel group ranks.
    :type dp_group_ranks: List[List[int]]
    """

    tp_size: int
    pp_size: int
    dp_size: int
    tp_group_ranks: List[List[int]] = field(init=False)
    pp_group_ranks: List[List[int]] = field(init=False)
    dp_group_ranks: List[List[int]] = field(init=False)

    def _construct_local_groups(self):
        """Create device mesh for GPU communication.

        Example: Given 16 GPUS, where (tp, pp, dp) = (4, 2, 2):
            tp groups -> [gpu0, gpu1, gpu2, gpu3],
                         [gpu4, gpu5, gpu6, gpu7],
                         [gpu8, gpu9, gpu10, gpu11],
                         [gpu12, gpu13, gpu14, gpu15],

            pp groups -> [gpu0, gpu8],
                         [gpu1, gpu9],
                         [gpu2, gpu10],
                         [gpu3, gpu11],
                         [gpu4, gpu12],
                         [gpu5, gpu13],
                         [gpu6, gpu14],
                         [gpu7, gpu15],

            dp groups -> [gpu0, gpu4],
                         [gpu1, gpu5],
                         [gpu2, gpu6],
                         [gpu3, gpu7],
                         [gpu8, gpu12],
                         [gpu9, gpu13],
                         [gpu10, gpu14],
                         [gpu11, gpu15],

        :param int tensor_parallel_size: Number of GPUs in each tensor parallel
            group.
        :param int pipeline_parallel_size: Number of GPUs in each pipeline
            parallel group.
        :param int data_parallel_size: Number of GPUs in each data parallel
            group.
        """
        world_size = self.tp_size * self.pp_size * self.dp_size
        mesh = torch.arange(world_size).reshape(
            self.pp_size, self.dp_size, self.tp_size
        )

        tp_group_ranks: List[List[int]] = []
        for i in range(len(mesh)):
            for j in range(len(mesh[0])):
                tp_group_ranks.append(mesh[i, j, :].tolist())

        dp_group_ranks: List[List[int]] = []
        for i in range(len(mesh)):
            for k in range(len(mesh[0][0])):
                dp_group_ranks.append(mesh[i, :, k].tolist())

        pp_group_ranks: List[List[int]] = []
        for j in range(len(mesh[0])):
            for k in range(len(mesh[0][0])):
                pp_group_ranks.append(mesh[:, j, k].tolist())

        return tp_group_ranks, dp_group_ranks, pp_group_ranks

    def __post_init__(self):
        (
            self.tp_group_ranks,
            self.dp_group_ranks,
            self.pp_group_ranks,
        ) = self._construct_local_groups()

    def __repr__(self):
        _repr = "DeviceMesh(\n"
        for _field in fields(self):
            _repr += f"\t{_field.name}: {getattr(self, _field.name)}\n"
        _repr += ")"
        return _repr


@dataclass
class _LocalGPUDeviceMesh(_BaseGPUDeviceMesh):
    def view_subset(self, local_to_subset_view_map: Dict[int, int]):
        # Convert each axis from local -> subset rank.
        def _local_to_subset_view(local_axis_groups):
            subset_axis_groups = [
                [local_to_subset_view_map[rank] for rank in local_axis_group]
                for local_axis_group in local_axis_groups
            ]
            return subset_axis_groups

        subset_mesh = _SubsetGPUDeviceMesh(
            self.tp_size, self.pp_size, self.dp_size
        )
        subset_mesh.tp_group_ranks = _local_to_subset_view(self.tp_group_ranks)
        subset_mesh.pp_group_ranks = _local_to_subset_view(self.pp_group_ranks)
        subset_mesh.dp_group_ranks = _local_to_subset_view(self.dp_group_ranks)

        return subset_mesh


@dataclass
class _SubsetGPUDeviceMesh(_BaseGPUDeviceMesh):
    def view_local(self, subset_to_local_view_map: Dict[int, int]):
        # Convert each axis from subset -> local rank.
        def _subset_to_local_view(subset_axis_groups):
            local_axis_groups = [
                [subset_to_local_view_map[rank] for rank in subset_axis_group]
                for subset_axis_group in subset_axis_groups
            ]
            return local_axis_groups

        local_mesh = _LocalGPUDeviceMesh(
            self.tp_size, self.pp_size, self.dp_size
        )
        local_mesh.tp_group_ranks = _subset_to_local_view(self.tp_group_ranks)
        local_mesh.pp_group_ranks = _subset_to_local_view(self.pp_group_ranks)
        local_mesh.dp_group_ranks = _subset_to_local_view(self.dp_group_ranks)

        return local_mesh


@dataclass
class _BaseParallelState:
    """Provides helper methods for child classes."""

    def __repr__(self):
        _repr = "ParallelState(\n"
        for _field in fields(self):
            _repr += f"\t{_field.name}: {getattr(self, _field.name)}\n"
        _repr += ")"
        return _repr


@dataclass
class _LocalParallelState(_BaseParallelState):
    """Contains information about local parallel state."""

    local_rank: int
    local_world_view: List[int]
    local_world_view_device_mesh: _LocalGPUDeviceMesh


@dataclass
class _SubsetParallelState(_BaseParallelState):
    """Contains information about subset parallel state.

    Subset parallel state refers to a subset of the global distributed state,
    ie. the entire world of devices. Each `FlexModel` instance uses an
    associated subset of the entire world.
    """

    # Subset state variables.
    subset_process_group: dist.ProcessGroup
    subset_rank: int
    subset_world_view: List[int]
    subset_world_view_device_mesh: _SubsetGPUDeviceMesh

    # Symmetric mappings between local and subset world views.
    subset_to_local_view_map: Dict[int, int]
    local_to_subset_view_map: Dict[int, int]

    # Associated local parallel state.
    local_parallel_state: _LocalParallelState

    # Torch distributed process groups for axes.
    subset_tp_group: dist.ProcessGroup = field(
        init=False
    )  # Torch PGs use subset ranks.
    subset_pp_group: dist.ProcessGroup = field(init=False)
    subset_dp_group: dist.ProcessGroup = field(init=False)
    subset_tp_group_ranks: List[int] = field(init=False)
    subset_pp_group_ranks: List[int] = field(init=False)
    subset_dp_group_ranks: List[int] = field(init=False)

    def _concretize_process_groups(self, axis_group_ranks: List[List[int]]):
        # All processes must participate in `dist.new_group()`.
        subset_tp_group = None
        subset_tp_group_ranks = None
        for group_ranks in axis_group_ranks:
            group = dist.new_group(group_ranks)

            if self.subset_rank in group_ranks:
                subset_tp_group = group
                subset_tp_group_ranks = group_ranks

        return subset_tp_group, subset_tp_group_ranks

    def __post_init__(self):
        (
            self.subset_tp_group,
            self.subset_tp_group_ranks,
        ) = self._concretize_process_groups(
            self.subset_world_view_device_mesh.tp_group_ranks
        )
        (
            self.subset_pp_group,
            self.subset_pp_group_ranks,
        ) = self._concretize_process_groups(
            self.subset_world_view_device_mesh.pp_group_ranks
        )
        (
            self.subset_dp_group,
            self.subset_dp_group_ranks,
        ) = self._concretize_process_groups(
            self.subset_world_view_device_mesh.dp_group_ranks
        )


@dataclass
class _GlobalParallelState(_BaseParallelState):
    global_world_view: List[int]
    module_to_subset_state_map: weakref.WeakKeyDictionary = field(init=False)

    def __post_init__(self):
        self.module_to_subset_state_map = weakref.WeakKeyDictionary()

    def __getitem__(self, key: nn.Module):
        return self.module_to_subset_state_map[key]

    def __setitem__(self, key: nn.Module, value: _SubsetParallelState):
        self.module_to_subset_state_map[key] = value


def _validate_distributed_state_params(
    tp_size: int,
    pp_size: int,
    dp_size: int,
    process_group: Optional[dist.ProcessGroup],
):
    group_world_size = dist.get_world_size(group=process_group)
    req_world_size = tp_size * pp_size * dp_size
    assert group_world_size == req_world_size, (
        f"Default process group ({group_world_size} ranks) was provided, "
        f"but parallelism axes result in less/more than requested processes "
        f"({req_world_size} ranks). If you are initializing using a subset of "
        f"the default process group, please create a new group using "
        f"`torch.distributed.new_group(ranks)`"
    )


def _construct_distributed_world_views(
    process_group: dist.ProcessGroup, tp_size: int, pp_size: int, dp_size: int
):
    global_world_view = list(range(dist.get_world_size()))

    if process_group is None:
        subset_world_view = list(range(dist.get_world_size()))
    else:
        subset_world_view = dist.get_process_group_ranks(group=process_group)

    local_world_view = list(range(len(subset_world_view)))

    return global_world_view, subset_world_view, local_world_view


def _construct_world_view_mappings(
    subset_world_view: List[int], local_world_view: List[int]
):
    subset_to_local_view_map = {}
    local_to_subset_view_map = {}
    for subset_rank, local_rank in zip(subset_world_view, local_world_view):
        subset_to_local_view_map[subset_rank] = local_rank
        local_to_subset_view_map[local_rank] = subset_rank

    return subset_to_local_view_map, local_to_subset_view_map


def _construct_world_view_device_ranks(
    subset_to_local_view_map: Dict[int, int]
):
    subset_rank = dist.get_rank()
    local_rank = subset_to_local_view_map[subset_rank]

    return subset_rank, local_rank


def _initialize_subset_torch_distributed(subset_world_view: List[int]):
    return dist.new_group(
        ranks=subset_world_view,
        backend="nccl",
    )


def _initialize_local_distributed_state(
    local_rank: int,
    local_world_view: List[int],
    tp_size: int,
    pp_size: int,
    dp_size: int,
):
    local_world_view_device_mesh = _LocalGPUDeviceMesh(
        tp_size, pp_size, dp_size
    )

    return _LocalParallelState(
        local_rank, local_world_view, local_world_view_device_mesh
    )


def _initialize_subset_distributed_state(
    subset_rank: int,
    subset_world_view: List[int],
    local_world_view_device_mesh: _LocalGPUDeviceMesh,
    subset_to_local_view_map: Dict[int, int],
    local_to_subset_view_map: Dict[int, int],
    local_parallel_state: _LocalParallelState,
    subset_process_group: Optional[dist.ProcessGroup] = None,
):
    subset_world_view_device_mesh = local_world_view_device_mesh.view_subset(
        local_to_subset_view_map
    )

    if subset_process_group is None:
        subset_process_group = _initialize_subset_torch_distributed(
            subset_world_view
        )

    return _SubsetParallelState(
        subset_process_group,
        subset_rank,
        subset_world_view,
        subset_world_view_device_mesh,
        subset_to_local_view_map,
        local_to_subset_view_map,
        local_parallel_state,
    )


def _initialize_global_state(global_world_view: List[int]):
    global _GLOBAL_PARALLEL_STATE

    _GLOBAL_PARALLEL_STATE = _GlobalParallelState(global_world_view)


def _global_state_is_initialized():
    global _GLOBAL_PARALLEL_STATE
    return False if _GLOBAL_PARALLEL_STATE is None else True


def _get_global_state():
    global _GLOBAL_PARALLEL_STATE
    assert _GLOBAL_PARALLEL_STATE is not None
    return _GLOBAL_PARALLEL_STATE


def _set_global_state(
    module: nn.Module, subset_parallel_state: _SubsetParallelState
):
    assert _global_state_is_initialized()
    gs = _get_global_state()
    gs[module] = subset_parallel_state


def _destroy_global_state():
    global _GLOBAL_PARALLEL_STATE
    _GLOBAL_PARALLEL_STATE = None


class _ParallelStateAPI:
    def __init__(self, model_reference):
        self.model_reference = model_reference

        self.sps = _get_global_state()[self.model_reference]
        self.lps = self.sps.local_parallel_state

    def get_local_rank(self) -> int:
        return self.lps.local_rank

    def get_local_world_size(self) -> int:
        return len(self.lps.local_world_view)

    def get_subset_process_group(self) -> int:
        return self.sps.subset_process_group

    def get_subset_rank(self) -> int:
        return self.sps.subset_rank

    def get_subset_world_size(self) -> int:
        return len(self.sps.subset_world_view)

    def get_tensor_parallel_group(self) -> dist.ProcessGroup:
        return self.sps.subset_tp_group

    def get_pipeline_parallel_group(self) -> dist.ProcessGroup:
        return self.sps.subset_pp_group

    def get_data_parallel_group(self) -> dist.ProcessGroup:
        return self.sps.subset_dp_group

    def get_tensor_parallel_world_size(self) -> int:
        return len(self.sps.subset_tp_group_ranks)

    def get_pipeline_parallel_world_size(self) -> int:
        return len(self.sps.subset_pp_group_ranks)

    def get_data_parallel_world_size(self) -> int:
        return len(self.sps.subset_dp_group_ranks)

    def get_tensor_parallel_rank(self) -> int:
        return dist.get_rank(self.sps.subset_tp_group)

    def get_pipeline_parallel_rank(self) -> int:
        return dist.get_rank(self.sps.subset_pp_group)

    def get_data_parallel_rank(self) -> int:
        return dist.get_rank(self.sps.subset_dp_group)


def _log_debug(msg: str):
    rank = dist.get_rank()

    prefix = f"[GPU{rank}]: "

    logger.debug(prefix + msg)


def initialize_distributed_state(
    module: nn.Module,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    process_group: Optional[dist.ProcessGroup] = None,
):
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    _validate_distributed_state_params(
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
        process_group,
    )
    _log_debug("Finished validating distributed state parameters")

    (
        global_world_view,
        subset_world_view,
        local_world_view,
    ) = _construct_distributed_world_views(
        process_group,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    _log_debug(
        f"World [G] - {global_world_view} "
        f"world [S] - {subset_world_view} "
        f"world [L] - {local_world_view}"
    )

    (
        subset_to_local_view_map,
        local_to_subset_view_map,
    ) = _construct_world_view_mappings(subset_world_view, local_world_view)
    _log_debug(f"Built map [S]->[L] {subset_to_local_view_map}")
    _log_debug(f"Built map [L]->[S] {local_to_subset_view_map}")

    subset_rank, local_rank = _construct_world_view_device_ranks(
        subset_to_local_view_map
    )
    _log_debug(f"Assigned rank [S] = {subset_rank}")
    _log_debug(f"Assigned rank [L] = {local_rank}")

    local_parallel_state = _initialize_local_distributed_state(
        local_rank,
        local_world_view,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    _log_debug(f"[LPS] = {local_parallel_state}")

    subset_parallel_state = _initialize_subset_distributed_state(
        subset_rank,
        subset_world_view,
        local_parallel_state.local_world_view_device_mesh,
        subset_to_local_view_map,
        local_to_subset_view_map,
        local_parallel_state,
        process_group,
    )
    _log_debug(f"[SPS] = {subset_parallel_state}")

    if not _global_state_is_initialized():
        _initialize_global_state(global_world_view)
    _log_debug("[GPS] Initialized")

    _set_global_state(module, subset_parallel_state)
    _log_debug("[GPS] <- [SPS]")

    fmps = _ParallelStateAPI(module)
    _log_debug("Parallel state API handle created")

    return fmps
