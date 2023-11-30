import logging
from dataclasses import dataclass, asdict, fields
import pprint
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


_GLOBAL_PARALLEL_STATE = None


@dataclass
class GPUDeviceMesh:
    """Device mesh containing torch distributed groups.

    Contains metadata on torch distributed groups facilitating tensor,
    pipeline and data parallelism. See :code:`build` for details on how these groups
    are constructed.

    :var tp_group_ranks: Tensor parallel group ranks.
    :type tp_group_ranks: List[List[int]]
    :var pp_group_ranks: Pipeline parallel group ranks.
    :type pp_group_ranks: List[List[int]]
    :var dp_group_ranks: Data parallel group ranks.
    :type dp_group_ranks: List[List[int]]
    """

    tp_group_ranks: List[List[int]]
    pp_group_ranks: List[List[int]]
    dp_group_ranks: List[List[int]]

    @classmethod
    def build(
        cls,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
    ):
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

            Where gathering in the dp groups would need to be done
            hierarchically, starting from a gather across tp groups first
            and then the gather across the dp groups. Gathering the pipeline
            parallel groups is done at the end of the forward pass when all
            pipeline parallel ranks have their retrieved activations on CPU.

            Note that initialization is almost the exact same as Megatron-LM's
            `parallel_state`, but we choose to extend PP groups down the entire
            mesh end-to-end.

        :param int world_size: Total number of GPUs.
        :param int tensor_parallel_size: Number of GPUs in each tensor parallel
            group.
        :param int pipeline_parallel_size: Number of GPUs in each pipeline
            parallel group.
        :param int data_parallel_size: Number of GPUs in each data parallel
            group.
        :returns: Instantiated :class:`GPUDeviceMesh` containing mesh of all
            distributed groups.
        """
        world_size = (
            tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        )
        mesh = torch.arange(world_size).reshape(
            pipeline_parallel_size,
            data_parallel_size,
            tensor_parallel_size,
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

        logger.debug(
            f"Building GPUDeviceMesh with group ranks: "
            f"TP: {tp_group_ranks}, "
            f"PP: {pp_group_ranks}, "
            f"DP: {dp_group_ranks}"
        )

        return cls(tp_group_ranks, pp_group_ranks, dp_group_ranks)


def _initialize_local_gpu_device_mesh(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
) -> GPUDeviceMesh:
    device_mesh = GPUDeviceMesh.build(
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    return device_mesh


def _initialize_distributed_groups(
    rank: int,
    device_mesh: GPUDeviceMesh,
) -> None:
    """Initializes the torch distributed groups.

    Constructs the torch distributed groups defined in the device mesh
    and saves the corresponding group handles for the local device.
    """
    # Initialize torch distributed tp groups
    for group_ranks in device_mesh.tp_group_ranks:
        group = dist.new_group(group_ranks)
        if rank in group_ranks:
            tp_group = group
            logger.debug(
                f"Torch rank {dist.get_rank()} init TP group: {group_ranks}"
            )

    # Initialize torch distributed dp groups
    for group_ranks in device_mesh.dp_group_ranks:
        group = dist.new_group(group_ranks)
        if rank in group_ranks:
            dp_group = group
            logger.debug(
                f"Torch rank {dist.get_rank()} init DP group: {group_ranks}"
            )

    # Initialize torch distributed pp groups
    for group_ranks in device_mesh.pp_group_ranks:
        group = dist.new_group(group_ranks)
        if rank in group_ranks:
            pp_group = group
            logger.debug(
                f"Torch rank {dist.get_rank()} init PP group: {group_ranks}"
            )

    return tp_group, pp_group, dp_group


@dataclass
class LocalParallelState:
    local_process_group: dist.ProcessGroup
    local_rank: int
    local_world_view: List[int]
    local_tp_group: dist.ProcessGroup
    local_pp_group: dist.ProcessGroup
    local_dp_group: dist.ProcessGroup
    local_mesh: GPUDeviceMesh

    def __repr__(self):
        _repr = "LocalParallelState(\n"
        for field in fields(self):
            _repr += f"\t{field.name}: {getattr(self, field.name)}\n"
        _repr += ")"
        return _repr


@dataclass
class GlobalParallelState:
    global_world_view: List[int]
    model_to_subset_state_map: Dict[nn.Module, LocalParallelState] = None


def _initialize_global_distributed_state(global_world_view):
    global _GLOBAL_PARALLEL_STATE
    _GLOBAL_PARALLEL_STATE = GlobalParallelState(
        global_world_view=global_world_view,
        model_to_subset_state_map={},
    )

    return


def _dataclass_from_dict(cls, mapping):
    cls_fields = {f.name for f in fields(cls) if f.init}
    arg_fields = {
        name: value for name, value in mapping.items() if name in cls_fields
    }
    return cls(**arg_fields)


def _local_to_subset_device_mesh(
    local_world_view_device_mesh, local_to_subset_view_map
):
    axis_name_to_local_axis_groups = asdict(local_world_view_device_mesh)

    # Convert each axis from local -> subset rank.
    subset_axis_groups = {}
    for axis_name, local_axis_groups in axis_name_to_local_axis_groups.items():
        subset_axis_group = [
            [local_to_subset_view_map[rank] for rank in local_axis_group]
            for local_axis_group in local_axis_groups
        ]
        subset_axis_groups[axis_name] = subset_axis_group

    # Instantiate new device mesh.
    subset_world_view_device_mesh = _dataclass_from_dict(
        GPUDeviceMesh, subset_axis_groups
    )

    return subset_world_view_device_mesh


def _initialize_local_distributed_state(
    process_group,
    subset_rank,
    local_rank,
    subset_world_view,
    local_world_view,
    subset_to_local_view_map,
    local_to_subset_view_map,
    tp_size,
    pp_size,
    dp_size,
):
    # Create device meshes.
    local_world_view_device_mesh = _initialize_local_gpu_device_mesh(
        tp_size, pp_size, dp_size
    )
    subset_world_view_device_mesh = _local_to_subset_device_mesh(
        local_world_view_device_mesh,
        local_to_subset_view_map,
    )

    # Create distributed groups.
    if process_group is None:
        process_group = dist.new_group(
            ranks=subset_world_view,
            backend="nccl",
        )

    (
        local_tp_group,
        local_pp_group,
        local_dp_group,
    ) = _initialize_distributed_groups(
        subset_rank,
        subset_world_view_device_mesh,
    )

    # Create local state.
    local_parallel_state = LocalParallelState(
        process_group,
        local_rank,
        local_world_view,
        local_tp_group,
        local_pp_group,
        local_dp_group,
        local_world_view_device_mesh,
    )

    # Complete new local state entry.
    model_to_state_map_entry = {
        "local_parallel_state": local_parallel_state,
        "subset_rank": subset_rank,
        "subset_world_view": subset_world_view,
        "subset_world_view_device_mesh": subset_world_view_device_mesh,
        "subset_to_local_view_map": subset_to_local_view_map,
        "local_to_subset_view_map": local_to_subset_view_map,
    }

    return model_to_state_map_entry


def _global_state_is_initialized():
    global _GLOBAL_PARALLEL_STATE
    return False if _GLOBAL_PARALLEL_STATE is None else True


def _get_global_state():
    global _GLOBAL_PARALLEL_STATE
    assert _GLOBAL_PARALLEL_STATE is not None
    return _GLOBAL_PARALLEL_STATE


def _set_global_state(model, model_to_state_map_entry):
    gs = _get_global_state()
    gs.model_to_subset_state_map[model] = model_to_state_map_entry


def _destroy_global_state():
    global _GLOBAL_PARALLEL_STATE
    _GLOBAL_PARALLEL_STATE = None
    dist.barrier()


def initialize_distributed_state(
    model: nn.Module,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    process_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Main entry point from :class:`FlexModel` to initialize distributed backend.

    Given tensor, pipeline and data parallel sharding scheme of the wrapped
    :code:`nn.Module`, detect which backend to use and assemble a GPU device mesh
    which facilitates activation communication.

    :param int world_size: Total number of devices used to host the wrapped module.
    :param int tensor_parallel_size: Number of devices in each tensor parallel
        group.
    :param int pipeline_parallel_size: Number of devices in the pipeline parallel
        group.
    :param int data_parallel_size: Number of devices in each data parallel group.

    :raises AssertionError: If the world size is inconsistent with the tp, pp
        and dp sizes.
    """
    assert (
        dist.is_initialized()
    ), "Please call `torch.distributed.init_process_group()`"
    group_world_size = dist.get_world_size(group=process_group)
    req_world_size = (
        tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    )
    assert group_world_size == req_world_size, (
        f"Default process group ({group_world_size} ranks) was provided, "
        f"but parallelism axes result in less/more than requested processes "
        f"({req_world_size} ranks). If you are initializing using a subset of "
        f"the default process group, please create a new group using "
        f"`torch.distributed.new_group(ranks)`"
    )
    # Global world view: [0, 1, 2, 3, 4, 5, 6, 7]
    # Subset world view: [2, 3, 6, 7]
    # Local world view: [0, 1, 2, 3]
    # Global-local map: {2: 0, 3: 1, 6: 2, 7: 3}

    # Construct world views.
    global_world_view = list(range(dist.get_world_size()))

    if process_group is None:  # subset is entire world
        subset_world_view = list(range(dist.get_world_size()))
    else:
        # `dist.get_process_group_ranks` is always wrt default pg.
        subset_world_view = dist.get_process_group_ranks(group=process_group)

    local_world_view = list(range(len(subset_world_view)))

    logger.debug(
        f"Initializing distributed state with world views:\n"
        f"Global: {global_world_view}\n"
        f"Subset: {subset_world_view}\n"
        f"Local : {local_world_view}"
    )

    # Construct mappings between world views.
    subset_to_local_view_map = {
        g_rank: l_rank
        for g_rank, l_rank in zip(subset_world_view, local_world_view)
    }
    local_to_subset_view_map = {
        l_rank: g_rank
        for g_rank, l_rank in zip(subset_world_view, local_world_view)
    }
    logger.debug(
        f"Created mappings between world views: "
        f"[subset -> local]{subset_to_local_view_map} - "
        f"[local -> subset]{local_to_subset_view_map}"
    )

    # Get device rank assignment in each world view.
    subset_rank = dist.get_rank(group=None)
    local_rank = subset_to_local_view_map[subset_rank]
    logger.debug(
        f"Rank assignment: subset[{subset_rank}] - local[{local_rank}]"
    )

    # Initialize global state.
    if not _global_state_is_initialized():
        _initialize_global_distributed_state(global_world_view)
        logger.debug("Initialized global state")

    # Initialize local state and attach it to the global state.
    model_to_state_map_entry = _initialize_local_distributed_state(
        process_group,
        subset_rank,
        local_rank,
        subset_world_view,
        local_world_view,
        subset_to_local_view_map,
        local_to_subset_view_map,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )

    _set_global_state(model, model_to_state_map_entry)

    logger.debug(
        f"Initialized local state:\n"
        f"Model: {model}\n"
        f"Local state:\n"
        f"{pprint.pformat(model_to_state_map_entry, indent=4, sort_dicts=False)}"
    )

    # FlexModel-local Parallel State.
    fmps = _LocalParallelStateAPI(model)

    return fmps


def _finalize_local_parallel_state_api(model_reference):
    global _GLOBAL_PARALLEL_STATE

    # Likely that model is being destructed, remove all references.
    del _GLOBAL_PARALLEL_STATE.model_to_subset_state_map[model_reference]

    # Cleanup global state if there are no more local state contexts.
    if len(_GLOBAL_PARALLEL_STATE.model_to_subset_state_map) == 0:
        _destroy_global_state()

    logger.debug("Local parallel state api finalized")
    dist.barrier()


class _LocalParallelStateAPI:
    def __init__(self, model_reference):
        self.model_reference = model_reference

        global _GLOBAL_PARALLEL_STATE
        self.model_entry = _GLOBAL_PARALLEL_STATE.model_to_subset_state_map[
            self.model_reference
        ]
        self.ls = self.model_entry["local_parallel_state"]

        # TODO: Finalizer runs before gc which may cause global parallel state
        #       to die even if a local parallel state is being created. Should
        #       we register global state instances with hashes? Or could just
        #       not destroy it until program death.
        """
        self._finalizer = weakref.finalize(
            self,
            _finalize_local_parallel_state_api,
            self.model_reference,
        )
        """

    def get_local_process_group(self) -> int:
        return self.ls.local_process_group

    def get_local_rank(self) -> int:
        return dist.get_rank(group=self.get_local_process_group())

    def get_local_world_size(self) -> int:
        return dist.get_world_size(group=self.get_local_process_group())

    def get_subset_rank(self) -> int:
        return dist.get_rank()

    def get_subset_world_size(self) -> int:
        return dist.get_world_size()

    def get_tensor_parallel_group(self) -> dist.ProcessGroup:
        assert self.ls.local_tp_group is not None
        return self.ls.local_tp_group

    def get_pipeline_parallel_group(self) -> dist.ProcessGroup:
        assert self.ls.local_pp_group is not None
        return self.ls.local_pp_group

    def get_data_parallel_group(self) -> dist.ProcessGroup:
        assert self.ls.local_dp_group is not None
        return self.ls.local_dp_group

    def get_tensor_parallel_world_size(self) -> int:
        assert self.ls.local_tp_group is not None
        return dist.get_world_size(self.ls.local_tp_group)

    def get_pipeline_parallel_world_size(self) -> int:
        assert self.ls.local_pp_group is not None
        return dist.get_world_size(self.ls.local_pp_group)

    def get_data_parallel_world_size(self) -> int:
        assert self.ls.local_dp_group is not None
        return dist.get_world_size(self.ls.local_dp_group)

    def get_tensor_parallel_rank(self) -> int:
        assert self.ls.local_tp_group is not None
        return dist.get_rank(self.ls.local_tp_group)

    def get_pipeline_parallel_rank(self) -> int:
        assert self.ls.local_pp_group is not None
        return dist.get_rank(self.ls.local_pp_group)

    def get_data_parallel_rank(self) -> int:
        assert self.ls.local_dp_group is not None
        return dist.get_rank(self.ls.local_dp_group)
