import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


_TENSOR_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None
_PIPELINE_PARALLEL_GROUP = None


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
        world_size: int,
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


def _initialize_gpu_mesh(
    world_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
) -> GPUDeviceMesh:
    device_mesh = GPUDeviceMesh.build(
        world_size,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    return device_mesh


def _initialize_distributed_groups(device_mesh: GPUDeviceMesh) -> None:
    """Initializes the torch distributed groups.

    Constructs the torch distributed groups defined in the device mesh
    and saves the corresponding group handles for the local device.
    """
    rank = dist.get_rank()

    # Initialize torch distributed tp groups
    global _TENSOR_PARALLEL_GROUP
    for group_ranks in device_mesh.tp_group_ranks:
        group = dist.new_group(group_ranks)
        if rank in group_ranks:
            _TENSOR_PARALLEL_GROUP = group
            logger.debug(
                f"Torch rank {dist.get_rank()} init TP group: {group_ranks}"
            )

    # Initialize torch distributed dp groups
    global _DATA_PARALLEL_GROUP
    for group_ranks in device_mesh.dp_group_ranks:
        group = dist.new_group(group_ranks)
        if rank in group_ranks:
            _DATA_PARALLEL_GROUP = group
            logger.debug(
                f"Torch rank {dist.get_rank()} init DP group: {group_ranks}"
            )

    # Initialize torch distributed pp groups
    global _PIPELINE_PARALLEL_GROUP
    for group_ranks in device_mesh.pp_group_ranks:
        group = dist.new_group(group_ranks)
        if rank in group_ranks:
            _PIPELINE_PARALLEL_GROUP = group
            logger.debug(
                f"Torch rank {dist.get_rank()} init PP group: {group_ranks}"
            )


def initialize_distributed_state(
    world_size: int,  # TODO: Not needed.
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
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
        world_size
        == tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    )

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    device_mesh = _initialize_gpu_mesh(
        world_size,
        tensor_parallel_size,
        pipeline_parallel_size,
        data_parallel_size,
    )
    _initialize_distributed_groups(device_mesh)


def distributed_state_is_initialized() -> bool:
    """Check if activation parallel backend has been initialized.

    :returns: True if any tensor, pipeline or data parallel groups have been
        initialized.
    :rtype: bool
    """
    global _TENSOR_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _PIPELINE_PARALLEL_GROUP

    if (
        _TENSOR_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
        or _PIPELINE_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_tensor_parallel_group() -> dist.ProcessGroup:
    """Get the tensor parallel group handle the local device belongs to.

    :returns: The tensor parallel distributed group.
    :rtype: Optional[pt_dist.ProcessGroup]

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.
    """
    global _TENSOR_PARALLEL_GROUP
    assert (
        _TENSOR_PARALLEL_GROUP is not None
    ), "Tensor parallel group is not initialized"
    return _TENSOR_PARALLEL_GROUP


def get_pipeline_parallel_group() -> dist.ProcessGroup:
    """Get the pipeline parallel group handle the local device belongs to.

    :returns: The pipeline parallel distributed group.
    :rtype: pt_dist.ProcessGroup

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.
    """
    global _PIPELINE_PARALLEL_GROUP
    assert (
        _PIPELINE_PARALLEL_GROUP is not None
    ), "Pipeline parallel group is not initialized"
    return _PIPELINE_PARALLEL_GROUP


def get_data_parallel_group() -> dist.ProcessGroup:
    """Get the data parallel group handle the local device belongs to.

    :returns: The data parallel distributed group.
    :rtype: pt_dist.ProcessGroup

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.

    """
    global _DATA_PARALLEL_GROUP
    assert (
        _DATA_PARALLEL_GROUP is not None
    ), "Data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_tensor_parallel_world_size() -> int:
    """Get the number of devices in the tensor parallel group.

    :returns: Tensor parallel world size.
    :rtype: int

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.
    """
    return dist.get_world_size(group=get_tensor_parallel_group())


def get_pipeline_parallel_world_size() -> int:
    """Get the number of devices in the pipeline parallel group.
    :returns: Pipeline parallel world size.
    :rtype: int

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.

    """
    return dist.get_world_size(group=get_pipeline_parallel_group())


def get_data_parallel_world_size() -> int:
    """Get the number of devices in the data parallel group.
    :returns: Data parallel world size.
    :rtype: int

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.

    """
    return dist.get_world_size(group=get_data_parallel_group())


def get_tensor_parallel_rank() -> int:
    """Get the rank of the local device in the tensor parallel group.
    :returns: Tensor parallel rank.
    :rtype: int

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.

    """
    return dist.get_rank(group=get_tensor_parallel_group())


def get_pipeline_parallel_rank() -> int:
    """Get the rank of the local device in the pipeline parallel group.
    :returns: Pipeline parallel rank.
    :rtype: int

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.

    """
    return dist.get_rank(group=get_pipeline_parallel_group())


def get_data_parallel_rank() -> int:
    """Get the rank of the local device in the data parallel group.
    :returns: Data parallel rank.
    :rtype: int

    :raises AssertionError: If the activation parallel backend hasn't been
        initialized yet.

    """
    return dist.get_rank(group=get_data_parallel_group())


def destroy_distributed_state() -> None:
    """Delete all handles to tensor, pipeline and data parallel groups."""
    global _TENSOR_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    global _PIPELINE_PARALLEL_GROUP
    _TENSOR_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP = None
    _PIPELINE_PARALLEL_GROUP = None
