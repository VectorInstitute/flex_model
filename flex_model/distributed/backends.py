from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import List, Optional

import torch
import torch.distributed as pt_dist
from accelerate import PartialState


logger = logging.getLogger(__name__)


@dataclass
class GPUDeviceMesh:
    """Device mesh containing torch distributed groups.

    Contains metadata on torch distributed groups facilitating tensor,
    pipeline and data parallelism. See `build` for details on how these groups
    are constructed.
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

        For example, given 16 GPUS, where (tp, pp, dp) = (4, 2, 2):
            tp groups -> [gpu0, gpu1, gpu2, gpu3],
                         [gpu4, gpu5, gpu6, gpu7],
                         [gpu8, gpu9, gpu10, gpu11],
                         [gpu12, gpu13, gpu14, gpu15],

            pp groups -> [gpu0, gpu8]

            dp groups -> [gpu0, gpu4],
                         [gpu8, gpu12],

            Where gathering in the dp groups would need to be done
            hierarchically, starting from a gather across tp groups first
            and then the gather across the dp groups. Gathering the pipeline
            parallel groups is done at the end of the forward pass when all
            pipeline parallel ranks have their retrieved activations on CPU.
        """
        if world_size == 1:
            return cls([[0]], [[0]], [[0]])

        num_tp_groups = world_size // tensor_parallel_size
        num_pp_groups = 1
        num_dp_groups = pipeline_parallel_size

        mesh = torch.arange(world_size).reshape(
            pipeline_parallel_size,
            data_parallel_size,
            tensor_parallel_size,
        )
        # Build tensor parallel groups
        tp_group_ranks: List[List[int]] = []
        for i in range(len(mesh)):
            for j in range(len(mesh[0])):
                tp_group_ranks.append(mesh[i, j, :].tolist())

        # Build data parallel groups
        dp_group_ranks: List[List[int]] = []
        for i in range(len(mesh)):
            dp_group_ranks.append(mesh[i, :, 0].tolist())

        # Build pipeline parallel groups
        pp_group_ranks: List[List[int]] = []
        pp_group_ranks.append(mesh[:, 0, 0].tolist())

        logger.debug(
            f"Building GPUDeviceMesh with group ranks: "
            f"TP: {tp_group_ranks}, "
            f"PP: {pp_group_ranks}, "
            f"DP: {dp_group_ranks}"
        )

        return cls(tp_group_ranks, pp_group_ranks, dp_group_ranks)


class DistributedBackend(ABC):
    """Basic interface for distributed backends.

    Interface exposing core functions for distributed communication. The
    distributed backends must implement tools for organizing models into a
    3D (tensor, pipeline and parallel) mesh.
    """
    @abstractmethod
    def initialize_activation_parallel(self) -> None:
        ...

    @abstractmethod
    def activation_parallel_is_initialized(self) -> bool:
        ...

    @abstractmethod
    def in_tensor_parallel_group(self) -> bool:
        ...

    @abstractmethod
    def in_pipeline_parallel_group(self) -> bool:
        ...

    @abstractmethod
    def in_data_parallel_group(self) -> bool:
        ...

    @abstractmethod
    def get_activation_tensor_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        ...

    @abstractmethod
    def get_activation_data_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        ...

    @abstractmethod
    def get_activation_pipeline_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        ...

    @abstractmethod
    def get_activation_tensor_parallel_world_size(self) -> int:
        ...

    @abstractmethod
    def get_activation_data_parallel_world_size(self) -> int:
        ...

    @abstractmethod
    def get_activation_pipeline_parallel_world_size(self) -> int:
        ...

    @abstractmethod
    def get_activation_tensor_parallel_rank(self) -> int:
        ...

    @abstractmethod
    def get_activation_data_parallel_rank(self) -> int:
        ...

    @abstractmethod
    def get_activation_pipeline_parallel_rank(self) -> int:
        ...

    @abstractmethod
    def destroy_activation_parallel(self) -> None:
        ...


class TorchDistributedBackend(DistributedBackend):
    """Distributed backend using for Pytorch distributed.

    See parent class `DistributedBackend` for details.
    """
    def __init__(self, device_mesh: GPUDeviceMesh):
        """Instantiates the torch distributed backend.

        all_tp_groups: Tensor parallel group handles for all tensor parallel
            groups.
        all_pp_groups: Pipeline parallel group handles for all pipeline parallel
            groups
        all_tp_groups: Data parallel group handles for all data parallel
            groups
        tp_group: Local device tensor parallel group.
        pp_group: Local device pipeline parallel group.
        dp_group: Local device data parallel group.
        """
        self.device_mesh = device_mesh
        self.all_tp_groups: Optional[List[List[int]]] = None
        self.all_pp_groups: Optional[List[List[int]]] = None
        self.all_dp_groups: Optional[List[List[int]]] = None
        self.tp_group: Optional[pt_dist.ProcessGroup] = None
        self.pp_group: Optional[pt_dist.ProcessGroup] = None
        self.dp_group: Optional[pt_dist.ProcessGroup] = None

    def initialize_activation_parallel(self) -> None:
        """Initializes the torch distributed groups.

        Constructs the torch distributed groups defined in the device mesh
        and saves the corresponding group handles for the local device.
        """
        assert self.tp_group is None and self.dp_group is None
        rank = pt_dist.get_rank()

        # Construct tp groups
        self.all_tp_groups = []
        for group_ranks in self.device_mesh.tp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.tp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init TP group: {group_ranks}"
                )
            self.all_tp_groups.append(group_ranks)

        # Construct dp groups
        self.all_dp_groups = []
        for group_ranks in self.device_mesh.dp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.dp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init DP group: {group_ranks}"
                )
            self.all_dp_groups.append(group_ranks)

        # Construct pp groups
        self.all_pp_groups = []
        for group_ranks in self.device_mesh.pp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.pp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init PP group: {group_ranks}"
                )
            self.all_pp_groups.append(group_ranks)

    def activation_parallel_is_initialized(self) -> bool:
        """Check if activation parallel backend has been initialized.

        Returns:
            True if any tensor, pipeline or data parallel groups have been
            initialized.
        """
        # DP groups only active on group rank0 TP workers, so this is
        # intersection not union
        if self.tp_group is None and self.dp_group is None and self.pp_group is None:
            return False
        return True

    def in_tensor_parallel_group(self) -> bool:
        """Check if local device is in a tensor parallel group."""
        return self.tp_group is not None

    def in_pipeline_parallel_group(self) -> bool:
        """Check if local device is in a pipeline parallel group."""
        return self.pp_group is not None

    def in_data_parallel_group(self) -> bool:
        """Check if local device is in a data parallel group."""
        return self.dp_group is not None

    def get_activation_tensor_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the tensor parallel group handle the local device belongs to."""
        assert self.activation_parallel_is_initialized()
        return self.tp_group

    def get_activation_data_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the pipeline parallel group handle the local device belongs to."""
        assert self.activation_parallel_is_initialized()
        return self.dp_group

    def get_activation_pipeline_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the data parallel group handle the local device belongs to."""
        assert self.activation_parallel_is_initialized()
        return self.pp_group

    def get_activation_tensor_parallel_world_size(self) -> int:
        """Get the number of devices in the tensor parallel group."""
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.tp_group)

    def get_activation_data_parallel_world_size(self) -> int:
        """Get the number of devices in the pipeline parallel group."""
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.dp_group)

    def get_activation_pipeline_parallel_world_size(self) -> int:
        """Get the number of devices in the data parallel group."""
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.pp_group)

    def get_activation_tensor_parallel_rank(self) -> int:
        """Get the rank of the local device in the tensor parallel group."""
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.tp_group)

    def get_activation_data_parallel_rank(self) -> int:
        """Get the rank of the local device in the pipeline parallel group."""
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.dp_group)

    def get_activation_pipeline_parallel_rank(self) -> int:
        """Get the rank of the local device in the data parallel group."""
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.pp_group)

    def destroy_activation_parallel(self) -> None:
        """Delete all handles to tensor, pipeline and data parallel groups."""
        self.all_tp_groups = None
        self.all_pp_groups = None
        self.all_dp_groups = None
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None


class AccelerateDistributedBackend(DistributedBackend):
    def __init__(self, device_mesh: GPUDeviceMesh):
        self.device_mesh = device_mesh
        self.all_tp_groups: Optional[List[List[int]]] = None
        self.all_pp_groups: Optional[List[List[int]]] = None
        self.all_dp_groups: Optional[List[List[int]]] = None
        self.tp_group: Optional[pt_dist.ProcessGroup] = None
        self.pp_group: Optional[pt_dist.ProcessGroup] = None
        self.dp_group: Optional[pt_dist.ProcessGroup] = None

    def initialize_activation_parallel(self) -> None:
        assert self.tp_group is None and self.dp_group is None
        rank = pt_dist.get_rank()

        # Construct tp groups
        self.all_tp_groups = []
        for group_ranks in self.device_mesh.tp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.tp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init TP group: {group_ranks}"
                )
            self.all_tp_groups.append(group_ranks)

        # Construct dp groups
        self.all_dp_groups = []
        for group_ranks in self.device_mesh.dp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.dp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init DP group: {group_ranks}"
                )
            self.all_dp_groups.append(group_ranks)

        # Construct pp groups
        self.all_pp_groups = []
        for group_ranks in self.device_mesh.pp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.pp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init PP group: {group_ranks}"
                )
            self.all_pp_groups.append(group_ranks)

    def activation_parallel_is_initialized(self) -> bool:
        # DP groups only active on group rank0 TP workers, so this is
        # intersection not union
        if self.tp_group is None and self.dp_group is None and self.pp_group is None:
            return False
        return True

    def in_tensor_parallel_group(self) -> bool:
        return self.tp_group is not None

    def in_pipeline_parallel_group(self) -> bool:
        return self.pp_group is not None

    def in_data_parallel_group(self) -> bool:
        return self.dp_group is not None

    def get_activation_tensor_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        assert self.activation_parallel_is_initialized()
        return self.tp_group

    def get_activation_data_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        assert self.activation_parallel_is_initialized()
        return self.dp_group

    def get_activation_pipeline_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        assert self.activation_parallel_is_initialized()
        return self.pp_group

    def get_activation_tensor_parallel_world_size(self) -> int:
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.tp_group)

    def get_activation_data_parallel_world_size(self) -> int:
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.dp_group)

    def get_activation_pipeline_parallel_world_size(self) -> int:
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.pp_group)

    def get_activation_tensor_parallel_rank(self) -> int:
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.tp_group)

    def get_activation_data_parallel_rank(self) -> int:
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.dp_group)

    def get_activation_pipeline_parallel_rank(self) -> int:
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.pp_group)

    def destroy_activation_parallel(self) -> None:
        self.all_tp_groups = None
        self.all_pp_groups = None
        self.all_dp_groups = None
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
