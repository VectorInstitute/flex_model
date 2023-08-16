from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch.distributed as pt_dist
from accelerate import PartialState


@dataclass
class GPUDeviceMesh:
    """Device mesh containing torch distributed groups.

    Even with huggingface accelerate backend, torch distributed is still used.
    """
    tp_group_ranks: List[List[int]]
    pp_groups_ranks: List[List[int]]
    dp_groups_ranks: List[List[int]]

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

            pp groups -> NA (we don't gather across different layers)

            megatron dp groups -> [gpu0, gpu8],
                         [gpu1, gpu9],
                         [gpu2, gpu10],
                         [gpu3, gpu11],
                         [gpu4, gpu12],
                         [gpu5, gpu13],
                         [gpu6, gpu14],
                         [gpu7, gpu15],

            our dp groups -> [gpu0, gpu1, gpu2, gpu3, gpu8, gpu9, gpu10, gpu11],
                             [gpu4, gpu5, gpu6, gpu7, gpu12, gpu13, gpu14, gpu15]

            Where gathering in the dp groups would need to be done
            hierarchically, starting from a gather across tp groups first
            and then the gather across the dp groups.
        """
        if world_size == 1:
            return cls([[0]], [[0]], [[0]])

        num_tp_groups = world_size // tensor_parallel_size
        num_pp_groups = 0
        num_dp_groups = world_size // data_parallel_size

        # Build tensor parallel groups
        tp_group_ranks: List[List[int]] = []
        stride = 1
        for i in range(num_tp_groups):
            offset = i * tensor_parallel_size
            ranks = [offset + stride * j for j in range(tensor_parallel_size)]
            tp_group_ranks.append(ranks)

        # Build data parallel groups
        dp_group_ranks: List[List[int]] = []
        stride = pipeline_parallel_size
        for i in range(num_dp_groups):
            offset = i
            ranks = []
            for j in range(data_parallel_size):
                ranks += tp_group_ranks[offset + stride * j]

        return cls(tp_group_ranks, [[]], dp_group_ranks)


class DistributedBackend(ABC):
    @abstractmethod
    def initialize_activation_parallel(self) -> None:
        ...

    @abstractmethod
    def activation_parallel_is_initialized(self) -> bool:
        ...

    @abstractmethod
    def get_activation_tensor_parallel_world_size(self) -> int:
        ...

    @abstractmethod
    def get_activation_data_parallel_world_size(self) -> int:
        ...

    @abstractmethod
    def get_activation_tensor_parallel_rank(self) -> int:
        ...

    @abstractmethod
    def get_activation_data_parallel_rank(self) -> int:
        ...

    @abstractmethod
    def destroy_activation_parallel(self) -> None:
        ...


class TorchDistributedBackend(DistributedBackend):
    def __init__(self, device_mesh: GPUDeviceMesh):
        self.device_mesh = device_mesh
        self.all_tp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.all_pp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.all_dp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.tp_group: Optional[pt_dist.ProcessGroup] = None
        self.pp_group: Optional[pt_dist.ProcessGroup] = None
        self.dp_group: Optional[pt_dist.ProcessGroup] = None

    def initialize_activation_parallel(self) -> None:
        rank = pt_dist.get_rank()

        # Construct tp groups
        self.all_tp_groups = []
        for group_ranks in self.device_mesh.tp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.tp_group = group
            self.all_tp_groups.append(group)

        # Construct dp groups
        self.all_dp_groups = []
        for group_ranks in self.device_mesh.tp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.dp_group = group
            self.all_dp_groups.append(group)

    def activation_parallel_is_initialized(self) -> bool:
        return self.tp_group is not None and self.dp_group is not None

    def get_activation_tensor_parallel_world_size(self) -> int:
        return pt_dist.get_world_size(group=self.tp_group)

    def get_activation_data_parallel_world_size(self) -> int:
        return pt_dist.get_world_size(group=self.dp_group)

    def get_activation_tensor_parallel_rank(self) -> int:
        return pt_dist.get_rank(group=self.tp_group)

    def get_activation_data_parallel_rank(self) -> int:
        return pt_dist.get_rank(group=self.dp_group)

    def destroy_activation_parallel(self) -> None:
        self.all_tp_groups = None
        self.all_pp_groups = None
        self.all_dp_groups = None
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None


class AcclerateDistributedBackend(DistributedBackend):
    def __init__(self, device_mesh: GPUDeviceMesh):
        self.device_mesh = device_mesh
        self.all_tp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.all_pp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.all_dp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.tp_group: Optional[pt_dist.ProcessGroup] = None
        self.pp_group: Optional[pt_dist.ProcessGroup] = None
        self.dp_group: Optional[pt_dist.ProcessGroup] = None

    def initialize_activation_parallel(self):
        rank = pt_dist.get_rank()

        # Construct tp groups
        self.all_tp_groups = []
        for group_ranks in self.device_mesh.tp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.tp_group = group
            self.all_tp_groups.append(group)

        # Construct dp groups
        self.all_dp_groups = []
        for group_ranks in self.device_mesh.tp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.dp_group = group
            self.all_dp_groups.append(group)

    def activation_parallel_is_initialized(self) -> bool:
        return self.tp_group is not None and self.dp_group is not None

    def get_activation_tensor_parallel_world_size(self) -> int:
        return pt_dist.get_world_size(group=self.tp_group)

    def get_activation_data_parallel_world_size(self) -> int:
        return pt_dist.get_world_size(group=self.dp_group)

    def get_activation_tensor_parallel_rank(self) -> int:
        return pt_dist.get_rank(group=self.tp_group)

    def get_activation_data_parallel_rank(self) -> int:
        return pt_dist.get_rank(group=self.dp_group)

    def destroy_activation_parallel(self) -> None:
        self.all_tp_groups = None
        self.all_pp_groups = None
        self.all_dp_groups = None
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
