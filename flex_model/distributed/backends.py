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

            pp groups -> [gpu0, gpu8]

            dp groups -> [gpu0, gpu4],
                         [gpu8, gpu12],

            Where gathering in the dp groups would need to be done
            hierarchically, starting from a gather across tp groups first
            and then the gather across the dp groups. Gathering the pipeline
            parallel groups is done at the end of the forward pass when all
            pipeline parallel ranks have their retrieved activations on CPU.

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
        # TODO: World size param can probably be inferred.
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
    distributed backends must implement methods for managing models as a 3D
    (tensor, pipeline and parallel) mesh.
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

    See parent class :class:`DistributedBackend` for details.
    
    :var GPUDeviceMesh device_mesh: Mesh of GPU devices that defines the
        :class:`FlexModel` activation management strategy.
    :var all_tp_groups: All tensor parallel :code:`torch.distributed` groups.
    :type all_tp_groups: Optional[List[pt_dist.Process_group]]
    :var all_pp_groups: All pipeline parallel :code:`torch.distributed` groups.
    :type all_pp_groups: Optional[List[pt_dist.Process_group]]
    :var all_dp_groups: All data parallel :code:`torch.distributed` groups.
    :type all_dp_groups: Optional[List[pt_dist.Process_group]]
    :var tp_group: The local GPUs corresponding tensor parallel group.
    :type tp_group: Optional[pt_dist.ProcessGroup]
    :var pp_group: The local GPUs corresponding pipeline parallel group.
    :type pp_group: Optional[pt_dist.ProcessGroup]
    :var dp_group: The local GPUs corresponding data parallel group.
    :type dp_group: Optional[pt_dist.ProcessGroup]

    :note: Even after :code:`initialize_activation_parallel` is called,
        :code:`pp_group` and :code:`dp_group` may still be :code:`None`. This
        is because unlike 3-D parallelism model training frameworks like
        :code:`Megatron-LM`, some of the data and pipeline parallel groups are
        redundant for our use cases.
    """
    def __init__(self, device_mesh: GPUDeviceMesh) -> None:
        """Instantiates the torch distributed backend.

        :param GPUDeviceMesh device_mesh: GPU device mesh that provides
            distributed group ranks to initialize.
        """
        self.device_mesh = device_mesh
        self.all_tp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.all_pp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.all_dp_groups: Optional[List[pt_dist.ProcessGroup]] = None
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

        # Initialize torch distributed tp groups
        self.all_tp_groups = []
        for group_ranks in self.device_mesh.tp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.tp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init TP group: {group_ranks}"
                )
            self.all_tp_groups.append(group)

        # Initialize torch distributed dp groups
        self.all_dp_groups = []
        for group_ranks in self.device_mesh.dp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.dp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init DP group: {group_ranks}"
                )
            self.all_dp_groups.append(group)

        # Initialize torch distributed pp groups
        self.all_pp_groups = []
        for group_ranks in self.device_mesh.pp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.pp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init PP group: {group_ranks}"
                )
            self.all_pp_groups.append(group)

    def activation_parallel_is_initialized(self) -> bool:
        """Check if activation parallel backend has been initialized.

        :returns: True if any tensor, pipeline or data parallel groups have been
            initialized.
        :rtype: bool
        """
        if self.tp_group is None and self.dp_group is None and self.pp_group is None:
            return False
        return True

    def in_tensor_parallel_group(self) -> bool:
        """Check if local device is in a tensor parallel group.

        :returns: True if the local device is in a tensor parallel group.
        :rtype: bool
        """
        return self.tp_group is not None

    def in_pipeline_parallel_group(self) -> bool:
        """Check if local device is in a pipeline parallel group.

        :returns: True if the local device is in a pipeline parallel group.
        :rtype: bool
        """
        return self.pp_group is not None

    def in_data_parallel_group(self) -> bool:
        """Check if local device is in a data parallel group.

        :returns: True if the local device is in a data parallel group.
        :rtype: bool
        """
        return self.dp_group is not None

    def get_activation_tensor_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the tensor parallel group handle the local device belongs to.

        :returns: The tensor parallel distributed group.
        :rtype: Optional[pt_dist.ProcessGroup]

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.
        """
        assert self.activation_parallel_is_initialized()
        return self.tp_group

    def get_activation_pipeline_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the pipeline parallel group handle the local device belongs to.

        :returns: The pipeline parallel distributed group.
        :rtype: Optional[pt_dist.ProcessGroup]

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.
        """
        assert self.activation_parallel_is_initialized()
        return self.pp_group

    def get_activation_data_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the data parallel group handle the local device belongs to.

        :returns: The data parallel distributed group.
        :rtype: Optional[pt_dist.ProcessGroup]

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return self.dp_group

    def get_activation_tensor_parallel_world_size(self) -> int:
        """Get the number of devices in the tensor parallel group.

        :returns: Tensor parallel world size.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.
        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.tp_group)

    def get_activation_pipeline_parallel_world_size(self) -> int:
        """Get the number of devices in the pipeline parallel group.
        :returns: Pipeline parallel world size.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.pp_group)

    def get_activation_data_parallel_world_size(self) -> int:
        """Get the number of devices in the data parallel group.
        :returns: Data parallel world size.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.dp_group)

    def get_activation_tensor_parallel_rank(self) -> int:
        """Get the rank of the local device in the tensor parallel group.
        :returns: Tensor parallel rank.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.tp_group)

    def get_activation_pipeline_parallel_rank(self) -> int:
        """Get the rank of the local device in the pipeline parallel group.
        :returns: Pipeline parallel rank.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.pp_group)

    def get_activation_data_parallel_rank(self) -> int:
        """Get the rank of the local device in the data parallel group.
        :returns: Data parallel rank.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.dp_group)

    def destroy_activation_parallel(self) -> None:
        """Delete all handles to tensor, pipeline and data parallel groups.
        """
        self.all_tp_groups = None
        self.all_pp_groups = None
        self.all_dp_groups = None
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None


class AccelerateDistributedBackend(DistributedBackend):
    """Distributed backend using for Pytorch distributed.

    See parent class :class:`DistributedBackend` for details.
    
    :var GPUDeviceMesh device_mesh: Mesh of GPU devices that defines the
        :class:`FlexModel` activation management strategy.
    :var all_tp_groups: All tensor parallel :code:`torch.distributed` groups.
    :type all_tp_groups: Optional[List[pt_dist.Process_group]]
    :var all_pp_groups: All pipeline parallel :code:`torch.distributed` groups.
    :type all_pp_groups: Optional[List[pt_dist.Process_group]]
    :var all_dp_groups: All data parallel :code:`torch.distributed` groups.
    :type all_dp_groups: Optional[List[pt_dist.Process_group]]
    :var tp_group: The local GPUs corresponding tensor parallel group.
    :type tp_group: Optional[pt_dist.ProcessGroup]
    :var pp_group: The local GPUs corresponding pipeline parallel group.
    :type pp_group: Optional[pt_dist.ProcessGroup]
    :var dp_group: The local GPUs corresponding data parallel group.
    :type dp_group: Optional[pt_dist.ProcessGroup]

    :note: Even after :code:`initialize_activation_parallel` is called,
        :code:`pp_group` and :code:`dp_group` may still be :code:`None`. This
        is because unlike 3-D parallelism model training frameworks like
        :code:`Megatron-LM`, some of the data and pipeline parallel groups are
        redundant for our use cases.
    """
    # TODO: Just a copy of torch distributed backend. Need to add in
    #       accelerate-specific methods.
    def __init__(self, device_mesh: GPUDeviceMesh) -> None:
        """Instantiates the torch distributed backend.

        :param GPUDeviceMesh device_mesh: GPU device mesh that provides
            distributed group ranks to initialize.
        """
        self.device_mesh = device_mesh
        self.all_tp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.all_pp_groups: Optional[List[pt_dist.ProcessGroup]] = None
        self.all_dp_groups: Optional[List[pt_dist.ProcessGroup]] = None
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

        # Initialize torch distributed tp groups
        self.all_tp_groups = []
        for group_ranks in self.device_mesh.tp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.tp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init TP group: {group_ranks}"
                )
            self.all_tp_groups.append(group)

        # Initialize torch distributed dp groups
        self.all_dp_groups = []
        for group_ranks in self.device_mesh.dp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.dp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init DP group: {group_ranks}"
                )
            self.all_dp_groups.append(group)

        # Initialize torch distributed pp groups
        self.all_pp_groups = []
        for group_ranks in self.device_mesh.pp_group_ranks:
            group = pt_dist.new_group(group_ranks)
            if rank in group_ranks:
                self.pp_group = group
                logger.debug(
                    f"Torch rank {pt_dist.get_rank()} init PP group: {group_ranks}"
                )
            self.all_pp_groups.append(group)

    def activation_parallel_is_initialized(self) -> bool:
        """Check if activation parallel backend has been initialized.

        :returns: True if any tensor, pipeline or data parallel groups have been
            initialized.
        :rtype: bool
        """
        if self.tp_group is None and self.dp_group is None and self.pp_group is None:
            return False
        return True

    def in_tensor_parallel_group(self) -> bool:
        """Check if local device is in a tensor parallel group.

        :returns: True if the local device is in a tensor parallel group.
        :rtype: bool
        """
        return self.tp_group is not None

    def in_pipeline_parallel_group(self) -> bool:
        """Check if local device is in a pipeline parallel group.

        :returns: True if the local device is in a pipeline parallel group.
        :rtype: bool
        """
        return self.pp_group is not None

    def in_data_parallel_group(self) -> bool:
        """Check if local device is in a data parallel group.

        :returns: True if the local device is in a data parallel group.
        :rtype: bool
        """
        return self.dp_group is not None

    def get_activation_tensor_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the tensor parallel group handle the local device belongs to.

        :returns: The tensor parallel distributed group.
        :rtype: Optional[pt_dist.ProcessGroup]

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.
        """
        assert self.activation_parallel_is_initialized()
        return self.tp_group

    def get_activation_pipeline_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the pipeline parallel group handle the local device belongs to.

        :returns: The pipeline parallel distributed group.
        :rtype: Optional[pt_dist.ProcessGroup]

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.
        """
        assert self.activation_parallel_is_initialized()
        return self.pp_group

    def get_activation_data_parallel_group(self) -> Optional[pt_dist.ProcessGroup]:
        """Get the data parallel group handle the local device belongs to.

        :returns: The data parallel distributed group.
        :rtype: Optional[pt_dist.ProcessGroup]

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return self.dp_group

    def get_activation_tensor_parallel_world_size(self) -> int:
        """Get the number of devices in the tensor parallel group.

        :returns: Tensor parallel world size.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.
        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.tp_group)

    def get_activation_pipeline_parallel_world_size(self) -> int:
        """Get the number of devices in the pipeline parallel group.
        :returns: Pipeline parallel world size.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.pp_group)

    def get_activation_data_parallel_world_size(self) -> int:
        """Get the number of devices in the data parallel group.
        :returns: Data parallel world size.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_world_size(group=self.dp_group)

    def get_activation_tensor_parallel_rank(self) -> int:
        """Get the rank of the local device in the tensor parallel group.
        :returns: Tensor parallel rank.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.tp_group)

    def get_activation_pipeline_parallel_rank(self) -> int:
        """Get the rank of the local device in the pipeline parallel group.
        :returns: Pipeline parallel rank.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.pp_group)

    def get_activation_data_parallel_rank(self) -> int:
        """Get the rank of the local device in the data parallel group.
        :returns: Data parallel rank.
        :rtype: int

        :raises AssertionError: If the activation parallel backend hasn't been
            initialized yet.

        """
        assert self.activation_parallel_is_initialized()
        return pt_dist.get_rank(group=self.dp_group)

    def destroy_activation_parallel(self) -> None:
        """Delete all handles to tensor, pipeline and data parallel groups.
        """
        self.all_tp_groups = None
        self.all_pp_groups = None
        self.all_dp_groups = None
        self.tp_group = None
        self.pp_group = None
        self.dp_group = None
