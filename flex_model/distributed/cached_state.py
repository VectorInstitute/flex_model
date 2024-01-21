from torch import Tensor
import torch.nn as nn

from flex_model.distributed.distributed_state import _ParallelStateAPI
from flex_model.distributed.stratgies import (
    SaveCtxStrategy,
    TrainableModulesStrategy,
)


def pipeline_sync(obj_to_sync, fmps: _ParallelStateAPI):
    raise NotImplementedError


class SaveContext:
    def __init__(
        self,
        fmps: _ParallelStateAPI,
        strategy: SaveCtxStrategy = SaveCtxStrategy.REPLICATE_ALL,
    ):
        self.strategy = strategy
        self.fmps = fmps

    def save(self, *tensors: Tensor):
        for t in tensors:
            assert isinstance(t, Tensor), (
                "The `save` function should only be used on tensor instances. ",
                "Non-tensor data can be saved using `save_ctx.data = item.",
            )

        # Don't cache tensors depending on strategy.
        # dp_rank = self.fmps.get_data_parallel_rank()
        # pp_rank = self.fmps.get_pipeline_parallel_rank()
        # tp_rank = self.fmps.get_tensor_parallel_rank()

        # Check if this rank should actually cache tensors.
        do_cache_tensors = True
        if self.strategy != SaveCtxStrategy.REPLICATE_ALL:
            if self.strategy == SaveCtxStrategy.REPLICATE_PP:
                raise NotImplementedError(
                    "Passing save context along pipeline ranks is not "
                    "implemented."
                )

        # Cache tensors if necessary.
        if do_cache_tensors is True:
            self.cached_tensors = tensors

    def get_cached_tensors(self):
        return self.cached_tensors

    def sync(self):
        # Check if we need to sync
        do_sync = False
        if self.strategy == SaveCtxStrategy.REPLICATE_ALL:
            do_sync = True

        # Sync if necessary.
        if do_sync is True:
            pipeline_sync(self._modules, self.fmps)
        do_sync = False


class TrainableModules(nn.ModuleDict):
    def __init__(
        self,
        fmps: _ParallelStateAPI,
        *args,
        strategy: TrainableModulesStrategy = TrainableModulesStrategy.REPLICATE_ALL,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.fmps = fmps

    def sync(self):
        # Check if we need to sync
        do_sync = False
        if self.strategy == TrainableModulesStrategy.REPLICATE_ALL:
            do_sync = True

        # Sync if necessary.
        if do_sync is True:
            pipeline_sync(self._modules, self.fmps)
