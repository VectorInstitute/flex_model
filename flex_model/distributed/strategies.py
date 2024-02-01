from typing import Any, Callable, Dict, List, Optional

from torch import Tensor

import flex_model.distributed as fm_dist
from flex_model.distributed.distributed_state import _ParallelStateAPI

"""
We define strategies for:
1. Routing: Where/how activations are communicated in a 3D mesh.
2. Offload: Which devices offload tensors to CPU.
3. Function: Which devices run user-provided editing functions.

A strategy is a function which defines some operation on a single tensor.
Instantiation of a strategy may require arguments dependent on the specific
strategy, but can be reapplied as long as the model definition and sharding
strategy do not change.
"""


class BaseRoutingStrategy:
    """
    Defines a routing strategy, which every device participates in. Moves
    corresponding tensors via collective communication.
    """

    def __init__(self, prologue_fn, epilogue_fn):
        self.prologue_fn = prologue_fn
        self.epilogue_fn = epilogue_fn

    @classmethod
    def initialize(cls, tensor, expected_shape) -> None:
        raise NotImplementedError("Routing Strategy must implement this")

    def execute_prologue(self, tensor: Tensor) -> Tensor:
        tensor = self.prologue_fn(tensor)
        return tensor

    def execute_epilogue(self, tensor: Tensor) -> Tensor:
        tensor = self.epilogue_fn(tensor)
        return tensor


class ParameterTensorParallelRoutingStrategy(BaseRoutingStrategy):
    """Defines a routing strategy for parameter tensors supporting TP sharding."""

    @classmethod
    def initialize(
        cls, fmps: _ParallelStateAPI, tensor: Tensor, expected_shape
    ) -> None:
        # Handle unspecified dimensions.
        input_shape = tensor.shape
        if expected_shape is None:
            expected_shape = tuple(None for _ in range(len(input_shape)))

        full_tensor_shape = tuple(
            d1 if d2 is None else d2
            for d1, d2 in zip(input_shape, expected_shape)
        )

        # Determine which, if any, dimensions need to be gathered over TP.
        gather_dims = []
        for i, (in_dim, full_dim) in enumerate(
            zip(input_shape, full_tensor_shape)
        ):
            if in_dim != full_dim:
                gather_dims.append(i)

        if len(gather_dims) < 1:
            gather_tp = False
        elif len(gather_dims) == 1:
            gather_tp = True
        else:
            # TODO: Multi-dim TP gathering.
            raise NotImplementedError(
                f"Tensor-parallel routing only supports one dimension, found {len(gather_dims)}"
            )
        sharded_dim = gather_dims[0] if len(gather_dims) > 0 else None

        def _gather_only_tp(t):
            return fm_dist.all_gather_tensor_parallel(t, sharded_dim, fmps)

        def _scatter_only_tp(t):
            return fm_dist.scatter_tensor_parallel(t, sharded_dim, fmps)

        def _unity(t):
            return fm_dist.unity(t, fmps)

        if gather_tp:
            prologue_fn = _gather_only_tp
            epilogue_fn = _scatter_only_tp
        else:
            prologue_fn = _unity
            epilogue_fn = _unity

        return cls(prologue_fn, epilogue_fn)


class ActivationTensorAllToAllRoutingStrategy(BaseRoutingStrategy):
    """
    Defines a routing strategy which materializes the activation tensor on all
    TP and DP ranks via all-gather collectives.
    """

    @classmethod
    def initialize(
        cls,
        fmps: _ParallelStateAPI,
        tensor: Optional[Tensor],
        expected_shape,
    ) -> None:
        def _unity(t):
            return fm_dist.unity(t, fmps)

        if tensor is None:
            return cls(_unity, _unity)

        dp_world_size = fmps.get_data_parallel_world_size()

        # Handle unspecified dimensions.
        input_shape = tensor.shape
        if expected_shape is None:
            expected_shape = tuple(None for _ in range(len(input_shape)))

        full_tensor_shape = tuple(
            d1 if d2 is None else d2
            for d1, d2 in zip(input_shape, expected_shape)
        )

        # Determine which, if any, dimensions need to be gathered over TP.
        gather_dims = []
        for i, (in_dim, full_dim) in enumerate(
            zip(input_shape, full_tensor_shape)
        ):
            if in_dim != full_dim:
                gather_dims.append(i)

        if len(gather_dims) < 1:
            gather_tp = False
        elif len(gather_dims) == 1:
            gather_tp = True
        else:
            # TODO: Multi-dim TP gathering.
            raise NotImplementedError(
                f"Tensor-parallel routing only supports one dimension, found {len(gather_dims)}"
            )

        # Only relevant if we need to gather_tp.
        sharded_dim = gather_dims[0] if len(gather_dims) > 0 else None

        gather_dp = True if dp_world_size > 1 else False

        # Define helper functions for collection/dispersion.
        def _gather_only_tp(t):
            return fm_dist.all_gather_tensor_parallel(t, sharded_dim, fmps)

        def _scatter_only_tp(t):
            return fm_dist.scatter_tensor_parallel(t, sharded_dim, fmps)

        def _gather_only_dp(t):
            return fm_dist.all_gather_data_parallel(t, 0, fmps)

        def _scatter_only_dp(t):
            return fm_dist.scatter_data_parallel(t, 0, fmps)

        def _gather_tp_then_dp(t):
            return fm_dist.all_gather_data_parallel(
                fm_dist.all_gather_tensor_parallel(t, sharded_dim, fmps),
                0,
                fmps,
            )

        def _scatter_dp_then_tp(t):
            return fm_dist.scatter_tensor_parallel(
                fm_dist.scatter_data_parallel(t, 0, fmps),
                sharded_dim,
                fmps,
            )

        if not gather_tp and not gather_dp:
            prologue_fn = _unity
            epilogue_fn = _unity

        elif not gather_tp and gather_dp:
            prologue_fn = _gather_only_dp
            epilogue_fn = _scatter_only_dp

        elif gather_tp and not gather_dp:
            prologue_fn = _gather_only_tp
            epilogue_fn = _scatter_only_tp

        else:
            prologue_fn = _gather_tp_then_dp
            epilogue_fn = _scatter_dp_then_tp

        return cls(prologue_fn, epilogue_fn)


class BaseOffloadStrategy:
    """
    Defines an offload strategy, which each device may or may not participate
    in. Offloading means taking the tensor and disconnecting it from any
    computation graph for separate downstream processing.
    """

    def __init__(self, offload_fn):
        self.offload_fn = offload_fn

    def execute(self, tensor: Tensor) -> None:
        self.offload_fn(tensor)


class NullMemoryOffloadStrategy(BaseOffloadStrategy):
    @classmethod
    def initialize(cls, name: str, output_ptr: Dict[str, List[Tensor]]) -> None:
        return cls(lambda x: x)


class CPUPinnedMemoryOffloadStrategy(BaseOffloadStrategy):
    @classmethod
    def initialize(cls, name: str, output_ptr: Dict[str, List[Tensor]]) -> None:
        def _offload(t):
            if name not in output_ptr:
                output_ptr[name] = []
            output_ptr[name].append(t.detach().to("cpu", non_blocking=True))

        return cls(_offload)


class CPUPagedMemoryOffloadStrategy(BaseOffloadStrategy):
    @classmethod
    def initialize(cls, name: str, output_ptr: Dict[str, List[Tensor]]) -> None:
        def _offload(t):
            if name not in output_ptr:
                output_ptr[name] = []

            output_ptr[name].append(t.detach().cpu())

        return cls(_offload)


class GPUMemoryOffloadStrategy(BaseOffloadStrategy):
    @classmethod
    def initialize(cls, name: str, output_ptr: Dict[str, List[Tensor]]) -> None:
        def _offload(t):
            if name not in output_ptr:
                output_ptr[name] = []

            output_ptr[name].append(t.detach().clone())

        return _offload


class BaseFunctionStrategy:
    """
    Defines an editing function execution strategy. Can validate an editing
    function (ie. check breakpoints, etc.). Determines which tensors have the
    editing function applied.
    """

    def __init__(self, user_func: Callable):
        self.user_func = user_func

    @classmethod
    def initialize(cls, func: Callable):
        raise NotImplementedError

    def execute(self, *args, **kwargs) -> Any:
        return self.user_func(*args, **kwargs)


class NonValidatedFunctionStrategy(BaseFunctionStrategy):
    @classmethod
    def initialize(cls, user_func: Callable):
        def is_valid(fn) -> bool:
            return True

        if is_valid(user_func):
            return cls(user_func)
        else:
            raise Exception("Provided editing function is not valid")
