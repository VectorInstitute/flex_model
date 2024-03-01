Distributed: Backend, mappings and strategies
=============================================

.. currentmodule:: flex_model.distributed


Distributed API
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    initialize_distributed_state


Mappings
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast_tensor_parallel
    broadcast_data_parallel
    all_gather_tensor_parallel
    all_gather_data_parallel
    scatter_tensor_parallel
    scatter_data_parallel
    gather_pipeline_parallel_tensor_dicts


Strategies
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseRoutingStrategy
    ParameterTensorParallelRoutingStrategy
    ActivationTensorAllToAllRoutingStrategy
    BaseOffloadStrategy
    NullMemoryOffloadStrategy
    CPUPinnedMemoryOffloadStrategy
    CPUPagedMemoryOffloadStrategy
    GPUMemoryOffloadStrategy
    BaseFunctionStrategy
    NonValidatedFunctionStrategy
