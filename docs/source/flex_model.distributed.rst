flex\_model.distributed package
===============================

.. currentmodule:: flex_model.distributed


Backends
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    GPUDeviceMesh
    TorchDistributedBackend


Distributed API
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    initialize_distributed_backend
    distributed_backend_is_initialized
    destroy_distributed_backend


Mappings
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    unity
    broadcast_tensor_parallel
    broadcast_data_parallel
    all_gather_tensor_parallel
    all_gather_data_parallel
    scatter_tensor_parallel
    scatter_data_parallel
    gather_pipeline_parallel


Parsers
-------

.. autosummary::
    :toctree: generated
    :nosignatures:

    parse_collect_from_parameter_tensor
    parse_collect_and_distribute_from_tensor
