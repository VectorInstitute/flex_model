flex\_model.distributed package
===============================

.. currentmodule:: flex_model.distributed


Distributed API
---------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    initialize_distributed_backend
    distributed_backend_is_initialized
    destroy_distributed_backend
    initialize_activation_parallel
    activation_parallel_is_initialized
    in_tensor_parallel_group
    in_pipeline_parallel_group
    in_data_parallel_group
    get_activation_tensor_parallel_group
    get_activation_pipeline_parallel_group
    get_activation_data_parallel_group
    get_activation_tensor_parallel_world_size
    get_activation_pipeline_parallel_world_size
    get_activation_data_parallel_world_size
    get_activation_tensor_parallel_rank
    get_activation_pipeline_parallel_rank
    get_activation_data_parallel_rank
    destroy_activation_parallel


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