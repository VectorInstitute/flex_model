#!/bin/bash
set -e

export NCCL_IB_DISABLE=1

torchrun \
	--nnodes 1 \
	--nproc_per_node 4 \
	--rdzv_id 6969 \
	--rdzv_backend c10d \
	core/test_megatron_layers.py

accelerate launch core/test_huggingface_multi_gpu.py

python core/test_core_wrapper_single_gpu.py

python core/test_hook_function_single_gpu.py

accelerate launch core/test_save_ctx.py

torchrun \
	--nnodes 1 \
	--nproc_per_node 4 \
	--rdzv_id 6969 \
	--rdzv_backend c10d \
	distributed/test_initialize.py

torchrun \
	--nnodes 1 \
	--nproc_per_node 2 \
	--rdzv_id 6969 \
	--rdzv_backend c10d \
	distributed/test_mappings.py
