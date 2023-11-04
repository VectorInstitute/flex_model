# FlexModel Performance Testing
To optimize performance of models wrapped with `FlexModel`, this folder is
used to generate PyTorch Kineto profiles using `torch.profiler.profile()`. To
generate these reports for a variety of single/multi-pu experiments, run the
command:
```
torchrun --nnodes <nnodes> --nproc_per_node <gpus_per_node> --rdzv_id 6969 \
profile_hooks.py --dtype bf16 \
--model_dim 4096 \
--profile_show
```

This will run the profiler on a test model (see `utils.TestNetwork`) which
uses Megatron core `ColumnParallelLinear` and `RowParallelLinear` layers.
Additional script options can be found inside of the `profile_hooks.py` script.

# Visualizing Profiles
Visualizations can be created by running the Jupyter Notebook within this
folder.
