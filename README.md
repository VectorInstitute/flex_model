# Installation
Run `pip install -e .` from the root directory.

# Important Notes
- Make sure to replace any instances of `module.forward(inputs)` with
`module(inputs)`. The forward hooks are not run by PyTorch if you directly call
the forward function of a module (this is the case with LLaMA).

# Usage
Here's a short example on how you would use the FlexModel and HookFunction
classes. When using distributed models using FSDP or Megatron layers, the
`FlexModel` class requires specification of data parallel (DP), tensor parallel
(TP), and pipeline parallel (PP) sizes.
```
from flex_model import FlexModel, HookFunction

# Load some model
model = ...
inputs = ...

# Activations will be dumped here
output_dict = {}

# Wrap the model
model = FlexModel(
	model,
	output_dict,
	tensor_parallel_size=1,
	pipeline_parallel_size=1,
	data_parallel_size=4,		# For FSDP over 4 GPUs
)

# Only need to provide a shape hint on the dimension which may be sharded
# across gpus.
# (batch, sequence, hidden) -> sharded along hidden
expected_shape = (None, None, hidden_dim)

# Define a hook function
def editing_function(current_module, inputs, save_ctx, modules) -> Tensor
	# Save input activation tensor for later use
	save_ctx.activation = inputs

	# Edit the activation tensor
	edited_activations = inputs * 2

	# Apply torch.nn.Modules to the activation tensor (can generate grads)
	edited_activations = modules["mlp"].forward(edited_activations)

	return edited_activations

# Create hook function
hook_function = HookFunction(
	module_name="layers.7.feed_forward.w1",
	expected_shape=expected_shape,
	editing_function=editing_function,
)

# Register the hook function into our model
flex_model.register_forward_hook(hook_function)

# Run a forward pass, activations will be placed in the output_dict
model.forward(inputs)
```

# Running Tests
Running single-gpu tests in the `tests/` folder using `pytest` can be done with
the command:
```
pytest --ignore=multi_gpu
```

Multi-gpu tests are run via `submitit` on a `slurm` cluster. Navigate to
`tests/multi_gpu` and run the command:
```
python run_multi_gpu_tests_slurm.py
```
The multi-gpu tests require 4 GPUs to run.
