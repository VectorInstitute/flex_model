# Installation
Run `pip install -e .` from the root directory.

# Important Notes
- Testing the correctness of the fairscale llama activations is still a WIP. The
huggingface llama model implementation is slightly different, especially when
it comes to the RoPE positional embedding implementations. Future tests will 
compare between non-distributed and distributed versions of the models.

- Make sure to replace any instances of `module.forward(inputs)` with
`module(inputs)`. The forward hooks are not run if you directly call
the forward function of a module (this is the case with LLaMA).

# Usage
Here's a short example on how you would use the FlexModel and HookFunction
classes. 
```
from flex_model import FlexModel, HookFunction

# Load some model
model = ...
inputs = ...

# Activations will be dumped here
output_dict = {}

# Wrap the model
model = FlexModel(model, output_dict)

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
flex_model.register_hook_function(hook_function)

# Run a forward pass, activations will be placed in the output_dict
model.forward(inputs)
```

# Running Tests
Use `torchrun` to run all of the tests requiring distributed. Else you can just
use `python3`. All distributed tests were done on 2 GPUs.
```
torchrun --nnodes 1 \
	--nproc_per_node 2 \
	--rdzv_id 6969 \
	--rdzv_backend c10d \
	--rdzv_endpoint <gpuXXX> \
	--test_<name>.py
```

# Running TunedLens
Here's an example command to do a training run of TunedLens using the
FlexModel backend to retrieve weights and activations. The implementation is
basic and does not do any checkpointing. For full launch options, check out
the `test_tunedlens.py` script.
```
torchrun --nnodes 1 \
	--nproc_per_node 2 \
	--rdzv_id 6969 \
	--rdzv_backend c10d \
	--rdzv_endpoint <gpuXXX> \
	--test_tunedlens.py \
	--log_level warning \
	--batch_size 16 \
	--num_steps 250
```
