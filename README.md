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

# We will use the model and tokenizer from fairscale's llama
generator = load_llama(...)
model = generator.model
tokenizer = generator.tokenizer

# Activations will be dumped here
output_dict = {}

# Wrap the llama model
model = FlexModel(model, output_dict)

# Note: Try to use huggingface tokenizer since it dumps directly into a torch
#		tensor, else you'll have to do it manually.
inputs = tokenizer(...)

# Only need to provide a shape hint on the dimension which may be sharded
# across gpus.
expected_shape = (None, None, 13824)

# Hook function requires a module to retrieve outputs from, the expected shape
# of the activation, and optionally an editing function.
hook_function = HookFunction(
	module_name="layers.7.feed_forward.w1",
	expected_shape=expected_shape,
	editing_function=lambda x: x * 2,
)

flex_model.register_hook_function(hook_function)

# Run the hooked forward pass. Activations will be available in the output_dict
# after the forward pass is complete
model.forward(inputs)
```

# Running Tests
`cd flex_model/tests`

`python test_single_gpu.py`

`accelerate launch test_distributed.py` <- Make sure this is run on 2 GPUs


