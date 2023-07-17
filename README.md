# Installation
Run `pip install -e .` from the root directory.


# Usage
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

# Hook function requires a module to retrieve outputs from, the expected shape
# of the activation, and optionally an editing function.
hook_function = HookFunction(
	module_name="layers.7.feed_forward.w1",
	expected_shape=(...),
	editing_function=lambda x: x * 2,
)

flex_model.register_hook_function(hook_function)

# Run the hooked forward pass. Activations will be available in the output_dict
# after the forward pass is complete
inputs = tokenizer(...)
model.forward(inputs)
```


# Running tests
`cd flex_model/tests`

`python test_single_gpu.py`

`accelerate launch test_distributed.py` <- Make sure this is run on 2 GPUs


