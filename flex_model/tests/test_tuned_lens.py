import torch
import torch.distributed as dist

from flex_model.model_wrappers import FlexModel, HookFunction
from flex_model.tests.testing_utils import get_llama_13b_megatron
from flex_model.tests.testing_constants import _PROMPTS


def main():
    model, tokenize_fn = get_llama_13b_megatron()

    hook_function = HookFunction(
        "layers.25",
        expected_shape=(None, None, 5120),
        editing_function=lambda x, _: x,
    )

    output_dict = {}

    model = FlexModel(model, output_dict)
    model.register_hook_function(hook_function)

    inputs = tokenize_fn(_PROMPTS)

    logits = model.forward(inputs, start_pos=0)

    unembed = model.get_module_parameter("output.weight", (32000, 5120))

    if dist.get_rank() == 0:
        print(logits)
        print(output_dict)

        #print(model.parameter_names())


        print(unembed)
        print(unembed.shape)


if __name__ == "__main__":
    main()
