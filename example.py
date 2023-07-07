import argparse

import accelerate
import torch
import transformers
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/ssd005/projects/llm/llama-65b-hf")
    args = parser.parse_args()
    return args


# TODO: Clean this up and test
def main(args):
    accelerator = Accelerator()
    model = LlamaForCausalLM.from_pretrained(
        args.ckpt_dir,
        local_files_only=True,
    )
    model = accelerator.prepare(model)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.ckpt_dir,
    )

    output_dict = {}

    flex_model = DistributedFlexModel(
        ranks=[0, 1],
        module=model,
        output_ptr=output_dict,
    )

    hft = HookFunctionTriple(
        module_name="_fsdp_wrapped_module.model.layers.28._fsdp_wrapped_module.mlp.up_proj",
        shape=(3, 11, 22016),
        editing_fn=lambda x: x,
    )
    flex_model.register_hook_function_triple(hft)

    prompts = [
        "Hi I'm Matt, where am I?",
        "Welcome to Vector",
        "The tensor has a shape of",
    ]
    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )["input_ids"]

    flex_model.forward(inputs)

    print(output_dict)



if __name__ == "__main__":
    args = parse_args()
    main(args)
