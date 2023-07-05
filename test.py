from functools import partial
import logging
import os
from typing import Tuple, Dict, List, Optional, Any, Callable

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch
import torch.nn as nn
from torch import Tensor

from flex_model import HookFunctionTriple, FlexModel, DistributedFlexModel
from utils import _recursively_find_first_tensor, print_rank0


logger = logging.getLogger(__name__)


# TODO: Break tests into smaller bite-sized chunks and prepare for pytest


class Model(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = in_dim * 4
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, self.latent_dim)
        self.nonlin = nn.ReLU()
        self.fc2 = nn.Linear(self.latent_dim, out_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        y = self.fc1(inputs)
        y = self.nonlin(y)
        y = self.fc2(y)
        return y


def print_named_modules(model: nn.Module) -> None:
    assert isinstance(model, nn.Module)
    names = [n for n, _ in model.named_modules()]
    print(names)


def apply_torch_fwd_hooks(
    model: nn.Module,
    inputs: Tensor,
    module_names: List[str],
    shapes: Optional[List[Tuple[int]]] = None,
    parse_fn: Optional[Callable] = None,
) -> Dict[str, Tensor]:
    """Retrieve activation using vanilla pytorch forward hooks."""
    module_dict = {name: shape for name, shape in zip(module_names, shapes)}

    def _fwd_hook(
        registered_name: str,
        return_dict: Dict[str, Tensor],
        module: nn.Module,
        inputs: Any,
        outputs: Any,
        parse_fn: Optional[Callable] = None,
        shape: Optional[Tuple[int]] = None,
    ) -> None:
        if parse_fn is not None:
            outputs = parse_fn(outputs)

        res = outputs.detach().cpu()

        if shape is not None:
            res.reshape(*shape)

        return_dict[registered_name] = res

    output_dict: Dict[str, Tensor] = {}
    hook_handles = []
    for name, module in model.named_modules():
        if name in module_dict:
            handle = module.register_forward_hook(
                partial(
                    _fwd_hook,
                    name,
                    output_dict,
                    parse_fn=parse_fn if parse_fn else lambda x: x,
                    shape=module_dict[name],
                )
            )
            hook_handles.append(handle)

            logger.info(f"Installing module: {name}")

    logger.info("Running forward")
    _ = model.forward(inputs)

    for handle in hook_handles:
        handle.remove()

    return output_dict


def apply_flex_model_fwd_hooks(
    model: nn.Module,
    inputs: Tensor,
    module_names: List[str],
    shapes: List[Tuple[int]],
) -> Dict[str, Tensor]:
    """Retrieve activaations using flex model forward hooks."""
    output_dict: Dict[str, Tensor] = {}
    flex_model = FlexModel(model, output_dict)
    for name, shape in zip(module_names, shapes):
        hfs = HookFunctionTriple(
            module_name=name,
            shape=shape,
            editing_fn=lambda x: x,
        )
        flex_model.register_hook_function_triple(hfs)

    flex_model.forward(inputs)

    return output_dict


def apply_distributed_flex_model_fwd_hooks(
    model: nn.Module,
    inputs: Tensor,
    module_names: List[str],
    shapes: List[Tuple[int]],
) -> Dict[str, Tensor]:
    """Retrieves activations using distributed flex model forward hooks."""
    output_dict: Dict[str, Tensor] = {}
    dist_flex_model = DistributedFlexModel([0, 1], module=model, output_ptr=output_dict)
    for name, shape in zip(module_names, shapes):
        hfs = HookFunctionTriple(
            module_name=name,
            shape=shape,
            editing_fn=lambda x: x,
        )
        dist_flex_model.register_hook_function_triple(hfs)

    dist_flex_model.forward(inputs)

    return output_dict



def compare_tensor_dicts(
    dict1: Dict[str, Tensor],
    dict2: Dict[str, Tensor],
) -> bool:
    """Check equality between two dicts."""
    for name in dict1.keys():
        if name not in dict2:
            print_rank0(f"{name} not in second dict for comparison")
            return False
        if not torch.allclose(
            dict1[name].to(torch.float32),
            dict2[name].to(torch.float32),
            atol=1e-3,
            #rtol=1e-3,
        ):
            print_rank0(f"Allclose failed: {name} -> {dict1[name].shape} - "
                        f"{dict2[name].shape}")
            print_rank0(dict1[name])
            print_rank0(dict2[name])
            return False

        logger.info(f"Allclose passed: {name}")
    return True


def test_simple_model():
    # Regular fwd pass
    model = Model(16, 32).cuda()

    inputs = torch.randn(8, 4, 16).cuda()
    outputs = model(inputs)

    # Test forward hooks
    module_names = ["fc1"]
    module_shapes = [(8 * 4, 16 * 4)]

    # Vanilla pytorch forward hooks
    logger.info("Running base forward hooks")
    test_base_dict = apply_torch_fwd_hooks(
        model,
        inputs,
        module_names,
        module_shapes,
    )

    # Flex model forward hooks
    logger.info("Running flex model forward hooks")
    test_flex_dict = apply_flex_model_fwd_hooks(
        model,
        inputs,
        module_names,
        module_shapes,
    )

    # Shape dumped is different, reshape back for comparison
    test_flex_dict["fc1"] = test_flex_dict["fc1"].reshape(8, 4, 16 * 4)

    # Correctness check
    assert compare_tensor_dicts(test_base_dict, test_flex_dict)


# TODO: More model test cases and editing functions
def test_huggingface_opt_model():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    prompt = "Hi I'm Matt, where am I?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Regular forward pass
    outputs = model.forward(inputs.input_ids)

    # Test forward hooks
    module_names = [
        "model.decoder.layers.5.self_attn",
        "model.decoder.layers.6.self_attn.out_proj",
        "model.decoder.layers.2.activation_fn",
        "model.decoder.layers.7.self_attn.v_proj",
        "model.decoder.layers.11.self_attn.k_proj",
        "model.decoder.layers.8.self_attn.q_proj",
        "model.decoder.layers.11.self_attn_layer_norm",
        "model.decoder.layers.7.fc1",
        "model.decoder.embed_tokens",
        "model.decoder.embed_positions",
        "model",
        "model.decoder",
        "model.decoder.layers",
        "model.decoder.layers.7" "model.decoder.layers.11.fc2",
        "lm_head",
    ]
    module_shapes = [None for n in range(len(module_names))]

    def _huggingface_parse_fn(x):
        """Hard-coded parse function for huggingface models.

        Note: Retrieval only.
        """
        if isinstance(x, tuple):
            return x[0]
        if isinstance(x, BaseModelOutputWithPast):
            return x.last_hidden_state
        return x

    logger.info("Running base forward hooks")
    test_base_dict = apply_torch_fwd_hooks(
        model,
        inputs.input_ids,
        module_names,
        module_shapes,
        _huggingface_parse_fn,
    )

    logger.info("Running flex model forward hooks")
    test_flex_dict = apply_flex_model_fwd_hooks(
        model,
        inputs.input_ids,
        module_names,
        module_shapes,
    )

    assert compare_tensor_dicts(test_base_dict, test_flex_dict)


def test_traversal_utils():
    from utils import _recursively_find_first_tensor

    target_tensor = torch.randn(2, 2)
    outputs = (
        (1, 3),
        ["abs", "sd", 2],
        [[target_tensor, torch.randint(10, size=(3, 3))], (2, 3)],
        (6, 5, 4, "123", nn.Linear(10, 20), torch.randn(3, 3)),
    )

    tensor = _recursively_find_first_tensor(outputs)

    assert torch.equal(tensor, target_tensor)

    from utils import _flatten, _unflatten

    # TODO: More test cases here
    outputs = [
        1,
        2,
        (torch.randn(2, 2), 3),
        "ab",
        (2, torch.randn(3, 3)),
    ]
    treedef, leaves = _flatten(outputs)

    leaves = [l * 10 for l in leaves]

    new_outputs = _unflatten(treedef, leaves)

    new_treedef, _ = _flatten(new_outputs)

    assert treedef == new_treedef


def test_distributed_flex_model():
    from accelerate import Accelerator
    from transformers import LlamaForCausalLM, LlamaTokenizer

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    tokenizer = LlamaTokenizer.from_pretrained(
        "/scratch/ssd002/projects/opt_test/llama-7b-hf",
        local_files_only=True,
    )
    tokenizer.pad_token_id=0
    tokenizer.bos_token_id=1
    tokenizer.eos_token_id=2
    def tokenize(ps):
        return tokenizer(ps, padding=True, return_tensors="pt")["input_ids"]

    prompts = [
        "Hi I'm Matt, where am I?",
        "Welcome to Vector",
        "The tensor has a shape of",
    ]

    def _get_torch_output():
        model = LlamaForCausalLM.from_pretrained(
            "/scratch/ssd002/projects/opt_test/llama-7b-hf",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        inputs = tokenize(prompts)

        module_names = [
            "model.layers.5.self_attn",
            "model.layers.6.self_attn.o_proj",
            "model.layers.2.act_fn",
            "model.layers.7.self_attn.v_proj",
            "model.layers.11.self_attn.k_proj",
            "model.layers.8.self_attn.q_proj",
            "model.layers.11.post_attention_layernorm",
            "model.layers.7.mlp",
            "model.layers.7.mlp.gate_proj",
            "model.layers.28.mlp.up_proj",
            "model.layers.9.mlp_down_proj",
            "model.embed_tokens",
            "model",
            "model.layers",
            "model.layers.7",
            "lm_head",
        ]
        module_shapes = [None for n in range(len(module_names))]

        def _huggingface_parse_fn(x):
            """Hard-coded parse function for huggingface models.

            Note: Retrieval only.
            """
            if isinstance(x, tuple):
                return x[0]
            if isinstance(x, BaseModelOutputWithPast):
                return x.last_hidden_state
            return x

        logger.info("Running base forward hooks")
        test_base_dict = apply_torch_fwd_hooks(
            model,
            inputs,
            module_names,
            module_shapes,
            _huggingface_parse_fn,
        )
        return test_base_dict


    def _get_flex_output():
        accelerator = Accelerator()
        model = LlamaForCausalLM.from_pretrained(
            "/scratch/ssd002/projects/opt_test/llama-7b-hf",
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to("cpu")
        model = accelerator.prepare(model)
        
        inputs = tokenize(prompts).to(accelerator.device)
        logger.info(f"Input tensor: {inputs}")

        # Test forward hooks
        module_names = [
            "_fsdp_wrapped_module.model.layers.5._fsdp_wrapped_module.self_attn",
            "_fsdp_wrapped_module.model.layers.6._fsdp_wrapped_module.self_attn.o_proj",
            "_fsdp_wrapped_module.model.layers.2._fsdp_wrapped_module.mlp.act_fn",
            "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.self_attn.v_proj",
            "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.self_attn.k_proj",
            "_fsdp_wrapped_module.model.layers.8._fsdp_wrapped_module.self_attn.q_proj",
            "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.post_attention_layernorm",
            "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.mlp",
            "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.mlp.gate_proj",
            "_fsdp_wrapped_module.model.layers.28._fsdp_wrapped_module.mlp.up_proj",
            "_fsdp_wrapped_module.model.layers.9._fsdp_wrapped_module.mlp_down_proj",
            "_fsdp_wrapped_module.model.embed_tokens",
            "_fsdp_wrapped_module.model",
            "_fsdp_wrapped_module.model.layers",
            "_fsdp_wrapped_module.model.layers.7",
            "_fsdp_wrapped_module.lm_head",
        ]
        module_shapes = [
            (3, 11, 4096) for n in range(len(module_names))
        ]
        module_shapes[2] = (3, 11, 11008)
        module_shapes[8] = (3, 11, 11008)
        module_shapes[9] = (3, 11, 11008)
        module_shapes[-1] = (3, 11, 32000)

        logger.info("Running flex model forward hooks")
        test_flex_dict = apply_distributed_flex_model_fwd_hooks(
            model,
            inputs,
            module_names,
            module_shapes,
        )
        # Rename keys for comparison against non-fsdp
        test_flex_dict_renamed = {}
        for k, v in test_flex_dict.items():
            elements = k.split(".")
            new_name = []
            for e in elements:
                if e != "_fsdp_wrapped_module":
                    new_name.append(e)
            test_flex_dict_renamed[".".join(new_name)] = v
        return test_flex_dict_renamed

    test_base_dict = _get_torch_output()
    test_flex_dict = _get_flex_output()

    if len(test_flex_dict) == 0:
        return

    def _print_dict(d):
        for n, m in d.items():
            print_rank0(f"{n}: {m.shape}")

    print("*" * 50)
    _print_dict(test_base_dict)
    print("*" * 50)
    _print_dict(test_flex_dict)
    print("*" * 50)

    assert compare_tensor_dicts(test_base_dict, test_flex_dict)

    logger.info("Test successful!")


# TODO: Be more consistent with logging messages
def main():
    logger.info("Testing simple PyTorch model...")
    test_simple_model()
    logger.info("Test successful!")

    logger.info("Testing traversal/parse utils...")
    test_traversal_utils()
    logger.info("Test successful!")

    logger.info("Testing Huggingface OPT-125m single gpu...")
    test_huggingface_opt_model()
    logger.info("Test successful!")

    logger.info("Testing Huggingface llama-7b dual gpu...")
    test_distributed_flex_model()


if __name__ == "__main__":
    main()
