# Set constants
_PROMPTS = [
    "Hi I'm Matt, where am I?",
    "Welcome to Vector",
    "The tensor has a shape of",
]

_LLAMA_VANILLA_TORCH_MODULE_NAMES = [
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
_LLAMA_VANILLA_TORCH_MODULE_SHAPES = [
    None for n in range(len(_LLAMA_VANILLA_TORCH_MODULE_NAMES))
]

_FSDP_PREFIX = "_fsdp_wrapped_module."
_LLAMA_FSDP_MODULE_NAMES = [
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
_LLAMA_FSDP_MODULE_SHAPES = [
    (3, 11, 5120) for n in range(len(_LLAMA_FSDP_MODULE_NAMES))
]
_LLAMA_FSDP_MODULE_SHAPES[2] = (3, 11, 13824)
_LLAMA_FSDP_MODULE_SHAPES[8] = (3, 11, 13824)
_LLAMA_FSDP_MODULE_SHAPES[9] = (3, 11, 13824)
_LLAMA_FSDP_MODULE_SHAPES[-1] = (3, 11, 32000)
