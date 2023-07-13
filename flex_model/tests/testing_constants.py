# Set constants
_PROMPTS = [
    "Hi I'm Matt, where am I?",
    "Welcome to Vector",
    "The tensor has a shape of",
]

# NOTE: original llama impl uses functional silu impl, so no act created
_LLAMA_VANILLA_TORCH_MODULES = {
    "model.layers.5.self_attn": None,
    "model.layers.6.self_attn.o_proj": None,
    "model.layers.7.self_attn.v_proj": None,
    "model.layers.11.self_attn.k_proj": None,
    "model.layers.8.self_attn.q_proj": None,
    "model.layers.11.post_attention_layernorm": None,
    "model.layers.7.mlp": None,
    "model.layers.7.mlp.gate_proj": None,
    "model.layers.28.mlp.up_proj": None,
    "model.layers.9.mlp.down_proj": None,
    "model.embed_tokens": None,
    "model.layers.7": None,
    "lm_head": None,
}

_FSDP_PREFIX = "_fsdp_wrapped_module."
_LLAMA_FSDP_MODULES = {
    "_fsdp_wrapped_module.model.layers.5._fsdp_wrapped_module.self_attn": (3, 11, 5120),
    "_fsdp_wrapped_module.model.layers.6._fsdp_wrapped_module.self_attn.o_proj": (3, 11, 5120),
    #"_fsdp_wrapped_module.model.layers.2._fsdp_wrapped_module.mlp.act_fn",
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.self_attn.v_proj": (3, 11, 40, 5120),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.self_attn.k_proj": (3, 11, 5120),
    "_fsdp_wrapped_module.model.layers.8._fsdp_wrapped_module.self_attn.q_proj": (3, 11, 5120),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.post_attention_layernorm": (3, 11, 5120),
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.mlp": (3, 11, 5120),
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.mlp.gate_proj": (3, 11, 13824),
    "_fsdp_wrapped_module.model.layers.28._fsdp_wrapped_module.mlp.up_proj": (3, 11, 13824),
    "_fsdp_wrapped_module.model.layers.9._fsdp_wrapped_module.mlp.down_proj": (3, 11, 5120),
    "_fsdp_wrapped_module.model.embed_tokens": (3, 11, 5120),
    #"_fsdp_wrapped_module.model": (3, 11, 5120),
    "_fsdp_wrapped_module.model.layers.7": (3, 11, 5120),
    "_fsdp_wrapped_module.lm_head": (3, 11, 32000),

}

_LLAMA_MEGATRON_MODULES = {
    "layers.5.attention": (3, 11, 5120),
    "layers.6.attention.wo": (3, 11, 5120),
    "layers.7.attention.wv": (3, 11, 5120),
    "layers.11.attention.wk": (3, 11, 5120),
    "layers.8.attention.wq": (3, 11, 5120),
    "layers.11.ffn_norm": (3, 11, 5120),
    "layers.7.feed_forward": (3, 11, 5120),
    "layers.7.feed_forward.w1": (3, 11, 13824),
    "layers.28.feed_forward.w3": (3, 11, 13824),
    "layers.9.feed_forward.w2": (3, 11, 5120),
    "tok_embeddings": (3, 11, 5120),
    #"output": (3, 11, 13824),
    "layers.7": (3, 11, 5120),
    "output": (3, 11, 32000),
}
