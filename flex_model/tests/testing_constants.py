# Set constants
_PROMPTS = [
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    """Translate English to French:
    
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>""",
]

# NOTE: original llama impl uses functional silu impl, so no act created
_LLAMA_VANILLA_TORCH_MODULES = {
    "model.layers.5.self_attn": None,
    "model.layers.20.self_attn": None,
    "model.layers.39.self_attn": None,
    "model.layers.6.self_attn.o_proj_dummy": None,
    "model.layers.7.self_attn.v_proj_dummy": None,
    "model.layers.11.self_attn.k_proj_dummy": None,
    "model.layers.8.self_attn.q_proj_dummy": None,
    "model.layers.11.post_attention_layernorm": None,
    "model.layers.32.mlp": None,
    "model.layers.7.mlp.gate_proj_dummy": None,
    "model.layers.28.mlp.up_proj_dummy": None,
    "model.layers.9.mlp.down_proj_dummy": None,
    "model.embed_tokens": None,
    "model.layers.7": None,
    "lm_head_dummy": None,
}

_FSDP_PREFIX = "_fsdp_wrapped_module."
_LLAMA_FSDP_MODULES = {
    "_fsdp_wrapped_module.model.layers.5._fsdp_wrapped_module.self_attn": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.20._fsdp_wrapped_module.self_attn": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.39._fsdp_wrapped_module.self_attn": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.6._fsdp_wrapped_module.self_attn.o_proj_dummy": (None, None, 5120),
    #"_fsdp_wrapped_module.model.layers.2._fsdp_wrapped_module.mlp.act_fn",
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.self_attn.v_proj_dummy": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.self_attn.k_proj_dummy": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.8._fsdp_wrapped_module.self_attn.q_proj_dummy": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.11._fsdp_wrapped_module.post_attention_layernorm": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.32._fsdp_wrapped_module.mlp": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.7._fsdp_wrapped_module.mlp.gate_proj_dummy": (None, None, 13824),
    "_fsdp_wrapped_module.model.layers.28._fsdp_wrapped_module.mlp.up_proj_dummy": (None, None, 13824),
    "_fsdp_wrapped_module.model.layers.9._fsdp_wrapped_module.mlp.down_proj_dummy": (None, None, 5120),
    "_fsdp_wrapped_module.model.embed_tokens": (None, None, 5120),
    #"_fsdp_wrapped_module.model": (None, None, 5120),
    "_fsdp_wrapped_module.model.layers.7": (None, None, 5120),
    "_fsdp_wrapped_module.lm_head_dummy": (None, None, 32000),

}

_LLAMA_MEGATRON_MODULES = {
    "layers.5.attention": (None, None, 5120),
    "layers.20.attention": (None, None, 5120),
    "layers.39.attention": (None, None, 5120),
    "layers.6.attention.wo": (None, None, 5120),
    "layers.7.attention.wv": (None, None, 5120),
    "layers.11.attention.wk": (None, None, 5120),
    "layers.8.attention.wq": (None, None, 5120),
    "layers.11.ffn_norm": (None, None, 5120),
    "layers.32.feed_forward": (None, None, 5120),
    "layers.7.feed_forward.w1": (None, None, 13824),
    "layers.28.feed_forward.w3": (None, None, 13824),
    "layers.9.feed_forward.w2": (None, None, 5120),
    "tok_embeddings": (None, None, 5120),
    #"output": (None, None, 13824),
    "layers.7": (None, None, 5120),
    "output": (None, None, 32000),
}
