import logging

import torch
import torch.nn as nn
import torch.distributed as dist


from flex_model.core import FlexModel
from flex_model.tunedlens.distributed_tunedlens import (
    Translators,
    DistributedTunedLens,
    kl_divergence,
)
import flex_model.tunedlens.distributed as tl_dist
from tests.test_utilities import Utils
from flex_model.tunedlens.test_tunedlens import make_llama2_model


def test_initialize_lens_model_parallel():
    # TODO: Separate this into own file, need to init dist here
    tl_dist.initialize_lens_model_parallel(dist.get_world_size())
    assert tl_dist.is_initialized()
    assert tl_dist.get_lens_model_parallel_group() is not None
    assert tl_dist.get_lens_model_parallel_world_size() == dist.get_world_size()
    assert tl_dist.get_lens_model_parallel_rank() == dist.get_rank()

    tl_dist.destroy_lens_model_parallel()
    assert not tl_dist.is_initialized()
    assert tl_dist.get_lens_model_parallel_group() is None


def test_Translators():
    num_layers = 4
    batch_size = 2
    seq_len = 128
    hidden_dim = 32
    translator = Translators(hidden_dim, num_layers).cuda()
    # LBSH
    inputs = (
        torch.randn((num_layers, batch_size, seq_len, hidden_dim))
        .cuda()
        .to(torch.bfloat16)
    )
    for layer in translator.translators:
        assert layer.shape == (hidden_dim, hidden_dim)
        assert torch.equal(layer, torch.eye(hidden_dim).to(layer.dtype).cuda())

    output = translator(inputs)
    assert torch.equal(output, inputs + translator.bias)

    inputs = torch.randn((2, batch_size, seq_len, hidden_dim)).cuda().to(torch.bfloat16)
    layer_indices = [1, 3]
    output = translator.partial_forward(inputs, layer_indices)
    assert torch.equal(output, inputs + translator.bias[layer_indices])


def test_DistributedTunedLens():
    Utils.initialize_distributed()

    model, tokenize_fn = make_llama2_model(
        "/ssd005/projects/llm/llama-2-13b",
        "/ssd005/projects/llm/llama-2-13b/tokenizer.model",
        128,
        4,
    )

    dtl = DistributedTunedLens(
        model,
        vocab_size=32000,
        hidden_dim=5120,
        lens_model_parallel_size=dist.get_world_size(),
    )

    # Test activation scatter
    inputs = torch.randint(0, 32000, (4, 128)).cuda()
    with torch.no_grad():
        _ = dtl.frozen_model(inputs, start_pos=0)

    rank_activations = dtl.scatter_activations()
    assert len(rank_activations) == len(model.layers) // 2

    # Test streamed loss vs. regular forward + loss
    streamed_loss = dtl.streamed_loss(inputs, kl_divergence)
    logits, pred_logits = dtl(inputs)
    mask = (inputs != 0)[None, :, :, None]
    regular_loss = kl_divergence(logits, pred_logits, mask)
    assert torch.allclose(streamed_loss, regular_loss)

    tl_dist.destroy_lens_model_parallel()


test_DistributedTunedLens()
test_Translators()
test_initialize_lens_model_parallel()
