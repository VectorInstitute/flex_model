import argparse
import logging
import math

from datasets import load_dataset
from llama import Llama
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from flex_model.model_wrappers import FlexModel, HookFunction, setup_logger


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--checkpoint_dir", type=str, default="/ssd005/projects/llm/llama-2-13b")
    parser.add_argument("--tokenizer_path", type=str, default="/ssd005/projects/llm/llama-2-13b/tokenizer.model")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--early_stopping", type=int, default=20)
    args = parser.parse_args()
    return args


def kl_divergence(p, q, mask=None):
    p = p.to(torch.float32)
    q = q.to(torch.float32)
    log_p = torch.log_softmax(p, dim=-1)
    p = torch.exp(log_p)

    log_q = torch.log_softmax(q, dim=-1)
    kl_div = p * (log_p - log_q)

    if mask is not None:
        kl_div *= mask

    kl_div = torch.mean(torch.sum(kl_div, dim=-1))

    return kl_div


class LlamaLens(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lenses_A = nn.parameter.Parameter(torch.eye(hidden_dim, dtype=torch.bfloat16)[None, :].tile((num_layers, 1, 1)), requires_grad=True)

        stdv = 1. / math.sqrt(self.lenses_A.size(-1))
        self.lenses_b = nn.parameter.Parameter(torch.zeros((num_layers, 1, 1, hidden_dim), dtype=torch.bfloat16), requires_grad=True)
        self.lenses_b.data.uniform_(-stdv, stdv)

    def forward(self, activations):
        activations = activations.to(torch.bfloat16)
        return torch.einsum("lbsh,lhH->lbsH", activations, self.lenses_A) + self.lenses_b

    def get_loss(self, activations, unembed, rms_norm, true_logits, mask):
        def norm(x):
            _norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            return _norm.type_as(x) * rms_norm

        # Apply translators
        activations = activations.to(torch.bfloat16)
        pseudo_logits = self.forward(activations)

        # Apply llama final layer norm
        pseudo_logits = norm(pseudo_logits)

        # Produce estimated logits
        pseudo_logits = torch.einsum("lbsh,vh->lbsv", pseudo_logits, unembed)

        if pseudo_logits.isnan().sum() != 0:
            breakpoint()

        # Calculate loss
        mask = mask[None, :, :, None]
        kl_div = kl_divergence(true_logits, pseudo_logits)

        return kl_div


def make_dataloader(tokenize_fn):
    def collate_fn(examples):
        examples = [
            ex["text"] for ex in examples
        ]
        return tokenize_fn(examples)

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    dataloader = DataLoader(dataset["train"], batch_size=8, num_workers=0, collate_fn=collate_fn)
    return dataloader


def make_loss_mask(inputs):
    # 0 is the pad id for llama
    return (inputs != 0)


def make_llama2_model(checkpoint_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(
        ckpt_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    model = generator.model
    tokenizer = generator.tokenizer

    def tokenize(prompts):
        input_tokens = [generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        bsz = len(input_tokens)
        total_len = max(len(t) for t in input_tokens)
        pad_id = 0
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(input_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        return tokens

    return model, tokenize


def main(args):
    # Setup logger to mute messages
    setup_logger(args.log_level)

    # Define activation output dictionary
    output_dict = {}

    # Create the model and tokenizer function
    model, tokenize_fn = make_llama2_model(
        args.checkpoint_dir,
        args.tokenizer_path,
        args.max_seq_len,
        args.max_batch_size,
    )
    model = FlexModel(model, output_dict)

    # Get model hparams from model and define layers to hook into
    hidden_dim = model.module.layers[0].dim
    vocab_size = model.module.vocab_size
    num_layers = model.module.n_layers // 2
    layers = [f"layers.{i}" for i in range(0, num_layers * 2, 2)]

    # Hook layers into model
    for layer in layers:
        hf = HookFunction(
            layer,
            expected_shape=(None, None, hidden_dim),
            editing_function=lambda x, _: x,
        )
        model.register_hook_function(hf)

    # Create dataloader
    dataloader = make_dataloader(tokenize_fn)

    # We will train out lenses only on the rank0 worker
    # Init lenses and their weights, init optim
    if dist.get_rank() == 0:
        lens = LlamaLens(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).cuda()
        optim = torch.optim.SGD(
            lens.parameters(),
            lr=3e-3,
            momentum=0.9,
            weight_decay=1e-3,
            nesterov=True,
        )
    
    # Training loop
    for i, batch in enumerate(dataloader):
        # NOTE: DEBUGGING PURPOSES
        if i == args.early_stopping:
            break

        # TODO (optimization): Set hooks
        # All ranks run forward pass and contribute to activation retrieval
        with torch.no_grad():
            logits = model.forward(batch.cuda(), start_pos=0)

        # Get the unembedding matrix
        unembed = model.get_module_parameter("output.weight", (vocab_size, hidden_dim))
        unembed = unembed.to(torch.bfloat16).cuda()
        rms_norm = model.get_module_parameter("norm.weight", (hidden_dim,))
        rms_norm = rms_norm.to(torch.bfloat16).cuda()
        
        # Only rank 0 trains the lenses
        if dist.get_rank() == 0:
            optim.zero_grad()

            # Stack activations that we retrieved into output dict
            activations = torch.stack([
                output_dict[layer]
                for layer in layers
            ], dim=0).cuda()

            # Mask out padded tokens
            mask = make_loss_mask(batch).cuda()

            loss = lens.get_loss(activations, unembed, rms_norm, logits, mask)
            assert loss.isfinite()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optim.step()

            print(">" * 4 + f"Loss at iteration {i}: {loss.item()}")

        dist.barrier()


if __name__ == "__main__":
    args = parse_args()
    main(args)
