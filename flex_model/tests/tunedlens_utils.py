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
from transformers import get_linear_schedule_with_warmup

from flex_model.model_wrappers import FlexModel, HookFunction, setup_logger


logger = logging.getLogger(__name__)


class LensTrainer:
    def __init__(
        self,
        hidden_dim,
        num_layers,
        vocab_size,
        layers,
        model,
        tokenize_fn,
        activation_dict,
        warmup_steps,
        num_steps,
        train_steps_per_eval,
        lr_scale,
        momentum,
        wd,
        clip,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.layers = layers
        self.model = model
        self.tokenize_fn = tokenize_fn
        self.activation_dict = activation_dict
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.train_steps_per_eval = train_steps_per_eval
        self.lr_scale = lr_scale
        self.momentum = momentum
        self.wd = wd
        self.clip = clip

    def make_loss_mask(self, inputs):
        # 0 is the pad id for llama
        return (inputs != 0)

    def make_dataloaders(self):
        def collate_fn(examples):
            examples = [
                ex["text"] for ex in examples
            ]
            return self.tokenize_fn(examples)

        dataset = load_dataset("wikitext", "wikitext-103-v1")
        train_dataloader = DataLoader(dataset["train"], batch_size=8, num_workers=0, collate_fn=collate_fn)
        eval_dataloader = DataLoader(dataset["validation"], batch_size=8, num_workers=0, collate_fn=collate_fn)
        return train_dataloader, eval_dataloader

    def make_optimizer(self, lens):
        return torch.optim.SGD(
            lens.parameters(),
            lr=self.lr_scale * (1 - self.momentum),
            momentum=self.momentum,
            weight_decay=self.wd,
            nesterov=True,
        )

    def make_scheduler(self, optim):
        return get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_steps,
        )

    def make_lens(self):
        return LlamaLens(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).cuda()

    def train_step(self, step, lens, batch, optim, scheduler, unembed, rms_norm):
        with torch.no_grad():
            logits = self.model.forward(batch.cuda(), start_pos=0)

        if dist.get_rank() == 0:
            optim.zero_grad()

            activations = torch.stack([
                self.activation_dict.pop(layer)
                for layer in self.layers
            ], dim=0).cuda()

            mask = self.make_loss_mask(batch).cuda()

            loss, layer_pp = lens.get_loss(activations, unembed, rms_norm, logits, mask)
            assert loss.isfinite()

            loss.backward()

            nn.utils.clip_grad_norm_(lens.parameters(), self.clip)

            optim.step()

            print(">" * 4 + f"Loss at iteration {step}: {loss.item()}")
            print(f"Layer perplexity at iteration {step}: {layer_pp}")

        dist.barrier()


    def eval_step(self):
        pass

    def train(self):
        train_dataloader, eval_dataloader = self.make_dataloaders()
        train_dataloader = iter(train_dataloader)
        eval_dataloader = iter(eval_dataloader)

        if dist.get_rank() == 0:
            lens = self.make_lens()
            optim = self.make_optimizer(lens)
            scheduler = self.make_scheduler(optim)
        else:
            optim = None
            scheduler = None
            lens = None

        unembed = self.model.get_module_parameter("output.weight", (self.vocab_size, self.hidden_dim))
        rms_norm = self.model.get_module_parameter("norm.weight", (self.hidden_dim,))

        if dist.get_rank() == 0:
            unembed = unembed.to(torch.bfloat16).cuda()
            rms_norm = rms_norm.to(torch.bfloat16).cuda()
        else:
            unembed = None
            rms_norm = None

        for i in range(self.num_steps):
            batch = next(train_dataloader)
            self.train_step(
                i,
                lens,
                batch,
                optim,
                scheduler,
                unembed,
                rms_norm,
            )


def kl_divergence(p, q, mask=None):
    p = p.to(torch.float32)
    q = q.to(torch.float32)
    log_p = torch.log_softmax(p, dim=-1)
    p = torch.exp(log_p)

    log_q = torch.log_softmax(q, dim=-1)
    kl_div = p * (log_p - log_q)

    if mask is not None:
        kl_div *= mask

    kl_div = torch.sum(kl_div, dim=-1)
    layer_perplexity = torch.exp(torch.mean(kl_div, dim=(1, 2))).detach().cpu()

    kl_div = torch.mean(kl_div)

    return kl_div, layer_perplexity


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
        kl_div, layer_perplexity = kl_divergence(true_logits, pseudo_logits)

        return kl_div, layer_perplexity


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
        input_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        bsz = len(input_tokens)
        total_len = min(max(len(t) for t in input_tokens), max_seq_len)
        pad_id = 0
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(input_tokens):
            seq_len = min(len(t), max_seq_len)
            tokens[k, : seq_len] = torch.tensor(t[:seq_len], dtype=torch.long, device="cuda")
        return tokens

    return model, tokenize
