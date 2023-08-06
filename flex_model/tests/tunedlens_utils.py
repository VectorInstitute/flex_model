import argparse
import logging
import math

from tqdm import tqdm
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
        batch_size,
        warmup_steps,
        num_steps,
        train_steps,
        eval_steps,
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
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.lr_scale = lr_scale
        self.momentum = momentum
        self.wd = wd
        self.clip = clip

        unembed = self.make_unembed()
        rms_norm = self.make_rms_norm()
        if dist.get_rank() == 0:
            self.unembed = unembed.to(torch.bfloat16).cuda()
            self.rms_norm = rms_norm.to(torch.bfloat16).cuda()
            self.lens = self.make_lens()
            self.optim = self.make_optimizer()
            self.scheduler = self.make_scheduler()

        self.train_dataloader, self.eval_dataloader = self.make_dataloaders()

    def make_loss_mask(self, batch):
        # 0 is the pad id for llama
        return (batch != 0)

    def make_unembed(self):
        unembed = self.model.get_module_parameter(
            "output.weight",
            (self.vocab_size, self.hidden_dim),
        )
        return unembed

    def make_rms_norm(self):
        rms_norm = self.model.get_module_parameter(
            "norm.weight",
            (self.hidden_dim,),
        )
        return rms_norm

    def make_dataloaders(self):
        def collate_fn(examples):
            examples = [
                ex["text"] for ex in examples
            ]
            return self.tokenize_fn(examples)

        dataset = load_dataset("wikitext", "wikitext-103-v1")
        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=collate_fn,
        )
        eval_dataloader = DataLoader(
            dataset["validation"],
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=collate_fn,
        )
        return train_dataloader, eval_dataloader

    def make_optimizer(self):
        return torch.optim.SGD(
            self.lens.parameters(),
            lr=self.lr_scale * (1 - self.momentum),
            momentum=self.momentum,
            weight_decay=self.wd,
            nesterov=True,
        )

    def make_scheduler(self):
        return get_linear_schedule_with_warmup(
            self.optim,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_steps,
        )

    def make_lens(self):
        return LlamaLens(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            unembed=self.unembed,
            rms_norm=self.rms_norm,
        ).cuda()

    def train_step(self, batch, step):
        with torch.no_grad():
            logits = self.model.forward(batch.cuda(), start_pos=0)

        if dist.get_rank() == 0:
            self.optim.zero_grad()

            activations = torch.stack([
                self.activation_dict.pop(layer)
                for layer in self.layers
            ], dim=0).cuda()

            mask = self.make_loss_mask(batch).cuda()

            loss, layer_pp = self.lens.get_loss(activations, logits, mask)
            assert loss.isfinite()

            loss.backward()

            nn.utils.clip_grad_norm_(self.lens.parameters(), self.clip)

            self.optim.step()
            self.scheduler.step()

            print(">" * 4 + f"Loss at iteration {step}: {loss.item()}")
            print(f"Layer perplexity at iteration {step}: {layer_pp}")

        dist.barrier()

    def train(self):
        if dist.get_rank() == 0:
            self.lens.train()

        train_dataloader = iter(self.train_dataloader)

        for step in range(self.num_steps):
            print(f"GPU usage: {torch.cuda.memory_allocated(0) / 1_000_000_000}G")
            if step % self.train_steps == 0:
                self.eval()
                if dist.get_rank() == 0:
                    self.lens.train()

            batch = next(train_dataloader)
            self.train_step(batch, step)

    def eval_step(self, batch):
        with torch.no_grad():
            logits = self.model.forward(batch.cuda(), start_pos=0)

        if dist.get_rank() == 0:
            activations = torch.stack([
                self.activation_dict.pop(layer)
                for layer in self.layers
            ], dim=0).cuda()

            mask = self.make_loss_mask(batch).cuda()

            with torch.no_grad():
                loss, _ = self.lens.get_loss(activations, logits, mask)

        else:
            loss = 0

        dist.barrier()

        return loss

    def eval(self):
        if dist.get_rank() == 0:
            self.lens.eval()

        eval_loss = 0
        eval_dataloader = iter(self.eval_dataloader)

        for step in tqdm(range(self.eval_steps)):
            batch = next(eval_dataloader)
            loss = self.eval_step(batch)
            eval_loss += loss

        if dist.get_rank() == 0:
            eval_loss /= self.eval_steps
            print(f"Eval loss: {eval_loss}")


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


class Translators(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Translators init to identity
        linear = torch.eye(hidden_dim, dtype=torch.bfloat16)[None, :]
        linear = linear.tile((num_layers, 1, 1))
        self.translators = nn.parameter.Parameter(linear, requires_grad=True)

        # Bias init like regular affine layer bias
        stdv = 1. / math.sqrt(self.translators.size(-1))
        bias = torch.zeros((num_layers, 1, 1, hidden_dim), dtype=torch.bfloat16)
        self.bias = nn.parameter.Parameter(bias, requires_grad=True)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, layer_activations):
        layer_activations.to(torch.bfloat16)
        out = torch.einsum(
            "lbsh,lhH->lbsH",
            layer_activations,
            self.translators,
        )
        out += self.bias
        return out


class LlamaLens(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        unembed,
        rms_norm,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.translators = Translators(
            self.hidden_dim,
            self.num_layers,
        )
        self.unembed = unembed
        self.rms_norm = rms_norm

    def norm(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return out.type_as(x) * self.rms_norm

    def forward(self, layer_activations):
        layer_activations = layer_activations.to(torch.bfloat16)
        pred_logits = self.translators(layer_activations)
        pred_logits = self.norm(pred_logits)
        pred_logits = torch.einsum(
            "lbsh,vh->lbsv",
            pred_logits,
            self.unembed,
        )
        return pred_logits
       
    def get_loss(self, activations, true_logits, mask):
        pred_logits = self.forward(activations)

        mask = mask[None, :, :, None]

        kl_div, layer_perplexity = kl_divergence(true_logits, pred_logits)

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
        tokens = torch.full(
            (bsz, total_len),
            pad_id,
            dtype=torch.long,
            device="cuda",
        )
        for k, t in enumerate(input_tokens):
            seq_len = min(len(t), max_seq_len)
            tokens[k, : seq_len] = torch.tensor(
                t[:seq_len], dtype=torch.long, device="cuda"
            )
        return tokens

    return model, tokenize
