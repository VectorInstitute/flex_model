import math

from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from flex_model.model_wrappers import FlexModel, HookFunction
from flex_model.tests.testing_utils import get_llama_13b_megatron
from flex_model.tests.testing_constants import _PROMPTS


def assert_no_nan(tensor):
    assert torch.isnan(tensor).sum() == 0


def kl_divergence(p, q, mask=None):
    log_p = torch.log_softmax(p, dim=-1)
    p = torch.exp(log_p)

    log_q = torch.log_softmax(q, dim=-1)
    kl_div = p * (log_p - log_q)

    if mask is not None:
        kl_div *= mask

    kl_div = torch.mean(torch.sum(kl_div, dim=-1))
    return kl_div


LAYERS = [
    f"layers.{i}"
    for i in range(40)
]


class LlamaLens(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lenses_A = nn.parameter.Parameter(torch.eye(dim)[None, :].tile((len(LAYERS), 1, 1)), requires_grad=True)

        stdv = 1. / math.sqrt(self.lenses_A.size(-1))
        self.lenses_b = nn.parameter.Parameter(torch.zeros((len(LAYERS), 1, 1, dim)), requires_grad=True)
        self.lenses_b.data.uniform_(-stdv, stdv)

    def forward(self, activations):
        return torch.einsum("lbsh,lhH->lbsH", activations, self.lenses_A) + self.lenses_b

    def get_loss(self, activations, unembed, true_logits, mask):
        pseudo_logits = self.forward(activations)
        pseudo_logits = torch.einsum("lbsh,vh->lbsv", pseudo_logits, unembed)
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


def main():
    output_dict = {}
    model, tokenize_fn = get_llama_13b_megatron()
    model = FlexModel(model, output_dict)

    # Hook layers into model

    for layer in LAYERS:
        hf = HookFunction(
            layer,
            expected_shape=(None, None, 5120),
            editing_function=lambda x, _: x,
        )
        model.register_hook_function(hf)

    dataloader = make_dataloader(tokenize_fn)

    # Init lenses and their weights, init optim
    if dist.get_rank() == 0:
        lens = LlamaLens(dim=5120).cuda()
        optim = torch.optim.SGD(lens.parameters(), lr=3e-3, momentum=0, weight_decay=0)

    # Training loop
    for i, batch in enumerate(dataloader):
        logits = model.forward(batch.cuda(), start_pos=0)
        unembed = model.get_module_parameter("output.weight", (32000, 5120)).cuda()

        if dist.get_rank() == 0:
            optim.zero_grad()

            activations = torch.stack([
                output_dict[layer]
                for layer in LAYERS
            ], dim=0).cuda()

            mask = make_loss_mask(batch).cuda()

            loss = lens.get_loss(activations, unembed, logits, mask)

            loss.backward()
            optim.step()

            print(">" * 4 + f"Loss at iteration {i}: {loss.item()}")

        dist.barrier()


if __name__ == "__main__":
    main()
