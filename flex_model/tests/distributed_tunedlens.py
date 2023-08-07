from dataclasses import dataclass
import logging
import math
from typing import List, Dict, Tuple, Callable, Optional

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from flex_model.model_wrappers import FlexModel, HookFunction


LENS_MODEL_PARALLEL_GROUP = None
logger = logging.getLogger(__name__)


def init_lens_model_parallel(
    lens_model_parallel_size: int = 1
) -> None:
    global LENS_MODEL_PARALLEL_GROUP
    assert LENS_MODEL_PARALLEL_GROUP is None

    ranks = list(range(lens_model_parallel_size))
    assert dist.get_world_size() >= len(ranks)

    LENS_MODEL_PARALLEL_GROUP = dist.new_group(
        ranks=ranks,
        backend="nccl",
    )


def get_lens_model_parallel_group():
    global LENS_MODEL_PARALLEL_GROUP
    assert LENS_MODEL_PARALLEL_GROUP is not None
    return LENS_MODEL_PARALLEL_GROUP


def get_lens_model_parallel_world_size():
    return dist.get_world_size(group=get_lens_model_parallel_group())


def get_lens_model_parallel_rank():
    return dist.get_rank(group=get_lens_model_parallel_group())


def print_gpu_dram():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    percent_used = int(allocated / reserved * 100)
    allocated_rounded = round(allocated / 1_000_000_000, 2)
    rank = get_lens_model_parallel_rank()
    logger.warning(f"Rank{rank}: GPU mem used: {allocated_rounded}G - {percent_used}%")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kl_divergence(logits, preds, mask):
    logits = logits.to(torch.float32)
    preds = preds.to(torch.float32)

    logits = torch.log_softmax(logits, dim=-1)
    kl_div = torch.sum(
        (torch.exp(logits) * (logits - torch.log_softmax(preds, dim=-1))) * mask,
        dim=-1,
    )
    return torch.mean(kl_div)


def kl_divergence_streamed(logits, preds, mask, chunks=20):
    logits = logits.to(torch.float32)
    preds = preds.to(torch.float32)

    assert preds.shape[0] % chunks == 0

    preds = torch.chunk(preds, chunks)
    logits = torch.log_softmax(logits, dim=-1)

    kl_div = 0
    for pred_chunk in preds:
        partial_kl_div = torch.sum(
            (torch.exp(logits) * (logits - torch.log_softmax(pred_chunk, dim=-1))) * mask,
            dim=-1,
        ).mean()
        kl_div += partial_kl_div

    kl_div = kl_div / chunks
    return kl_div


def kl_divergence_with_pp(p, q, mask=None):
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


class Norm(nn.Module):
    def __init__(self, norm_weight: Tensor) -> None:
        super().__init__()
        self.norm_weight = norm_weight

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return out.type_as(x) * self.norm_weight


class Unembed(nn.Module):
    def __init__(self, unembed_weight: Tensor) -> None:
        super().__init__()
        self.unembed_weight = unembed_weight

    def forward(self, x):
        out = torch.einsum("lbsh,vh->lbsv", x, self.unembed_weight)
        return out


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

    def partial_forward(self, layer_activations, indices):
        layer_activations.to(torch.bfloat16)
        out = torch.einsum(
            "lbsh,lhH->lbsH",
            layer_activations,
            self.translators[indices],
        )
        out = out + self.bias[indices]
        return out

    def forward(self, layer_activations):
        layer_activations.to(torch.bfloat16)
        out = torch.einsum(
            "lbsh,lhH->lbsH",
            layer_activations,
            self.translators,
        )
        out = out + self.bias
        return out


class DistributedTunedLens(nn.Module):
    def __init__(
        self,
        frozen_model: nn.Module,
        vocab_size: int,
        hidden_dim: int,
        lens_model_parallel_size: int = 1,
        layers_prefix: str = "layers",
        unembed_prefix: str = "output.weight",
        norm_prefix: str = "norm.weight",
        dtype: str = "bf16",
    ) -> None:
        super().__init__()
        # Init constants
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.lens_model_parallel_size = lens_model_parallel_size
        self.total_layers = len(getattr(frozen_model, layers_prefix))
        self.all_layers = [
            f"{layers_prefix}.{i}" for i in range(self.total_layers)
        ]
        logger.info(f"DistributedTunedLens: MP={self.lens_model_parallel_size}, "
                    f"total_layers={self.total_layers}, "
                    f"all_layers={self.all_layers}, ")

        # Init model parallel group
        init_lens_model_parallel(lens_model_parallel_size)

        # Init subset of layers for lens model parallel group
        stride = self.total_layers // lens_model_parallel_size
        bottom = get_lens_model_parallel_rank() * stride 
        top = bottom + stride
        self.layers = [f"{layers_prefix}.{i}" for i in range(bottom, top)]
        self.num_layers = len(self.layers)
        logger.info(f"Rank{get_lens_model_parallel_rank()}: {self.num_layers} "
                    f"layers - {self.layers}")

        # Canonicalize dtypes
        if dtype == "bf16":
            self.dtype = torch.bfloat16
        elif dtype == "fp16":
            self.dtype = torch.float16
        elif dtype == "fp32":
            self.dtype = torch.float32
        else:
            raise Exception
        
        # Init hooked model
        self.activation_dict: Dict[str, Tensor] = {}
        self.frozen_model = FlexModel(frozen_model, self.activation_dict)
        self.frozen_model.requires_grad_(False)
        for layer_name in self.all_layers:
            self.frozen_model.register_hook_function(
                HookFunction(
                    layer_name,
                    expected_shape=(None, None, self.hidden_dim),
                    editing_function=lambda x, _: x,
                ),
            )

        # Init unembedding
        unembed_weight = self.frozen_model.get_module_parameter(
            unembed_prefix,
            (self.vocab_size, self.hidden_dim),
        ).cuda().to(self.dtype)
        self.unembed = Unembed(unembed_weight)
        self.unembed.requires_grad_(False)

        # Init layernorm
        norm_weight = self.frozen_model.get_module_parameter(
            norm_prefix,
            (self.hidden_dim,),
        ).cuda().to(self.dtype)
        self.norm = Norm(norm_weight)
        self.norm.requires_grad_(False)

        # Init affine translators
        self.translators = Translators(
            hidden_dim=self.hidden_dim,
            num_layers = self.num_layers,
        ).cuda()

        logger.info(f"Initialized DistributedTunedLens: {self}")
        print(f"Parameter count: {count_parameters(self)}")

    def _convert_layer_dict_to_list(self, layer_dict: Dict[str, Tensor]) -> List[Tensor]:
        output = [None for _ in range(len(layer_dict))]

        for i, layer_name in enumerate(self.all_layers):
            output[i] = layer_dict.pop(layer_name)

        return output

    def scatter_activations(self):
        world_size = get_lens_model_parallel_world_size()
        if get_lens_model_parallel_rank() == 0:
            # List of activations indexed by layer index
            act_list = self._convert_layer_dict_to_list(self.activation_dict)

            # Need world_size # groups containint num_layers activations
            layer_group = 0
            objects = [[] for _ in range(world_size)]
            for act in act_list:
                # Move to next group of layers
                if len(objects[layer_group]) == self.num_layers:
                    layer_group += 1

                objects[layer_group].append(act)

        else:
            objects = [None for _ in range(world_size)]

        output_list = [None]

        dist.scatter_object_list(output_list, objects, src=0)

        # output_list: [[Tensor, ...]]
        output_list = output_list[0]
        if get_lens_model_parallel_rank() == 0:
            logger.info(f"Rank{get_lens_model_parallel_rank()} Scatter: "
                        f"[{len(objects) * len(objects[0])}] -> [{len(output_list)}]")


        return output_list

    def streamed_loss(self, batch, loss_fn, chunks=4):
        with torch.no_grad():
            logits = self.frozen_model.forward(batch.cuda(), start_pos=0)

        activations = self.scatter_activations()
        activations = torch.stack([act for act in activations], dim=0)
        activations = activations.cuda().to(self.dtype)

        # Chunk activations along layer dim
        chunk_size = self.num_layers // chunks
        activations = torch.chunk(activations, chunks, dim=0)

        # Create loss mask
        mask = (batch != 0)[None, :, :, None].cuda()

        # Stream the loss calculation to avoid expanding vocab size activation
        # across num_layers
        loss = 0
        for i, act_chunk in enumerate(activations):
            # Set of layers to unembed
            start = i * chunk_size
            end = start + chunk_size
            indices = list(range(start, end))

            pred_logits = self.translators.partial_forward(act_chunk, indices)
            pred_logits = self.norm(pred_logits)
            pred_logits = self.unembed(pred_logits)

            loss += loss_fn(logits, pred_logits, mask)

        loss = loss / chunks
        return loss

    def forward(self, batch):
        with torch.no_grad():
            logits = self.frozen_model.forward(batch.cuda(), start_pos=0)

        activations = self.scatter_activations()
        activations = torch.stack([act for act in activations], dim=0)
        activations = activations.cuda().to(self.dtype)

        pred_logits = self.translators(activations)
        pred_logits = self.norm(pred_logits)
        pred_logits = self.unembed(pred_logits)

        return logits, pred_logits


@dataclass
class DistributedTunedLensTrainerConfig:
    optimizer_type: torch.optim.Optimizer = torch.optim.SGD
    use_scheduler: bool = True
    batch_size: int = 8
    lr_warmup_steps: int = 0
    total_num_steps: int = 250
    train_steps_per_val: int = 20
    val_steps: int = 50
    log_interval: int = 5
    lr_scale: float = 1.0
    momentum: float = 0.9
    weight_decay: float = 1e-3
    clip: float = 1.0
    loss_fn: Callable = kl_divergence


class DistributedTunedLensTrainer:
    def __init__(
        self,
        distributed_tuned_lens: DistributedTunedLens,
        config: DistributedTunedLensTrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        self.distributed_tuned_lens = distributed_tuned_lens
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_datalaoder = test_dataloader

        self.parse_config()

    def parse_config(self) -> None:
        # TODO: Only works for SGD right now
        self.optimizer = self.config.optimizer_type(
            self.distributed_tuned_lens.parameters(),
            lr=self.config.lr_scale * (1 - self.config.momentum),
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=True,
        )
        if self.config.use_scheduler:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.lr_warmup_steps,
                num_training_steps=self.config.total_num_steps,
            )
        else:
            self.scheduler = None

    def train(self):
        self.distributed_tuned_lens.train()

        train_dataloader = iter(self.train_dataloader)

        interval_loss = 0
        for step in tqdm(range(self.config.total_num_steps)):
            if step != 0 and step % self.config.train_steps_per_val == 0:
                self.val()

            batch = next(train_dataloader)
            loss = self.train_step(batch)

            interval_loss += loss

            if step != 0 and step % self.config.log_interval == 0:
                print(f"Rank{get_lens_model_parallel_rank()} loss at {step}: {interval_loss / self.config.log_interval}")
                interval_loss = 0
            
    def train_step(self, batch):
        self.optimizer.zero_grad()

        """
        logits, pred_logits = self.distributed_tuned_lens(batch)

        # Mask is bs -> lbsv
        mask = (batch != 0).cuda()
        mask = mask[None, :, :, None]

        loss = self.config.loss_fn(logits, pred_logits, mask=mask)
        """
        loss = self.distributed_tuned_lens.streamed_loss(
            batch,
            loss_fn=self.config.loss_fn,
            chunks=4,
        )
        assert loss.isfinite()

        loss.backward()

        nn.utils.clip_grad_norm_(self.distributed_tuned_lens.parameters(), self.config.clip)

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def val(self):
        self.distributed_tuned_lens.eval()

        val_dataloader = iter(self.val_dataloader)

        val_loss = 0
        for step in tqdm(range(self.config.val_steps)):
            batch = next(val_dataloader)
            loss = self.val_step(batch)
            val_loss += loss.item()

        val_loss = val_loss / self.config.val_steps
        print(f"Rank{get_lens_model_parallel_rank()} val loss at {step}: {val_loss}")

    def val_step(self, batch):
        with torch.no_grad():
            loss = self.distributed_tuned_lens.streamed_loss(
                batch,
                loss_fn=self.config.loss_fn,
                chunks=4,
            )
        return loss
