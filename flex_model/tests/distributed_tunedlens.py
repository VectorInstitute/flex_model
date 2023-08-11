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
    """Initialize lens model parallel group."""
    global LENS_MODEL_PARALLEL_GROUP
    assert LENS_MODEL_PARALLEL_GROUP is None

    ranks = list(range(lens_model_parallel_size))
    assert dist.get_world_size() >= len(ranks)

    LENS_MODEL_PARALLEL_GROUP = dist.new_group(
        ranks=ranks,
        backend="nccl",
    )


def get_lens_model_parallel_group() -> dist.ProcessGroup:
    """Return the lens model parallel group."""
    global LENS_MODEL_PARALLEL_GROUP
    assert LENS_MODEL_PARALLEL_GROUP is not None
    return LENS_MODEL_PARALLEL_GROUP


def get_lens_model_parallel_world_size() -> int:
    """Return the lens model parallel group (world) size."""
    return dist.get_world_size(group=get_lens_model_parallel_group())


def get_lens_model_parallel_rank() -> int:
    """Return the lens model parallel rank."""
    return dist.get_rank(group=get_lens_model_parallel_group())


def print_gpu_dram() -> None:
    """Helper to print GPU DRAM stats."""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    percent_used = int(allocated / reserved * 100)
    allocated_rounded = round(allocated / 1_000_000_000, 2)
    rank = get_lens_model_parallel_rank()
    logger.warning(f"Rank{rank}: GPU mem used: {allocated_rounded}G - {percent_used}%")


def count_parameters(model: nn.Module) -> int:
    """Count the number of parameters in a torch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def kl_divergence(logits: Tensor, preds: Tensor, mask: Tensor) -> Tensor:
    """Compute the KL divergence between logits (p) and predicted logits (q).

    Compute the KL divergence between batched logits and predicted logits.
    Batches of sequences are also assumed padded, so a mask is applied to the
    loss.
    """
    logits = logits.to(torch.float32)
    preds = preds.to(torch.float32)

    logits = torch.log_softmax(logits, dim=-1)
    kl_div = torch.sum(
        (torch.exp(logits) * (logits - torch.log_softmax(preds, dim=-1))) * mask,
        dim=-1,
    )
    return torch.mean(kl_div)

def perplexity(logits: Tensor, preds: Tensor, mask: Tensor) -> Tensor:
    """Compute the perplexity score."""
    # TODO
    raise NotImplementedError


class Norm(nn.Module):
    """Default LayerNorm (RMSNorm) implementation from llama-2."""
    def __init__(self, norm_weight: Tensor) -> None:
        super().__init__()
        self.norm_weight = norm_weight

    def forward(self, x: Tensor) -> Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return out.type_as(x) * self.norm_weight


class Unembed(nn.Module):
    """Basic unembedding layer."""
    def __init__(self, unembed_weight: Tensor) -> None:
        super().__init__()
        self.unembed_weight = unembed_weight

    def forward(self, x: Tensor) -> Tensor:
        # (layer, batch, seq, hidden) x (vocab, hidden) -> (layer, batch, seq, hidden)
        out = torch.einsum("lbsh,vh->lbsv", x, self.unembed_weight)
        return out


class Translators(nn.Module):
    """Layer containing batched affine translators.

    In the original TunedLens the translators were contained within a list and
    operated on separately. For parallelism we instead choose to only have two
    parameter tensors (linear and bias) but with an extra leading dimension for
    the number of translators (ie. number of layers).
    """
    def __init__(self, hidden_dim: int, num_layers: int) -> None:
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

    def partial_forward(
        self,
        layer_activations: Tensor,
        indices: List[int]
    ) -> Tensor:
        """Forward pass on subset of translators given by indices."""
        layer_activations.to(torch.bfloat16)
        out = torch.einsum(
            "lbsh,lhH->lbsH",
            layer_activations,
            self.translators[indices],
        )
        out = out + self.bias[indices]
        return out

    def forward(self, layer_activations: Tensor) -> Tensor:
        """Forward pass on all translators."""
        layer_activations.to(torch.bfloat16)
        out = torch.einsum(
            "lbsh,lhH->lbsH",
            layer_activations,
            self.translators,
        )
        out = out + self.bias
        return out


class DistributedTunedLens(nn.Module):
    """Multi-GPU distributed version of TunedLens.

    Central object for TunedLens training and experiments. Implements the same
    training loop as the original TunedLens with some differences. Namely we
    choose to batch all parameter tensors, and shard them across GPUs layer-
    wise. Also uses the `FlexModel` backend for fetching highly-distributed
    activation and parameter tensors.
    """
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
        """Convert a dictionary containing layer names and tensors into a list.

        FlexModel returns activation tensors continuously into some dict, keyed
        by the layer name. This function converts that into a list which is
        indexed by the layer number.
        """
        output = [None for _ in range(len(layer_dict))]

        for i, layer_name in enumerate(self.all_layers):
            output[i] = layer_dict.pop(layer_name)

        return output

    def scatter_activations(self) -> Tensor:
        """Scatter the activations to lens model parallel workers.

        Only rank0 worker has the activations on CPU. First we organize them
        into a list keyed by list index in model. Then we chunk that list
        into # workers sublists and send them to the rest of the workers.

        Example:
            Given {"layer.1": tensor1, "layer.2": tensor2} and 2 GPU workers:

                # Organize into list:
                {"layer.1": tensor1, "layer.2": tensor2} -> [tensor1, tensor2]

                # Chunk into sublists
                [tensor1, tensor2] -> [[tensor1], [tensor2]]

                # Each workers gets item where the index of the item = worker
                # rank
                On rank0: output_list = [[tensor1]]
                On rank1: output_list = [[tensor2]]
        """
        world_size = get_lens_model_parallel_world_size()

        # Non-distributed fallback
        if world_size == 1:
            return self._convert_layer_dict_to_list(self.activation_dict)

        # Worker 0
        if get_lens_model_parallel_rank() == 0:
            # Activation dict to list
            act_list = self._convert_layer_dict_to_list(self.activation_dict)

            # Chunk activation list into world size # of sublists
            layer_group = 0
            objects = [[] for _ in range(world_size)]
            for act in act_list:
                # Move to next group of layers
                if len(objects[layer_group]) == self.num_layers:
                    layer_group += 1

                objects[layer_group].append(act)

        # Workers != 0
        else:
            objects = [None for _ in range(world_size)]

        output_list = [None]

        dist.scatter_object_list(output_list, objects, src=0)

        # output_list: [[Tensor, ...]]
        output_list = output_list[0]

        # Log scatter logic
        if get_lens_model_parallel_rank() == 0:
            logger.info(f"Rank{get_lens_model_parallel_rank()} Scatter: "
                        f"[{len(objects) * len(objects[0])}] -> [{len(output_list)}]")


        return output_list

    def streamed_loss(self, batch, loss_fn, chunks=4):
        """Save VRAM by sequentially calculating loss across outer dimension.

        Calculating the KL all at once causes unembedding from hidden dimension
        to vocab dimension, which is huge considering we have between 20-100+
        layers. Instead we can opt to calculate these in chunks to save memory.
        """
        assert self.num_layers % chunks == 0

        # Forward pass to generate activations and target logits
        with torch.no_grad():
            logits = self.frozen_model.forward(batch.cuda(), start_pos=0)

        # Distribute activations from rank0 to all others and consolidate
        activations = self.scatter_activations()
        activations = torch.stack([act for act in activations], dim=0)
        activations = activations.cuda().to(self.dtype)

        # Chunk activations along layer dimension
        chunk_size = self.num_layers // chunks
        activations = torch.chunk(activations, chunks, dim=0)

        # Create loss mask (layer, batch, seq, hidden)
        mask = (batch != 0)[None, :, :, None].cuda()

        # Stream the loss calculation to avoid expanding vocab size activation
        # across num_layers
        loss = 0
        for i, act_chunk in enumerate(activations):
            # Set of layers to unembed
            start = i * chunk_size
            end = start + chunk_size
            indices = list(range(start, end))

            # Run lens unembed on activations
            # See `forward` function below for explanation on residual connection
            pred_logits = self.translators.partial_forward(act_chunk, indices) + act_chunk
            pred_logits = self.norm(pred_logits)
            pred_logits = self.unembed(pred_logits)

            # Calculate batched loss
            loss += loss_fn(logits, pred_logits, mask)

        # Average loss over all chunks
        loss = loss / chunks
        return loss

    def forward(self, batch):
        """Standard forward pass to unembed activations."""
        with torch.no_grad():
            logits = self.frozen_model.forward(batch.cuda(), start_pos=0)

        activations = self.scatter_activations()
        activations = torch.stack([act for act in activations], dim=0)
        activations = activations.cuda().to(self.dtype)

        # Residual connection not in paper, but included in TunedLens codebase:
        # https://github.com/AlignmentResearch/tuned-lens/blob/9bf1f35ec664d8d224b7f35756af820660b003a2/tuned_lens/nn/lenses.py#L309
        # "Ensure[s] that weight wecay regularizes the transform toward the
        # identity, not the zero transformation.
        # NOTE: Weight decay is applied element-wise to the weights - batched
        #       linear layers w/ weight decay is equivalent to a list of linear
        #       layers.
        pred_logits = self.translators(activations) + activations
        pred_logits = self.norm(pred_logits)
        pred_logits = self.unembed(pred_logits)

        return logits, pred_logits


@dataclass
class DistributedTunedLensTrainerConfig:
    """Metadata object for distributed lens training."""
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
    """Distributed TunedLens trainer."""
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
        """Parse out config for optimizers, schedulers, etc."""
        # TODO: Only works for SGD right now
        self.optimizer = self.config.optimizer_type(
            self.distributed_tuned_lens.parameters(),
            lr=self.config.lr_scale * (1 - self.config.momentum),
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            nesterov=True,
        )
        # TODO: Parameterize the type of scheduler
        #       Ie. scheduler_type in ["linear", "cosine", ...] and select
        #       constructor function with args based on that.
        if self.config.use_scheduler:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.lr_warmup_steps,
                num_training_steps=self.config.total_num_steps,
            )
        else:
            self.scheduler = None

    def train(self):
        """Train loop."""
        self.distributed_tuned_lens.train()

        train_dataloader = iter(self.train_dataloader)

        interval_loss = 0
        for step in tqdm(range(self.config.total_num_steps)):
            if step != 0 and step % self.config.train_steps_per_val == 0:
                self.validate()

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

    def validate(self):
        self.distributed_tuned_lens.eval()

        val_dataloader = iter(self.val_dataloader)

        val_loss = 0
        for step in tqdm(range(self.config.val_steps)):
            batch = next(val_dataloader)
            loss = self.validate_step(batch)
            val_loss += loss.item()

        val_loss = val_loss / self.config.val_steps
        print(f"Rank{get_lens_model_parallel_rank()} val loss at {step}: {val_loss}")

    def validate_step(self, batch):
        with torch.no_grad():
            loss = self.distributed_tuned_lens.streamed_loss(
                batch,
                loss_fn=self.config.loss_fn,
                chunks=4,
            )
        return loss
