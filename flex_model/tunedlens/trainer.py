from dataclasses import dataclass, asdict
import logging
import math
import os
from typing import List, Dict, Tuple, Callable, Optional

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import flex_model.tunedlens.distributed as tl_dist
from flex_model.tunedlens.distributed_tunedlens import (
    DistributedTunedLens,
    kl_divergence,
)


logger = logging.getLogger(__name__)


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
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: Optional[int] = None


@dataclass
class DistributedTunedLensTrainerState:
    """Class containing all state for DistributedTunedLensTrainer."""
    train_loss: float
    val_loss: float
    step: int
    epoch: int
    model: DistributedTunedLens
    config: DistributedTunedLensTrainerConfig
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None


class DistributedTunedLensTrainer:
    """Distributed TunedLens trainer."""
    def __init__(self, state: DistributedTunedLensTrainerState) -> None:
        self.state = state

    @classmethod
    def init(
        cls,
        distributed_tuned_lens: DistributedTunedLens,
        config: DistributedTunedLensTrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ):
        # TODO: Only works for SGD right now
        optimizer = config.optimizer_type(
            distributed_tuned_lens.parameters(),
            lr=config.lr_scale * (1 - config.momentum),
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
        # TODO: Parameterize the type of scheduler
        #       Ie. scheduler_type in ["linear", "cosine", ...] and select
        #       constructor function with args based on that.
        if config.use_scheduler:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.lr_warmup_steps,
                num_training_steps=config.total_num_steps,
            )
        else:
            scheduler = None

        state = DistributedTunedLensTrainerState(
            train_loss=torch.inf,
            val_loss=torch.inf,
            step=0,
            epoch=0,
            model=distributed_tuned_lens,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return cls(state)

    @classmethod
    def from_pretrained(cls, load_path: str):
        state_dict = torch.load(load_path)
        state = DistributedTunedLensTrainerState(
            state_dict["train_loss"],
            state_dict["val_loss"],
            state_dict["step"],
            state_dict["epoch"],
            state_dict["model"],
            state_dict["config"],
            state_dict["train_dataloader"],
            state_dict["val_dataloader"],
            state_dict["test_dataloader"],
            state_dict["optimizer"],
            state_dict["scheduler"],
        )
        return cls(state)

    def save(self) -> None:
        assert self.state.config.checkpoint_dir is not None

        save_dir = self.state.config.checkpoint_dir
        os.makedirs(save_dir, exist_ok=False)

        rank = tl_dist.get_lens_model_parallel_rank()
        save_file = f"{save_dir}/shard{rank}.pt"

        torch.save(asdict(self.state), save_file)

    def train(self):
        """Train loop."""
        self.state.model.train()
        self.state.train_dataloader = iter(self.state.train_dataloader)

        interval_loss = 0
        val_loss = torch.inf
        for step in tqdm(range(self.state.config.total_num_steps)):
            # Validation
            if step != 0 and step % self.state.config.train_steps_per_val == 0:
                val_loss = self.validate()

            if step != 0 and step % self.state.config.checkpoint_interval == 0:
                self.save()

            # Train step
            batch = next(self.state.train_dataloader)
            loss = self.train_step(batch)

            interval_loss += loss

            if step != 0 and step % self.state.config.log_interval == 0:
                print(f"Rank{tl_dist.get_lens_model_parallel_rank()} loss at {step}: {interval_loss / self.state.config.log_interval}")
                interval_loss = 0
            
    def train_step(self, batch):
        self.state.optimizer.zero_grad()

        """
        logits, pred_logits = self.distributed_tuned_lens(batch)

        # Mask is bs -> lbsv
        mask = (batch != 0).cuda()
        mask = mask[None, :, :, None]

        loss = self.config.loss_fn(logits, pred_logits, mask=mask)
        """
        loss = self.state.model.streamed_loss(
            batch,
            loss_fn=self.state.config.loss_fn,
            chunks=4,
        )
        assert loss.isfinite()

        loss.backward()

        nn.utils.clip_grad_norm_(self.state.model.parameters(), self.state.config.clip)

        self.state.optimizer.step()

        if self.state.scheduler is not None:
            self.state.scheduler.step()

        return loss.item()

    def validate(self):
        self.state.model.eval()

        self.state.val_dataloader = iter(self.state.val_dataloader)

        val_loss = 0
        for step in tqdm(range(self.state.config.val_steps)):
            batch = next(self.state.val_dataloader)
            loss = self.validate_step(batch)
            val_loss += loss.item()

        val_loss = val_loss / self.state.config.val_steps
        print(f"Rank{tl_dist.get_lens_model_parallel_rank()} val loss at {step}: {val_loss}")
        return val_loss

    def validate_step(self, batch):
        with torch.no_grad():
            loss = self.state.model.streamed_loss(
                batch,
                loss_fn=self.state.config.loss_fn,
                chunks=4,
            )
        return loss
