import logging
import argparse

from datasets import load_dataset
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from llama import Llama

from flex_model.utils import setup_logger
from flex_model.tunedlens.distributed_tunedlens import (
    DistributedTunedLens,
)
from flex_model.tunedlens.trainer import (
    DistributedTunedLensTrainer,
    DistributedTunedLensTrainerConfig,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="warning")
    parser.add_argument("--checkpoint_dir", type=str, default="/ssd005/projects/llm/llama-2-13b")
    parser.add_argument("--tokenizer_path", type=str, default="/ssd005/projects/llm/llama-2-13b/tokenizer.model")
    parser.add_argument("--hidden_dim", type=int, default=5120)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=250)
    parser.add_argument("--train_steps_per_val", type=int, default=20)
    parser.add_argument("--val_steps", type=int, default=50)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--tl_checkpoint_interval", type=int, default=1000)
    parser.add_argument("--tl_checkpoint_dir", type=str, default="/ssd003/home/mchoi/projects/flex_model/flex_model/tunedlens/checkpoints")
    args = parser.parse_args()
    return args


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


def get_wikitext103_dataloaders(batch_size, tokenize_fn):
    def collate_fn(examples):
        examples = [
            ex["text"] for ex in examples
        ]
        return tokenize_fn(examples)

    dataset = load_dataset("wikitext", "wikitext-103-v1")

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return train_dataloader, val_dataloader, test_dataloader


def distributed_main(args):
    torch.manual_seed(42069)

    setup_logger(args.log_level)

    # Create the model and tokenizer function
    model, tokenize_fn = make_llama2_model(
        args.checkpoint_dir,
        args.tokenizer_path,
        args.max_seq_len,
        args.batch_size,
    )

    train_dataloader, val_dataloader, test_dataloader = get_wikitext103_dataloaders(
        args.batch_size,
        tokenize_fn,
    )

    dtl = DistributedTunedLens(
        frozen_model=model,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        lens_model_parallel_size=torch.distributed.get_world_size(),
    )
    dtl_config = DistributedTunedLensTrainerConfig(
        optimizer_type=torch.optim.SGD,
        use_scheduler=True,
        batch_size=args.batch_size,
        lr_warmup_steps=args.warmup_steps,
        total_num_steps=args.num_steps,
        train_steps_per_val=args.train_steps_per_val,
        val_steps=args.val_steps,
        log_interval=args.log_interval,
        lr_scale=args.lr_scale,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        clip=args.clip,
        checkpoint_dir=args.tl_checkpoint_dir,
        checkpoint_interval=args.tl_checkpoint_interval,
    )
    dtl_trainer = DistributedTunedLensTrainer.init(
        dtl,
        dtl_config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )
    dtl_trainer.train()


if __name__ == "__main__":
    args = parse_args()
    distributed_main(args)
