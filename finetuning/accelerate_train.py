"""Barebones fine-tuning script for Llama-2-13b-hf on wikitext103.
"""

import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="/ssd005/projects/llm/llama-2-13b-hf")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--tokenizer_path", type=str, default="/ssd005/projects/llm/llama-2-13b-hf")
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.05)
    parser.add_argument("--clip_gradient", type=float, default=1.0)
    parser.add_argument("--train_steps_per_eval", type=int, default=50)
    parser.add_argument("--local_batch_size", type=int, default=4)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--local_batch_size_eval", type=int, default=16)
    args = parser.parse_args()
    return args


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
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    return train_dataloader, val_dataloader, test_dataloader


def main(args):
    # Set seed
    set_seed(args.seed)

    # Calculate grad accumulation steps and init accelerator
    # NOTE: This only works for local devices, multiple nodes needs to get this
    #       from slurm environment variables.
    num_processes = torch.cuda.device_count()
    gradient_accumulation_size = args.global_batch_size // args.local_batch_size * num_processes
    accelerator = Accelerator(gradient_accumulation_size)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
    )

    # Need to manually add pad token and increase model vocab embeddings size
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.model_max_length = args.model_max_length
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "right"
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_embeddings_avg = input_embeddings[:-1].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-1].mean(dim=0, keepdim=True)
    model.resize_token_embeddings(len(tokenizer))
    input_embeddings[-1:] = input_embeddings_avg
    output_embeddings[-1:] = output_embeddings_avg
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenization function for dataloaders
    def tokenize_fn(ps):
        return tokenizer(
            ps,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    # Load dataloaders from wikitext103 dataset
    train_dataloader, val_dataloader, test_dataloader = get_wikitext103_dataloaders(
        args.local_batch_size,
        tokenize_fn,
    )

    # Prepare model first - more efficient for fsdp
    model = accelerator.prepare(model)

    # Init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(args.beta_1, args.beta_2),
        lr=args.lr,
        weight_decay=args.wd,
    )

    # Init learning rate scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_steps_ratio),
        num_training_steps=num_training_steps,
    )

    # Shard dataloaders and optimizer for distributed
    train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        optimizer,
        lr_scheduler,
    )

    # We set pads to -100 to prevent loss being calculated from pads
    def add_labels(batch):
        labels = batch["input_ids"]
        labels = torch.where(
            labels == 32000,
            -100,
            labels,
        )
        batch["labels"] = labels
        return batch

    # Main training loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(args.num_epochs):
        for i, batch in enumerate(train_dataloader):
            # Grad accumulating training step
            with accelerator.accumulate(model):
                # Eval loop
                if i % args.train_steps_per_eval == 0 and i != 0:
                    model.eval()
                    val_loss = torch.tensor(0.).to(accelerator.device)
                    for batch in val_dataloader:
                        with torch.no_grad():
                            batch = add_labels(batch)
                            outputs = model(**batch)
                            assert outputs.loss.isfinite(), f"{batch}"
                            val_loss += outputs.loss
                    val_loss = accelerator.gather(val_loss).mean().item()
                    if accelerator.is_main_process:
                        accelerator.print(f"Val loss at iter {i}: "
                                          f"{val_loss / len(val_dataloader)}")
                    model.train()

                batch = add_labels(batch)
                optimizer.zero_grad()

                outputs = model(**batch)

                # Sanity check on predictions
                """
                if accelerator.is_main_process:
                    accelerator.print(
                        tokenizer.decode(outputs.logits[0].argmax(-1).tolist())
                    )
                    accelerator.print("*" * 200)
                    accelerator.print(
                        tokenizer.decode(batch["input_ids"][0].tolist())
                    )
                """

                loss = outputs.loss

                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), args.clip_gradient)

                train_loss = accelerator.gather(loss).mean().item()

                if accelerator.is_main_process:
                    accelerator.print(f"Train loss at iter {i}: {train_loss}")

                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)


if __name__ == "__main__":
    args = parse_args()
    # Need to disable infiniband
    os.environ["NCCL_IB_DISABLE"] = "1"
    main(args)
