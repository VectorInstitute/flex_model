import logging
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist

from tunedlens_utils import LensTrainer, make_llama2_model, setup_logger
from flex_model.model_wrappers import FlexModel, HookFunction


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
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=250)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--train_steps_per_eval", type=int, default=50)
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(42069)

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
    num_layers = len(model.module.layers) // 2
    layers = [f"layers.{i}" for i in range(0, num_layers * 2, 2)]

    # Hook layers into model
    for layer in layers:
        hf = HookFunction(
            layer,
            expected_shape=(None, None, hidden_dim),
            editing_function=lambda x, _: x,
        )
        model.register_hook_function(hf)

    # Create trainer and run training loop
    trainer = LensTrainer(
        hidden_dim=hidden_dim,
        num_layers=len(layers),
        vocab_size=vocab_size,
        layers=layers,
        model=model,
        tokenize_fn=tokenize_fn,
        activation_dict=output_dict,
        warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        train_steps_per_eval=args.train_steps_per_eval,
        lr_scale=args.lr_scale,
        momentum=args.momentum,
        wd=args.wd,
        clip=args.clip
    )

    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
