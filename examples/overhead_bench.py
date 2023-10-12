import argparse
import logging
import time

import torch
from llama import Llama

from flex_model.core import FlexModel, HookFunction
from flex_model.utils import setup_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="/model-weights/Llama-2-13b"
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="/model-weights/Llama-2-13b/tokenizer.model",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)

    args = parser.parse_args()
    return args


def main(args):
    setup_logger(args.log_level)
    # Build the llama-2 model to benchmark on.
    generator = Llama.build(
        ckpt_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_dir,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )
    model = generator.model

    # Synthetic batch.
    data = torch.randint(0, 32000, (args.max_batch_size, args.max_seq_len)).cuda()

    # Benchmark regular inference time without flex model.
    start_t = time.time()
    for i in range(args.steps):
        _ = model(data, start_pos=0)
    torch.cuda.synchronize()
    without_t = (time.time() - start_t) / args.steps

    # Benchmark inference time with flex model, one hook per layer.
    out_dict = {}
    flex_model = FlexModel(
        model, out_dict, tensor_parallel_size=torch.distributed.get_world_size(),
    )
    hidden_dim = model.layers[0].feed_forward.w1.out_features
    n_layers = model.params.n_layers
    hook_functions = [
        HookFunction(
            f"layers.{i}.feed_forward.w1",
            expected_shape=(None, None, hidden_dim),
            editing_function=None,
        )
        for i in range(n_layers)
    ]
    for hf in hook_functions:
        flex_model.register_hook_function(hf)
    flex_model.enable_forward_hooks()

    start_t = time.time()
    for i in range(args.steps):
        _ = flex_model(data, start_pos=0)
    torch.cuda.synchronize()
    with_t = (time.time() - start_t) / args.steps

    if torch.distributed.get_rank() == 0:
        logger.info(f"Without flex model: {round(without_t, 4)}s/iter")
        logger.info(f"With flex model: {round(with_t, 4)}s/iter")


if __name__ == "__main__":
    args = parse_args()
    main(args)
