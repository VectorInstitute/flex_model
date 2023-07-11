from pathlib import Path
import json
from typing import Tuple
import os

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import torch
import torch.distributed as dist

from llama import ModelArgs, Transformer, Tokenizer


def setup_model_parallel() -> Tuple[int, int]:
    """Parse distributed state and initialize model parallel."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", -1))
    nnodes = int(os.environ.get("SLURM_JOB_NUM_NODES", -1))
    world_size = nnodes * num_gpus

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    global_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return global_rank, world_size


def load_llama(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> Transformer:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), (f"Loading a checkpoint for MP={len(checkpoints)} but world size is "
        f"{world_size}")
    ckpt_path = checkpoints[local_rank]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    return model, tokenizer
