"""
Resharding script from Llama-1 models. Use to test activation retrieval across
various MP configurations.
"""

import argparse
import json
from pathlib import Path
import psutil
import os

from tqdm import tqdm
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", type=str, default="/ssd005/projects/llm/llama/LLaMA/30B"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/ssd005/projects/llm/llama/LLaMA/30B_mp-1"
    )
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()
    return args


LAYER_TYPES = {
    "tok_embeddings": "ParallelEmbedding",
    "output": "ColumnParallelLinear",
    "wq": "ColumnParallelLinear",
    "wk": "ColumnParallelLinear",
    "wv": "ColumnParallelLinear",
    "wo": "RowParallelLinear",
    "w1": "ColumnParallelLinear",
    "w2": "RowParallelLinear",
    "w3": "ColumnParallelLinear",
    "attention_norm": None,
    "ffn_norm": None,
    "norm": None,
    "rope": None,
}


def main(args):
    print(
        f"Resharding from: {args.ckpt_dir} to {args.output_dir} | MP={args.num_shards}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(Path(args.ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    assert params["dim"] % args.num_shards == 0

    checkpoints = sorted(Path(args.ckpt_dir).glob("*.pth"))
    checkpoints = [torch.load(ckpt, map_location="cpu") for ckpt in checkpoints]

    original_shards = len(checkpoints)
    dims = params["dim"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // original_shards
    dims_per_head = dims // n_heads

    output = [{} for _ in range(args.num_shards)]

    layer_names = set(checkpoints[0].keys())

    for name in tqdm(layer_names):
        print(f"Resharding {name}...")
        sharded_tensors = [c[name] for c in checkpoints]

        postfix = name.split(".")[-2]  # Prune out ".weight" suffix

        assert postfix in LAYER_TYPES, f"key {name} not found"

        if LAYER_TYPES[postfix] in ["ParallelEmbedding", "RowParallelLinear"]:
            merged = torch.cat(sharded_tensors, dim=-1)
            resharded = torch.chunk(merged, args.num_shards, dim=-1)

        elif LAYER_TYPES[postfix] == "ColumnParallelLinear":
            if postfix in ["wq", "wk", "wv"]:
                merged = torch.cat(
                    [
                        t.view(n_heads_per_shard, dims_per_head, dims)
                        for t in sharded_tensors
                    ],
                    dim=0,
                )
                merged = merged.reshape(dims, dims)
                resharded = torch.chunk(
                    merged.view(n_heads, dims_per_head, dims), args.num_shards, dim=0
                )
                resharded = [
                    t.reshape(dims // args.num_shards, dims) for t in resharded
                ]
            else:
                merged = torch.cat(sharded_tensors, dim=-2)
                resharded = torch.chunk(merged, args.num_shards, dim=-2)

        else:
            # All shards have replicated weights
            resharded = [sharded_tensors[0] for _ in range(args.num_shards)]

        for rank, tensor in enumerate(resharded):
            output[rank][name] = tensor

        # Free up memory
        for c_dict in checkpoints:
            del c_dict[name]

        print(f"CPU usage: {psutil.cpu_percent()}")

    for rank, new_ckpt in enumerate(output):
        torch.save(new_ckpt, f"{args.output_dir}/consolidated_{rank}.pth")


if __name__ == "__main__":
    args = parse_args()
    main(args)
