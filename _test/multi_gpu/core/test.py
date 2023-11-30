import os

import torch
import torch.distributed as dist


def do_something():
    x = torch.empty(1).cuda()
    tensor_list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, x)
    print("All-gather complete")


local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)


dist.init_process_group("nccl")
print(f"Rank{dist.get_rank()} initiaized pg")

print(f"Rank{dist.get_rank()} doing something...")
do_something()

dist.destroy_process_group()
print("pg destroyed")


dist.init_process_group("nccl")
print(f"Rank{dist.get_rank()} initiaized pg")

print(f"Rank{dist.get_rank()} doing something...")
do_something()

dist.destroy_process_group()
print("pg destroyed")
