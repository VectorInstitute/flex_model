import argparse
import copy

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    args = parser.parse_args()
    return args


class MatmulModel(nn.Module):
    def __init__(self, fn_to_bench):
        super().__init__()
        self.fc1 = nn.Linear(4096, 4096, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(4096, 4096, dtype=torch.bfloat16)
        self.fn_to_bench = fn_to_bench

    def forward(self, x):
        out = self.fc1(x)
        out = self.fn_to_bench(out)
        out = self.fc2(out)
        return out


def regular(tensor, acc):
    acc.append(tensor.detach().cpu())

    return tensor


def streams(tensor, acc):
    s = torch.cuda.Stream()

    with torch.cuda.stream(s):
        dump_t = copy.deepcopy(tensor.detach())
        acc.append(dump_t.cpu())

    return tensor


pinned_buffer = torch.empty(4096, 4096, dtype=torch.bfloat16).pin_memory()


def pinned(tensor, acc):
    pinned_buffer.copy_(tensor, non_blocking=True)
    res = torch.empty_like(pinned_buffer)
    res.copy_(pinned_buffer, non_blocking=True)
    acc.append(res)
    return tensor


def main(args):
    torch.manual_seed(0)
    wait = 1
    warmup = 2
    active = 10
    repeat = 1
    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat,
    )
    num_steps = wait + warmup + active

    inputs = torch.randn(num_steps, 4096, 4096, dtype=torch.bfloat16).cuda()

    acc = []

    root_dir = "/h/mchoi/projects/flex_model/examples/perf_testing"

    exp = args.exp

    # exp_fn = globals().get(exp, None)
    # assert exp_fn is not None

    regular_acc = []
    pinned_acc = []
    exps = {
        regular: regular_acc,
        pinned: pinned_acc,
    }
    for i, (e, acc) in enumerate(exps.items()):
        if i == 0:
            network = MatmulModel(lambda x: e(x, acc=acc)).cuda()
        else:
            network.fn_to_bench = lambda x: e(x, acc=acc)
        trace_handler = torch.profiler.tensorboard_trace_handler(
            root_dir + f"/profiles/test_profiles/{e.__name__}"
        )
        # trace_handler = None
        with torch.profiler.profile(
            schedule=schedule,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
            profile_memory=True,
            on_trace_ready=trace_handler,
        ) as prof:
            for i in range(num_steps):
                _ = network(inputs[i])
                prof.step()

    for reg, pin in zip(regular_acc, pinned_acc):
        torch.allclose(reg, pin)


if __name__ == "__main__":
    args = parse_args()
    main(args)
