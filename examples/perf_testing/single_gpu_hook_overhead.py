import argparse
import functools
import time

import torch
import torch.nn as nn
import wandb

import flex_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    return args


class Network(nn.Module):
    def __init__(self, model_dim, n_layers):
        super().__init__()
        self.model_dim = model_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [nn.Linear(model_dim, model_dim) for _ in range(self.n_layers)]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)

            if i < self.n_layers - 1:
                x = self.relu(x)

        return x


def _run(network, inputs, steps):
    torch.cuda.synchronize()
    start = time.time()

    for i in range(steps):
        _ = network(inputs[i])

    torch.cuda.synchronize()

    end = time.time()

    return end - start


def run_exp(network, inputs, steps):
    time = _run(network, inputs, steps)

    time_per_step = time / steps

    return time, time_per_step


def _hook_fn_with_cpu(self, inputs, outputs, acc, name):
    cpu_ten = outputs.detach().cpu()
    acc[name] = cpu_ten
    return outputs


ACC = {}
FM_ACC = {}


def hook_fn_without_cpu(self, inputs, outputs):
    return outputs


def make_network(cls, exp, *args, **kwargs):
    network = cls(*args, **kwargs)

    layers = [f"layers.{i}" for i in range(len(network.layers))]

    if exp == "hooks_without_cpu":
        for n, m in network.named_modules():
            if n in layers:
                m.register_forward_hook(hook_fn_without_cpu)

    elif exp == "hooks_with_cpu":
        for n, m in network.named_modules():
            if n in layers:
                m.register_forward_hook(
                    functools.partial(_hook_fn_with_cpu, name=n, acc=ACC)
                )

    elif exp == "flex_model":
        network = flex_model.core.FlexModel(network, FM_ACC,)
        for l in layers:
            network.register_hook_function(
                flex_model.core.HookFunction(l, expected_shape=(None, None))
            )

    return network


def main(args):
    if args.wandb:
        wandb.init(project="flex_model_exps", config={})
        wandb.define_metric("model_dim")

    all_exps = ["no_hooks", "hooks_without_cpu", "hooks_with_cpu", "flex_model"]

    for exp in all_exps:
        model_dim = [2 ** i for i in range(9, 15)]

        if args.wandb:
            wandb.define_metric(f"{exp}_wallclock_time", step_metric="model_dim")
            wandb.define_metric(f"{exp}_iter_time", step_metric="model_dim")

        for d in model_dim:
            # Inputs are shape (steps, bsz, model_dim).
            inputs = torch.randn(args.steps, 128, d).cuda()
            network = make_network(Network, exp, model_dim=d, n_layers=32,).cuda()

            wallclock, iter_time = run_exp(network, inputs, args.steps)

            if args.wandb:
                wandb.log(
                    {
                        "model_dim": d,
                        f"{exp}_wallclock_time": wallclock,
                        f"{exp}_iter_time": iter_time,
                    }
                )

            print(f"{exp} ({d}) wallclock:      {round(wallclock, 4)}s")
            print(f"{exp} ({d}) speed:          {round(iter_time, 4)}s/iter")


if __name__ == "__main__":
    args = parse_args()
    main(args)
