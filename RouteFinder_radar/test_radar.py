"""Evaluate the released RADAR checkpoint on the 16 asymmetric VRP variants (N=100).

Greedy decoding with multi-start (one rollout per customer as the first node, best kept),
fp16 autocast, batches of 250 instances. Prints the average cost and total inference time
per variant and the average over the 16 variants.

Usage:
    python test_radar.py [--ckpt checkpoints/radar_mtvrp100.ckpt] [--data data] [--batch 250]
"""
import argparse
import time

import torch

from routefinder.envs import MTVRPEnv
from routefinder.models import RouteFinderBase

VARIANTS = ["cvrp", "ovrp", "vrpb", "vrpl", "ovrpb", "ovrpl", "vrpbl", "vrptw",
            "ovrpbl", "ovrptw", "vrpbtw", "vrpltw", "ovrpbtw", "ovrpltw", "vrpbltw", "ovrpbltw"]

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/radar_mtvrp100.ckpt")
parser.add_argument("--data", default="data")
parser.add_argument("--batch", type=int, default=250)
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=1234,
                    help="seed for torch.svd_lowrank (randomized SVD); results vary slightly without it")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
policy = RouteFinderBase.load_from_checkpoint(args.ckpt, map_location="cpu").policy.to(device).eval()
env = MTVRPEnv(load_solutions=False)


def solve(td):
    td = env.reset(td).to(device)
    n = env.get_num_starts(td)  # number of customers
    with torch.inference_mode(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        out = policy(td, env, phase="test", num_starts=n)
    return -out["reward"].view(n, -1).float().max(0).values.cpu()


costs = {}
print(f"{'variant':>10} | {'#inst':>5} | {'cost':>7} | time (s)")
for v in VARIANTS:
    td = env.load_data(f"{args.data}/{v}/test/100.npz")
    torch.manual_seed(args.seed)
    solve(td[:2])  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    c = torch.cat([solve(td[s:s + args.batch]) for s in range(0, len(td), args.batch)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    costs[v] = c.mean().item()
    print(f"{v.upper():>10} | {len(c):>5} | {costs[v]:.4f} | {time.time() - t:.2f}")
print(f"{'average':>10} |       | {sum(costs.values()) / len(costs):.4f} |")
