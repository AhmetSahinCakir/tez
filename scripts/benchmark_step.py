#!/usr/bin/env python
"""Per-step throughput benchmark (batch size 1, single thread) reported in docs/RAPOR.md §2-§3.

    python scripts/benchmark_step.py [--steps 1000] [--hidden 100,100,100]

Measures samples/s of the online training step (forward + backward + optimiser + method hooks, mirroring
plasticity/training/online.py) for the standard / sin / tanh models and for every registered method on a
784-h-h-h-10 MLP with synthetic inputs, plus the canonical 3x2000 standard network.  Absolute numbers depend on
the machine; the ratios are what the report quotes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plasticity.methods import build_method  # noqa: E402
from plasticity.models import MLP  # noqa: E402


def bench(model, method_cfg, steps, lr=0.003, warmup=100):
    gen = torch.Generator().manual_seed(0)
    method = build_method(method_cfg, model, {"name": "sgd", "lr": lr}, generator=gen)
    opt = method.build_optimizer()
    x = torch.rand(steps + warmup, model.input_dim)
    y = torch.randint(0, model.n_classes, (steps + warmup,))
    need = bool(getattr(method, "requires_features", False))

    def step(i):
        out = model(x[i : i + 1], return_features=need)
        logits, feats = (out if need else (out, None))
        loss = F.cross_entropy(logits, y[i : i + 1])
        reg = method.regularizer()
        if reg is not None:
            loss = loss + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        method.before_step(i)
        opt.step()
        method.after_step(i, feats)

    for i in range(warmup):
        step(i)
    t = time.perf_counter()
    for i in range(warmup, warmup + steps):
        step(i)
    return steps / (time.perf_counter() - t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--hidden", default="100,100,100")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    rows = []

    def add(label, model_kw, method_cfg):
        torch.manual_seed(0)
        m = MLP(784, hidden, 10, **model_kw)
        r = bench(m, method_cfg, args.steps)
        rows.append((label, r))
        print(f"{label:34s} {r:8.0f} samples/s", flush=True)

    add("baseline (standard)", {}, {"name": "baseline"})
    add("baseline (sin model)", {"reparam": {"hidden": "sin", "output": "sin"}}, {"name": "baseline"})
    add("baseline (tanh model)", {"reparam": {"hidden": "tanh", "output": "tanh"}}, {"name": "baseline"})
    add("weight_clipping", {}, {"name": "weight_clipping"})
    add("l2_init", {}, {"name": "l2_init"})
    add("continual_backprop", {}, {"name": "continual_backprop"})
    add("nap (LN, no affine)", {"layer_norm": True, "ln_affine": False}, {"name": "nap"})
    add("ln_wd (LN)", {"layer_norm": True}, {"name": "ln_wd"})
    add("shrink_perturb", {}, {"name": "shrink_perturb"})
    add("upgd", {}, {"name": "upgd"})
    add("parseval", {}, {"name": "parseval"})
    add("scale_corrected (sin model)", {"reparam": {"hidden": "sin", "output": "sin"}}, {"name": "scale_corrected"})
    base = rows[0][1]
    print("\nrelative to baseline:")
    for label, r in rows:
        print(f"{label:34s} {r / base:5.2f}x")
    torch.manual_seed(0)
    big = MLP(784, (2000, 2000, 2000), 10)
    print(f"\n{'canonical 3x2000 standard':34s} {bench(big, {'name': 'baseline'}, max(50, args.steps // 20), warmup=10):8.0f} samples/s")


if __name__ == "__main__":
    main()
