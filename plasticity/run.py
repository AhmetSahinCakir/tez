"""CLI: run one experiment from a YAML config.

    python -m plasticity.run --config configs/pmnist_pilot.yaml --out results/dev/run1 \
        --set method.name=weight_clipping --set method.kappa=2 --set seed=3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import run_experiment
from .utils import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config (may include 'base: other.yaml')")
    ap.add_argument("--out", required=True, help="output directory of this run")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted config override (YAML-parsed)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.quiet:
        cfg["log_every"] = 0
    summary = run_experiment(cfg, Path(args.out))
    keys = [k for k in ("metric", "auc_norm", "early_window_mean", "final_window_mean", "retention_ratio", "fresh_gap_final",
                        "final_test_accuracy", "wall_time") if k in summary]
    print(json.dumps({k: summary[k] for k in keys}, indent=None))
    return summary


if __name__ == "__main__":
    main()
