#!/usr/bin/env python3
"""Progress of the scale-up suites in one screen (no heavy imports).

    python3 scripts/suite_status.py [--check-done] [suite ...]

Prints, per suite, ``done/total`` runs and every unfinished run with its task count, checkpoint age and lock
state.  With ``--check-done`` the exit code is 0 only when every run of every listed suite has a summary.json
(used by scripts/fallback_takeover.sh).  Default suites: the scale-up chain of scripts/run_scaleup_chain.sh."""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_suite import expand_jobs, load_suite  # noqa: E402

CHAIN = ["pmnist_adam_methods", "pmnist_fullstream", "pmnist_widenet"]


def _rows(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _age(path: Path) -> str:
    try:
        return f"{(time.time() - path.stat().st_mtime) / 60:.0f} min ago"
    except OSError:
        return "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suites", nargs="*", default=CHAIN)
    ap.add_argument("--check-done", action="store_true", help="exit 0 only if every run is complete")
    args = ap.parse_args()
    all_done = True
    for name in args.suites:
        spec = load_suite(ROOT / "suites" / f"{name}.yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unused selected_hparams keys are expected here
            jobs = expand_jobs(spec)
        n_tasks = int(spec.get("common", {}).get("stream.n_tasks", 0) or 0)
        done = [j for j in jobs if j.complete]
        print(f"{name}: {len(done)}/{len(jobs)} complete")
        for j in jobs:
            if j.complete:
                continue
            all_done = False
            rows = _rows(j.out_dir / "tasks.jsonl")
            ck = j.out_dir / "checkpoint.pt"
            lock = "locked" if (j.out_dir / "running.lock").exists() else "idle"
            print(f"  {j.name:<28} {rows:>4}/{n_tasks or '?'} tasks | checkpoint {_age(ck) if ck.exists() else 'none'} | {lock}")
    return (0 if all_done else 1) if args.check_done else 0


if __name__ == "__main__":
    sys.exit(main())
