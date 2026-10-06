"""Task-boundary checkpointing of the continual trainer: an interrupted run resumed in the same directory must
reproduce the uninterrupted trajectory exactly (model, optimiser, method traces and every RNG are restored)."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from plasticity.training import online
from plasticity.training.online import CHECKPOINT_NAME, run_experiment
from plasticity.utils import load_config, read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tiny_cfg(method: dict | None = None, optimizer: dict | None = None, epochs: int = 1, **model_over) -> dict:
    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    cfg["seed"] = 0
    cfg["log_every"] = 0
    cfg["stream"] = {"name": "pmnist", "n_tasks": 6, "samples_per_task": 60, "normalize": "unit"}
    cfg["model"]["hidden_sizes"] = [8, 8]
    cfg["model"].update(model_over)
    cfg["method"] = method or {"name": "baseline"}
    cfg["optimizer"] = optimizer or {"name": "sgd", "lr": 0.01}
    cfg["train"] = {"batch_size": 1, "epochs_per_task": epochs}
    cfg["metrics"] = {"every_n_tasks": 2, "probe_size": 40, "fisher_samples": 4, "update_tracking_every": 10}
    cfg["fresh_reference"] = {"every_n_tasks": 3}
    cfg["checkpoint"] = {"every_n_tasks": 2}
    return cfg


def numeric_rows(rows):
    return [{k: v for k, v in r.items() if k != "time"} for r in rows]


class _Killed(Exception):
    pass


def run_interrupted(monkeypatch, cfg: dict, out: Path, stop_at: int) -> None:
    """Run ``cfg`` into ``out`` and 'kill' the process right before training task ``stop_at``."""
    orig = online._train_on_task

    def wrapped(model, method, optimizer, task, train_cfg, global_step, step_stats=None, task_rng=None):
        if step_stats is not None and task.index == stop_at:  # the main model (fresh references pass step_stats=None)
            raise _Killed()
        return orig(model, method, optimizer, task, train_cfg, global_step, step_stats, task_rng)

    with monkeypatch.context() as m:
        m.setattr(online, "_train_on_task", wrapped)
        with pytest.raises(_Killed):
            run_experiment(copy.deepcopy(cfg), out)


CASES = {
    "baseline_sgd": dict(),
    "baseline_adam_two_epochs": dict(optimizer={"name": "adam", "lr": 0.001}, epochs=2),
    "continual_backprop": dict(method={"name": "continual_backprop", "replacement_rate": 0.05, "maturity_threshold": 20}),
    "shrink_perturb_adam": dict(method={"name": "shrink_perturb", "shrink": 1e-3, "noise_std": 1e-2}, optimizer={"name": "adam", "lr": 0.001}),
    "upgd": dict(method={"name": "upgd", "sigma": 1e-2}),
    "weight_clipping": dict(method={"name": "weight_clipping", "kappa": 1.0}),
    "nap": dict(method={"name": "nap"}, layer_norm=True, ln_affine=False),
    "sin_adam": dict(optimizer={"name": "adam", "lr": 0.001}, reparam={"hidden": "sin", "output": "sin"}),
    "tri_cbp": dict(method={"name": "continual_backprop", "replacement_rate": 0.05, "maturity_threshold": 20},
                    reparam={"hidden": "tri", "output": "tri"}, gamma=1.0),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("stop_at", [4, 3])  # 4: resume exactly at the checkpoint; 3: one logged row past it is discarded
def test_resume_reproduces_uninterrupted_run(mnist_cache, tmp_path, monkeypatch, capsys, case, stop_at):
    cfg = tiny_cfg(**CASES[case])
    ref_dir = tmp_path / "ref"
    ref_summary = run_experiment(copy.deepcopy(cfg), ref_dir)
    ref_rows = numeric_rows(read_jsonl(ref_dir / "tasks.jsonl"))
    assert not (ref_dir / CHECKPOINT_NAME).exists()  # removed at the end of a complete run

    out = tmp_path / "interrupted"
    run_interrupted(monkeypatch, cfg, out, stop_at)
    assert (out / CHECKPOINT_NAME).exists()
    ck = torch.load(out / CHECKPOINT_NAME, weights_only=False)
    assert ck["task"] == 3 if stop_at == 4 else ck["task"] == 1
    assert len(read_jsonl(out / "tasks.jsonl")) == stop_at

    capsys.readouterr()
    summary = run_experiment(copy.deepcopy(cfg), out)
    assert "resumed from checkpoint at task" in capsys.readouterr().out
    rows = numeric_rows(read_jsonl(out / "tasks.jsonl"))
    assert len(rows) == len(ref_rows) == 6
    assert rows == ref_rows
    for k, v in ref_summary.items():
        if k != "wall_time":
            assert summary[k] == v, k
    assert not (out / CHECKPOINT_NAME).exists()


def test_checkpoint_ignored_when_config_differs(mnist_cache, tmp_path, monkeypatch, capsys):
    cfg = tiny_cfg()
    out = tmp_path / "run"
    run_interrupted(monkeypatch, cfg, out, stop_at=4)
    cfg2 = copy.deepcopy(cfg)
    cfg2["optimizer"]["lr"] = 0.02
    ref = run_experiment(copy.deepcopy(cfg2), tmp_path / "ref")
    capsys.readouterr()
    got = run_experiment(copy.deepcopy(cfg2), out)
    assert "resumed" not in capsys.readouterr().out
    assert numeric_rows(read_jsonl(out / "tasks.jsonl")) == numeric_rows(read_jsonl(tmp_path / "ref" / "tasks.jsonl"))
    assert got["auc_norm"] == ref["auc_norm"]


def test_volatile_keys_do_not_change_fingerprint():
    cfg = tiny_cfg()
    fp = online.config_fingerprint(cfg)
    cfg2 = copy.deepcopy(cfg)
    cfg2["log_every"], cfg2["threads"], cfg2["checkpoint"] = 5, 4, {"every_n_tasks": 1}
    assert online.config_fingerprint(cfg2) == fp
    cfg2["optimizer"]["lr"] = 0.1
    assert online.config_fingerprint(cfg2) != fp


def test_checkpoint_off_by_default_writes_nothing(mnist_cache, tmp_path):
    cfg = tiny_cfg()
    cfg["checkpoint"] = {"every_n_tasks": 0}
    run_experiment(cfg, tmp_path / "run")
    assert not (tmp_path / "run" / CHECKPOINT_NAME).exists()
