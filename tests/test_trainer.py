"""End-to-end tests of the trainer (plasticity/training/online.py) on tiny configurations.

``run_experiment`` is exercised on a 2-task x 300-sample Online Permuted MNIST stream (with every
diagnostic switched on), on a 2-task CIFAR-100 binary stream, on the stationary MNIST control and
through the CLI entry point. Determinism across runs with the same seed is checked on the logged
numbers.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from plasticity.run import main as cli_main
from plasticity.training import run_experiment
from plasticity.training.online import run_stationary, run_stream
from plasticity.utils import PROJECT_ROOT, load_config, read_jsonl

# keys every per-task row of a continual run must carry (with all diagnostics enabled)
TASK_ROW_KEYS = {
    "task", "time", "online_accuracy", "online_loss", "n_train",
    "step_grad_norm_theta", "step_dw_norm", "step_dtheta_norm",
    "w_abs_mean/all", "w_fro_norm/all", "w_abs_max/all", "theta_abs_mean/all",
    "dead_frac/all", "inactive_frac/all", "effective_rank/last", "stable_rank/last",
    "effective_rank_c/last", "stable_rank_c/last",
    "grad_norm_theta", "grad_norm_w", "fisher_trace_theta", "fisher_trace_w",
    "fisher_erank_theta", "fisher_erank_w", "jac_fro_ratio", "fresh_accuracy",
}
SUMMARY_KEYS = {"auc_norm", "retention_ratio", "fresh_gap_final", "fresh_gap_mean", "early_window_mean",
                "final_window_mean", "drop", "n_tasks", "metric", "seed", "stream_seed", "global_steps", "method", "stream"}


def pmnist_cfg(n_tasks: int = 2, samples: int = 300, seed: int = 0, **model_over) -> dict:
    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    cfg["seed"] = seed
    cfg["log_every"] = 0
    cfg["stream"] = {"name": "pmnist", "n_tasks": n_tasks, "samples_per_task": samples, "normalize": "unit"}
    cfg["model"]["hidden_sizes"] = [100, 100, 100]
    cfg["model"].update(model_over)
    cfg["optimizer"] = {"name": "sgd", "lr": 0.01}
    cfg["train"] = {"batch_size": 1, "epochs_per_task": 1}
    cfg["metrics"] = {"every_n_tasks": 1, "probe_size": 200, "fisher_samples": 4, "update_tracking_every": 50}
    cfg["fresh_reference"] = {"every_n_tasks": 1}
    return cfg


def numeric_rows(rows):
    """Rows without the wall-clock field (the only non-deterministic entry)."""
    return [{k: v for k, v in r.items() if k != "time"} for r in rows]


@pytest.fixture(scope="module")
def pmnist_run(mnist_cache, tmp_path_factory):
    cfg = pmnist_cfg()
    out = tmp_path_factory.mktemp("pmnist_run")
    summary = run_experiment(copy.deepcopy(cfg), out)
    return cfg, out, read_jsonl(out / "tasks.jsonl"), summary


# ------------------------------------------------------------------------------- PMNIST
def test_pmnist_outputs_and_keys(pmnist_run):
    cfg, out, rows, summary = pmnist_run
    assert (out / "config.json").exists() and (out / "tasks.jsonl").exists() and (out / "summary.json").exists()
    assert json.loads((out / "config.json").read_text()) == cfg
    assert len(rows) == 2 and [r["task"] for r in rows] == [0, 1]
    for r in rows:
        missing = TASK_ROW_KEYS - set(r)
        assert not missing, f"missing keys: {sorted(missing)}"
        assert r["n_train"] == 300
        for li in range(3):
            assert f"dead_frac/L{li}" in r and f"w_abs_mean/L{li}" in r
        assert "w_abs_mean/L3" in r  # output layer weights are reported too
        assert 0.0 <= r["online_accuracy"] <= 1.0 and 0.0 <= r["dead_frac/all"] <= 1.0
        assert 1.0 <= r["effective_rank/last"] <= 100.0
        assert r["step_grad_norm_theta"] > 0 and r["step_dw_norm"] > 0
        # standard parametrisation: both coordinate systems coincide
        assert r["grad_norm_theta"] == pytest.approx(r["grad_norm_w"])
        assert r["jac_fro_ratio"] == pytest.approx(1.0)
        assert "w_over_A_mean/all" not in r and "sat_frac/all" not in r
        # plain SGD: |dTheta| = lr * |grad| exactly (no momentum / weight decay); the effective-weight
        # update excludes the biases, so it is at most the full parameter update
        assert r["step_dtheta_norm"] == pytest.approx(0.01 * r["step_grad_norm_theta"], rel=1e-5)
        assert 0.0 < r["step_dw_norm"] <= r["step_dtheta_norm"] * (1 + 1e-9)
    on_disk = json.loads((out / "summary.json").read_text())
    assert {k: v for k, v in on_disk.items() if k != "wall_time"} == {k: v for k, v in summary.items() if k != "wall_time"}
    missing = SUMMARY_KEYS - set(summary)
    assert not missing, f"missing summary keys: {sorted(missing)}"
    assert summary["metric"] == "online_accuracy" and summary["n_tasks"] == 2 and summary["global_steps"] == 600
    assert summary["stream"] == "pmnist" and summary["method"] == "baseline" and summary["fresh_points"] == [0, 1]
    assert summary["auc_norm"] == pytest.approx((rows[0]["online_accuracy"] + rows[1]["online_accuracy"]) / 2)
    assert summary["retention_ratio"] == pytest.approx(rows[1]["online_accuracy"] / rows[0]["online_accuracy"])
    assert "final_dead_frac/all" in summary and "early_dead_frac/all" in summary
    assert "final_step_dw_norm" in summary and "final_fisher_trace_w" in summary


def test_pmnist_learning_happens(pmnist_run):
    _, _, rows, summary = pmnist_run
    assert rows[0]["online_accuracy"] > 0.3  # far above chance (0.1) within 300 online samples
    assert rows[0]["online_loss"] < math.log(10) * 1.5
    # the fresh reference at task 0 is the very same model / data / seed -> identical accuracy
    assert rows[0]["fresh_accuracy"] == rows[0]["online_accuracy"]
    gaps = [r["fresh_accuracy"] - r["online_accuracy"] for r in rows]
    assert summary["fresh_gap_mean"] == pytest.approx(sum(gaps) / 2)
    assert summary["fresh_gap_final"] == pytest.approx(gaps[-1])


def test_pmnist_determinism_same_seed(pmnist_run, tmp_path):
    cfg, _, rows, summary = pmnist_run
    summary2 = run_experiment(copy.deepcopy(cfg), tmp_path / "again")
    rows2 = read_jsonl(tmp_path / "again" / "tasks.jsonl")
    assert numeric_rows(rows2) == numeric_rows(rows)
    skip = {"wall_time"}
    assert {k: v for k, v in summary2.items() if k not in skip} == {k: v for k, v in summary.items() if k not in skip}


def test_pmnist_other_seed_differs(pmnist_run, tmp_path):
    cfg, _, rows, _ = pmnist_run
    cfg2 = copy.deepcopy(cfg)
    cfg2["seed"] = 1
    cfg2["fresh_reference"] = {"every_n_tasks": 0}
    cfg2["stream"]["n_tasks"] = 1
    run_experiment(cfg2, tmp_path / "seed1")
    r1 = read_jsonl(tmp_path / "seed1" / "tasks.jsonl")[0]
    assert "fresh_accuracy" not in r1
    assert r1["w_fro_norm/all"] != rows[0]["w_fro_norm/all"]


def test_pmnist_sin_reparam_run(mnist_cache, tmp_path):
    cfg = pmnist_cfg(n_tasks=1, samples=200, reparam={"hidden": "sin", "output": "sin"}, theta_scale="amplitude")
    cfg["fresh_reference"] = {"every_n_tasks": 0}
    cfg["metrics"]["update_tracking_every"] = 20
    cfg["save_model"] = True
    summary = run_stream(cfg, tmp_path)
    r = read_jsonl(tmp_path / "tasks.jsonl")[0]
    for k in ("w_over_A_mean/all", "sat_frac/all", "jac_sq_mean/all", "jac_mean/all", "w_over_A_mean/L0", "sat_frac/L3"):
        assert k in r
    assert 0.0 < r["w_over_A_mean/all"] <= 1.0 and 0.0 <= r["sat_frac/all"] <= 1.0
    # the largest amplitude is that of the 100-input hidden layers: 1.5 * sqrt(2) * sqrt(3/100)
    assert r["w_abs_max/all"] <= 1.5 * math.sqrt(6.0 / 100) + 1e-6
    # Theta- and W-coordinates differ for the reparametrised model
    assert r["grad_norm_theta"] != r["grad_norm_w"] and r["jac_fro_ratio"] <= 1.0
    assert r["step_dw_norm"] <= r["step_dtheta_norm"] + 1e-9  # |cos| <= 1 shrinks the effective step
    assert "final_w_over_A_mean/all" in summary and "final_sat_frac/all" in summary
    sd = torch.load(tmp_path / "model_final.pt")
    assert "hidden.0.theta" in sd and "hidden.0.amplitude" in sd and "output.theta_bias" in sd


# ------------------------------------------------------------------------------- CIFAR-100 binary
@pytest.mark.slow
def test_cifar_binary_run_records_test_accuracy(cifar_cache, tmp_path):
    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    cfg["log_every"] = 0
    cfg["stream"] = {"name": "cifar100_binary", "n_tasks": 2, "train_per_class": 50, "test_per_class": 20, "normalize": "standard"}
    cfg["model"]["hidden_sizes"] = [64, 64]
    cfg["metrics"] = {"every_n_tasks": 1, "probe_size": 40, "fisher_samples": 4, "update_tracking_every": 25}
    cfg["fresh_reference"] = {"every_n_tasks": 1}
    summary = run_experiment(cfg, tmp_path)
    rows = read_jsonl(tmp_path / "tasks.jsonl")
    assert len(rows) == 2
    for r in rows:
        assert "test_accuracy" in r and 0.0 <= r["test_accuracy"] <= 1.0
        assert r["n_train"] == 100 and "online_accuracy" in r and "fresh_accuracy" in r
        assert "dead_frac/all" in r and "fisher_trace_w" in r
    assert summary["metric"] == "test_accuracy" and summary["stream"] == "cifar100_binary"
    assert summary["auc_norm"] == pytest.approx((rows[0]["test_accuracy"] + rows[1]["test_accuracy"]) / 2)
    assert rows[0]["fresh_accuracy"] == rows[0]["test_accuracy"]  # same model/data/seed at task 0
    assert summary["global_steps"] == 200


# ------------------------------------------------------------------------------- stationary control
def test_run_stationary_one_epoch(mnist_cache, tmp_path):
    cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    cfg["log_every"] = 0
    cfg["stream"] = {"name": "stationary", "dataset": "mnist", "normalize": "unit"}
    cfg["model"]["hidden_sizes"] = [100, 100]
    cfg["train"] = {"batch_size": 256, "epochs": 1}
    cfg["metrics"] = {"probe_size": 200, "fisher_samples": 4}
    summary = run_experiment(cfg, tmp_path)
    rows = read_jsonl(tmp_path / "tasks.jsonl")
    assert len(rows) == 1 and rows[0]["task"] == 0 and rows[0]["epoch"] == 0
    r = rows[0]
    for k in ("train_accuracy", "train_loss", "test_accuracy", "dead_frac/all", "effective_rank/last", "fisher_trace_w", "w_abs_mean/all"):
        assert k in r
    assert r["test_accuracy"] > 0.5  # one epoch of SGD on MNIST already learns
    assert summary["metric"] == "test_accuracy" and summary["epochs"] == 1 and summary["stream"] == "mnist_stationary"
    assert summary["final_test_accuracy"] == r["test_accuracy"] == summary["best_test_accuracy"]
    assert summary["final_train_accuracy"] == r["train_accuracy"]
    assert json.loads((tmp_path / "summary.json").read_text())["final_test_accuracy"] == r["test_accuracy"]
    # determinism of the stationary trainer as well
    summary2 = run_stationary(cfg, tmp_path / "again")
    assert summary2["final_test_accuracy"] == summary["final_test_accuracy"]
    assert summary2["final_train_accuracy"] == summary["final_train_accuracy"]


# ------------------------------------------------------------------------------- CLI
def test_cli_with_overrides(mnist_cache, tmp_path, capsys):
    out = tmp_path / "cli"
    summary = cli_main([
        "--config", str(PROJECT_ROOT / "configs" / "pmnist_pilot.yaml"), "--out", str(out), "--quiet",
        "--set", "stream.n_tasks=1", "--set", "stream.samples_per_task=100", "--set", "metrics.probe_size=50",
        "--set", "metrics.fisher_samples=2", "--set", "fresh_reference.every_n_tasks=0", "--set", "optimizer.lr=0.02",
        "--set", "model.reparam.hidden=tanh",
    ])
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["stream"]["n_tasks"] == 1 and cfg["stream"]["samples_per_task"] == 100
    assert cfg["optimizer"]["lr"] == 0.02 and cfg["model"]["reparam"] == {"hidden": "tanh", "output": "standard"}
    assert cfg["log_every"] == 0
    assert summary["n_tasks"] == 1 and summary["global_steps"] == 100
    rows = read_jsonl(out / "tasks.jsonl")
    assert len(rows) == 1 and "w_over_A_mean/L0" in rows[0] and "w_over_A_mean/L3" not in rows[0]
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["metric"] == "online_accuracy" and printed["auc_norm"] == summary["auc_norm"]
