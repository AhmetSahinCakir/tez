"""Online / sequential-task trainer and the stationary control trainer.

``run_experiment(cfg, out_dir)`` dispatches on ``cfg['stream']['name']``:

* ``pmnist`` / ``cifar100_binary`` -> :func:`run_stream` (continual protocol, one log row per task)
* ``stationary``                   -> :func:`run_stationary` (i.i.d. epochs, one log row per epoch)

Each run writes ``config.json``, ``tasks.jsonl`` (per-task rows) and ``summary.json``.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..data import StationaryDataset, Task, build_stream
from ..methods import build_method
from ..metrics import compute_mechanism_metrics, summarize_run
from ..models import MLP, build_model
from ..utils import append_jsonl, dump_json, set_seed


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _model_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) * 7919 + 17)


def _method_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) * 104729 + 23)


@torch.no_grad()
def evaluate(model: MLP, x: torch.Tensor, y: torch.Tensor, batch_size: int = 2048) -> float:
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], batch_size):
        correct += int((model(x[i : i + batch_size]).argmax(1) == y[i : i + batch_size]).sum())
    model.train()
    return correct / max(1, x.shape[0])


@torch.no_grad()
def _effective_weights(model: MLP) -> List[torch.Tensor]:
    return [lin.effective_weight().clone() for lin in model.linear_layers]


class _StepStats:
    """Running means of per-step diagnostics (gradient / update sizes) collected every ``every`` steps."""

    def __init__(self, every: int):
        self.every = max(int(every), 0)
        self.reset()

    def reset(self):
        self.n = 0
        self.sums: Dict[str, float] = {}

    def add(self, **vals):
        self.n += 1
        for k, v in vals.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)

    def means(self) -> Dict[str, float]:
        return {f"step_{k}": v / self.n for k, v in self.sums.items()} if self.n else {}


def _train_on_task(
    model: MLP,
    method,
    optimizer: torch.optim.Optimizer,
    task: Task,
    train_cfg: Dict[str, Any],
    global_step: int,
    step_stats: Optional[_StepStats] = None,
    task_rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """One pass of the protocol over ``task``; returns online accuracy etc. and the new global step."""
    B = int(train_cfg.get("batch_size", 1))
    epochs = int(train_cfg.get("epochs_per_task", 1))
    x = torch.from_numpy(task.x_train)
    y = torch.from_numpy(task.y_train)
    N = x.shape[0]
    need_feats = bool(getattr(method, "requires_features", False))
    first_correct, first_loss = 0, 0.0
    last_correct, last_loss = 0, 0.0
    track_every = step_stats.every if step_stats is not None else 0
    for ep in range(epochs):
        if ep == 0:
            order = None  # stream order (online protocol)
        else:
            order = torch.from_numpy((task_rng or np.random.default_rng(ep)).permutation(N))
        correct, loss_sum = 0, 0.0
        for i in range(0, N, B):
            idx = slice(i, i + B) if order is None else order[i : i + B]
            xb, yb = x[idx], y[idx]
            out = model(xb, return_features=need_feats)
            logits, feats = (out if need_feats else (out, None))
            with torch.no_grad():
                correct += (logits.argmax(1) == yb).sum()
            loss = F.cross_entropy(logits, yb)
            loss_sum += loss.detach() * yb.shape[0]
            reg = method.regularizer()
            if reg is not None:
                loss = loss + reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            method.before_step(global_step)
            track = track_every > 0 and (global_step % track_every == 0)
            if track:
                w_before = _effective_weights(model)
                th_before = [p.detach().clone() for p in model.parameters()]
                g2 = sum(float((p.grad * p.grad).sum()) for p in model.parameters() if p.grad is not None)
            optimizer.step()
            method.after_step(global_step, feats)
            if track:
                with torch.no_grad():
                    w_after = _effective_weights(model)
                    dw2 = sum(float(((a - b) ** 2).sum()) for a, b in zip(w_after, w_before))
                    dth2 = sum(float(((p.detach() - q) ** 2).sum()) for p, q in zip(model.parameters(), th_before))
                step_stats.add(grad_norm_theta=math.sqrt(g2), dw_norm=math.sqrt(dw2), dtheta_norm=math.sqrt(dth2))
            global_step += 1
        correct, loss_sum = int(correct), float(loss_sum)
        if ep == 0:
            first_correct, first_loss = correct, loss_sum
        last_correct, last_loss = correct, loss_sum
    res = {
        "online_accuracy": first_correct / N,
        "online_loss": first_loss / N,
        "global_step": global_step,
        "n_train": N,
    }
    if epochs > 1:
        res["train_accuracy_last_epoch"] = last_correct / N
        res["train_loss_last_epoch"] = last_loss / N
    if task.x_test is not None:
        res["test_accuracy"] = evaluate(model, torch.from_numpy(task.x_test), torch.from_numpy(task.y_test))
    return res


# ----------------------------------------------------------------------------------------------
# continual protocols
# ----------------------------------------------------------------------------------------------
def run_stream(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    stream_seed = int(cfg.get("stream_seed", seed + int(cfg.get("stream_seed_offset", 0))))
    set_seed(seed)
    torch.set_num_threads(int(cfg.get("threads", 1)))
    stream = build_stream(cfg["stream"], stream_seed)
    model = build_model(cfg["model"], stream.input_dim, stream.n_classes, generator=_model_generator(seed))
    method = build_method(cfg.get("method", {"name": "baseline"}), model, cfg["optimizer"], generator=_method_generator(seed))
    optimizer = method.build_optimizer()
    train_cfg = cfg.get("train", {})
    m_cfg = cfg.get("metrics", {})
    metrics_every = int(m_cfg.get("every_n_tasks", 1))
    probe_size = int(m_cfg.get("probe_size", 1000))
    fisher_gen = torch.Generator().manual_seed(seed * 31 + 5)
    fresh_cfg = cfg.get("fresh_reference", {}) or {}
    fresh_every = int(fresh_cfg.get("every_n_tasks", 0) or 0)
    log_every = int(cfg.get("log_every", 10))
    step_stats = _StepStats(int(m_cfg.get("update_tracking_every", 0)))

    rows: List[Dict[str, Any]] = []
    log_path = out_dir / "tasks.jsonl"
    if log_path.exists():
        log_path.unlink()
    global_step = 0
    t0 = time.time()
    n_tasks = len(stream)
    task_rng = np.random.default_rng(stream_seed + 1234)
    for task in stream:
        t = task.index
        method.on_task_start(t)
        step_stats.reset()
        tt = time.time()
        res = _train_on_task(model, method, optimizer, task, train_cfg, global_step, step_stats, task_rng)
        global_step = res.pop("global_step")
        row: Dict[str, Any] = {"task": t, "time": time.time() - tt, **res, **step_stats.means()}
        row.update({f"method/{k}": v for k, v in method.state_summary().items()})
        if metrics_every > 0 and (t % metrics_every == 0 or t == n_tasks - 1):
            if task.x_test is not None:
                xp, yp = task.x_test[:probe_size], task.y_test[:probe_size]
            else:
                xp, yp = task.x_train[-probe_size:], task.y_train[-probe_size:]
            row.update(compute_mechanism_metrics(model, torch.from_numpy(xp), torch.from_numpy(yp), m_cfg, generator=fisher_gen))
        method.on_task_end(t)
        if fresh_every > 0 and (t % fresh_every == 0 or t == n_tasks - 1):
            fresh_model = build_model(cfg["model"], stream.input_dim, stream.n_classes, generator=_model_generator(seed))
            fresh_method = build_method(cfg.get("method", {"name": "baseline"}), fresh_model, cfg["optimizer"], generator=_method_generator(seed))
            fresh_opt = fresh_method.build_optimizer()
            fr = _train_on_task(fresh_model, fresh_method, fresh_opt, task, train_cfg, 0, None, np.random.default_rng(seed))
            row["fresh_accuracy"] = fr.get("test_accuracy", fr["online_accuracy"]) if stream.metric == "test_accuracy" else fr["online_accuracy"]
        rows.append(row)
        append_jsonl(row, log_path)
        if log_every and (t % log_every == 0 or t == n_tasks - 1):
            perf = row.get(stream.metric)
            print(f"[{out_dir.name}] task {t+1}/{n_tasks}  {stream.metric}={perf:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    summary = summarize_run(rows, metric=stream.metric, window=cfg.get("summary", {}).get("window"))
    summary.update({"seed": seed, "stream_seed": stream_seed, "wall_time": time.time() - t0, "global_steps": global_step,
                    "method": cfg.get("method", {}).get("name", "baseline"), "stream": stream.name})
    dump_json(summary, out_dir / "summary.json")
    if cfg.get("save_model", False):
        torch.save(model.state_dict(), out_dir / "model_final.pt")
    return summary


# ----------------------------------------------------------------------------------------------
# stationary control
# ----------------------------------------------------------------------------------------------
def run_stationary(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    torch.set_num_threads(int(cfg.get("threads", 1)))
    ds: StationaryDataset = build_stream(cfg["stream"], seed)
    model = build_model(cfg["model"], ds.input_dim, ds.n_classes, generator=_model_generator(seed))
    method = build_method(cfg.get("method", {"name": "baseline"}), model, cfg["optimizer"], generator=_method_generator(seed))
    optimizer = method.build_optimizer()
    train_cfg = cfg.get("train", {})
    B = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 10))
    m_cfg = cfg.get("metrics", {})
    probe_size = int(m_cfg.get("probe_size", 1000))
    x, y = torch.from_numpy(ds.x_train), torch.from_numpy(ds.y_train)
    xt, yt = torch.from_numpy(ds.x_test), torch.from_numpy(ds.y_test)
    need_feats = bool(getattr(method, "requires_features", False))
    log_path = out_dir / "tasks.jsonl"
    if log_path.exists():
        log_path.unlink()
    rows: List[Dict[str, Any]] = []
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        order = torch.from_numpy(ds.epoch_order(ep))
        correct, loss_sum = 0, 0.0
        for i in range(0, x.shape[0], B):
            idx = order[i : i + B]
            xb, yb = x[idx], y[idx]
            out = model(xb, return_features=need_feats)
            logits, feats = (out if need_feats else (out, None))
            with torch.no_grad():
                correct += int((logits.argmax(1) == yb).sum())
            loss = F.cross_entropy(logits, yb)
            loss_sum += float(loss) * yb.shape[0]
            reg = method.regularizer()
            if reg is not None:
                loss = loss + reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            method.before_step(step)
            optimizer.step()
            method.after_step(step, feats)
            step += 1
        row = {"task": ep, "epoch": ep, "train_accuracy": correct / x.shape[0], "train_loss": loss_sum / x.shape[0],
               "test_accuracy": evaluate(model, xt, yt), "time": time.time() - t0}
        row.update(compute_mechanism_metrics(model, xt[:probe_size], yt[:probe_size], m_cfg))
        rows.append(row)
        append_jsonl(row, log_path)
        print(f"[{out_dir.name}] epoch {ep+1}/{epochs} train_acc={row['train_accuracy']:.4f} test_acc={row['test_accuracy']:.4f}", flush=True)
    summary = {"metric": "test_accuracy", "final_test_accuracy": rows[-1]["test_accuracy"], "best_test_accuracy": max(r["test_accuracy"] for r in rows),
               "final_train_accuracy": rows[-1]["train_accuracy"], "epochs": epochs, "seed": seed, "wall_time": time.time() - t0,
               "method": cfg.get("method", {}).get("name", "baseline"), "stream": ds.name}
    dump_json(summary, out_dir / "summary.json")
    return summary


def run_experiment(cfg: Dict[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(cfg, out_dir / "config.json")
    if cfg["stream"]["name"] == "stationary":
        return run_stationary(cfg, out_dir)
    return run_stream(cfg, out_dir)
