"""Run-level summary metrics (pre-registered in the thesis proposal, "Performans ve Mekanizma Ölçütleri").

Given the per-task performance curve ``p_1..p_T`` (online accuracy for PMNIST, test accuracy for
CIFAR-100):

* ``auc_norm``          normalised area under the task-performance curve = mean_t p_t
* ``early_window_mean`` mean over the first ``window`` tasks
* ``final_window_mean`` mean over the last ``window`` tasks
* ``retention_ratio``   final_window_mean / early_window_mean   (plasticity retention; 1 = no loss)
* ``drop``              early_window_mean - final_window_mean
* ``fresh_gap_final``   mean over the last fresh-reference points of (fresh_acc - continual_acc)
* ``slope_per_100``     OLS slope of p_t vs t, per 100 tasks
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_WINDOW = 0.1  # fraction of the stream (or an int number of tasks)


def _window_size(T: int, window) -> int:
    if window is None:
        window = DEFAULT_WINDOW
    if isinstance(window, float) and window <= 1.0:
        return max(1, int(round(T * window)))
    return max(1, min(int(window), T))


def summarize_curve(perf: Sequence[float], window=None) -> Dict[str, float]:
    p = np.asarray(perf, dtype=float)
    T = p.size
    if T == 0:
        return {}
    w = _window_size(T, window)
    early, final = float(p[:w].mean()), float(p[-w:].mean())
    t = np.arange(T)
    slope = float(np.polyfit(t, p, 1)[0] * 100) if T > 1 else float("nan")
    return {
        "n_tasks": int(T),
        "window": int(w),
        "auc_norm": float(p.mean()),
        "early_window_mean": early,
        "final_window_mean": final,
        "retention_ratio": final / early if early > 0 else float("nan"),
        "drop": early - final,
        "slope_per_100": slope,
        "min_task_perf": float(p.min()),
        "last_task_perf": float(p[-1]),
    }


def summarize_run(rows: List[Dict[str, Any]], metric: str, window=None, fresh_key: str = "fresh_accuracy") -> Dict[str, Any]:
    """Summary of a run from its per-task log rows."""
    rows = sorted(rows, key=lambda r: r["task"])
    perf = [r[metric] for r in rows if metric in r]
    out = summarize_curve(perf, window)
    out["metric"] = metric
    fresh = [(r["task"], r[fresh_key], r[metric]) for r in rows if fresh_key in r and r.get(fresh_key) is not None]
    if fresh:
        gaps = np.array([f - c for _, f, c in fresh])
        # fresh-reference points that fall inside the final window (same window as final_window_mean);
        # at least the last point is always used
        w_tasks = out.get("window", 1)
        first_task_in_window = rows[-w_tasks]["task"] if rows else 0
        in_window = [g for (t, _, _), g in zip(fresh, gaps) if t >= first_task_in_window]
        out["fresh_gap_final"] = float(np.mean(in_window)) if in_window else float(gaps[-1])
        out["fresh_gap_mean"] = float(gaps.mean())
        out["fresh_points"] = [t for t, _, _ in fresh]
    # last-window means of mechanism metrics (so analysis can relate them to performance)
    w = out.get("window", 1)
    tail = rows[-w:]
    for key in ("dead_frac/all", "inactive_frac/all", "w_abs_mean/all", "w_fro_norm/all", "stable_rank/last", "effective_rank/last", "stable_rank_c/last", "effective_rank_c/last",
                "jac_sq_mean/all", "w_over_A_mean/all", "sat_frac/all", "grad_norm_w", "grad_norm_theta", "fisher_trace_w",
                "fisher_trace_theta", "fisher_erank_w", "fisher_erank_theta", "step_grad_norm_theta", "step_dw_norm", "step_dtheta_norm"):
        vals = [r[key] for r in tail if key in r and r[key] is not None]
        if vals:
            out[f"final_{key}"] = float(np.mean(vals))
        head = [r[key] for r in rows[:w] if key in r and r[key] is not None]
        if head:
            out[f"early_{key}"] = float(np.mean(head))
    return out
