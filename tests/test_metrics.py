"""Tests for the mechanism diagnostics (plasticity/metrics/mechanism.py) and run summaries
(plasticity/metrics/summary.py).

Rank helpers are checked on matrices with known spectra; dead / inactive fractions on a hand-built
feature matrix injected through a monkeypatched forward; the gradient / Fisher statistics on the
coordinate identity (standard model) and the Jacobian scaling (sin model); the curve summaries on
closed-form sequences.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from plasticity.metrics import (
    DEFAULT_WINDOW,
    compute_mechanism_metrics,
    gradient_fisher_statistics,
    representation_statistics,
    summarize_run,
    weight_statistics,
)
from plasticity.metrics.mechanism import _effective_rank, _stable_rank
from plasticity.metrics.summary import summarize_curve
from plasticity.models import MLP

D_IN, HIDDEN, N_CLS = 20, (16, 12), 4


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_model(hidden="standard", output="standard", seed=0, **kw) -> MLP:
    return MLP(D_IN, hidden_sizes=HIDDEN, n_classes=N_CLS, reparam={"hidden": hidden, "output": output},
               generator=gen(seed), **kw)


# ------------------------------------------------------------------------------- rank helpers
def test_rank_helpers_on_known_matrices():
    n = 6
    eye = torch.eye(n)
    assert _effective_rank(eye) == pytest.approx(n, abs=1e-5)
    assert _stable_rank(eye) == pytest.approx(n, abs=1e-5)
    u, v = torch.randn(9, 1, generator=gen(1)), torch.randn(1, 5, generator=gen(2))
    rank1 = u @ v
    assert _effective_rank(rank1) == pytest.approx(1.0, abs=1e-5)
    assert _stable_rank(rank1) == pytest.approx(1.0, abs=1e-5)
    assert _effective_rank(torch.zeros(4, 4)) == 0.0 and _stable_rank(torch.zeros(4, 4)) == 0.0
    # two orthogonal columns with norms 1 and 2: stable rank (1+4)/4, effective rank exp(H(1/3, 2/3))
    m = torch.zeros(5, 2); m[0, 0] = 1.0; m[1, 1] = 2.0
    assert _stable_rank(m) == pytest.approx(5 / 4, abs=1e-6)
    p = np.array([1 / 3, 2 / 3])
    assert _effective_rank(m) == pytest.approx(float(np.exp(-(p * np.log(p)).sum())), abs=1e-5)
    # scale invariance
    assert _effective_rank(3.0 * eye) == pytest.approx(n, abs=1e-5) and _stable_rank(3.0 * eye) == pytest.approx(n, abs=1e-5)


# ------------------------------------------------------------------------------- representation
def _feature_matrix(n=20, seed=0):
    """(n, 10) features: 3 zero columns (dead + inactive), 2 identical constant columns (inactive),
    5 mutually orthogonal zero-mean columns of equal norm. Known spectrum (see test below)."""
    ones = torch.ones(n, 1)
    q, _ = torch.linalg.qr(torch.cat([ones, torch.randn(n, 5, generator=gen(seed))], 1))
    s = math.sqrt(n)
    const = torch.ones(n, 1)  # = +-sqrt(n) * q[:, :1]
    active = s * q[:, 1:6]  # orthogonal to the ones vector -> zero mean, same norm as `const`
    return torch.cat([torch.zeros(n, 3), const, const, active], 1)


def test_representation_statistics_known_feature_matrix(monkeypatch):
    model = make_model()
    H = _feature_matrix()
    n = H.shape[0]
    monkeypatch.setattr(model, "forward", lambda x, return_features=False: (torch.zeros(x.shape[0], N_CLS), [H, H.clone()]))
    out = representation_statistics(model, torch.randn(n, D_IN))
    assert out["dead_frac/L0"] == pytest.approx(0.3) and out["inactive_frac/L0"] == pytest.approx(0.5)
    assert out["dead_frac/L1"] == pytest.approx(0.3) and out["dead_frac/all"] == pytest.approx(0.3)
    assert out["inactive_frac/all"] == pytest.approx(0.5)
    # singular values: sqrt(2)*s (the duplicated constant column), s x5 -> closed-form ranks
    assert out["stable_rank/L0"] == pytest.approx(7 / 2, abs=1e-4)
    sv = np.array([math.sqrt(2)] + [1.0] * 5); p = sv / sv.sum()
    assert out["effective_rank/L0"] == pytest.approx(float(np.exp(-(p * np.log(p)).sum())), abs=1e-4)
    # centring removes the constant columns: five equal singular values remain
    assert out["stable_rank_c/L0"] == pytest.approx(5.0, abs=1e-4) and out["effective_rank_c/L0"] == pytest.approx(5.0, abs=1e-4)
    assert out["act_abs_mean/L0"] == pytest.approx(float(H.abs().mean()), abs=1e-6)
    assert out["act_fro_norm/L0"] == pytest.approx(float(H.norm()) / math.sqrt(n), abs=1e-5)
    for k in ("stable_rank", "effective_rank", "stable_rank_c", "effective_rank_c"):
        assert out[f"{k}/last"] == out[f"{k}/L1"]
    # a positive dead tolerance also counts the constant-1 columns as dead
    assert representation_statistics(model, torch.randn(n, D_IN), dead_tol=1.0)["dead_frac/L0"] == pytest.approx(0.5)
    assert model.training  # train mode restored


def test_representation_statistics_on_real_model():
    model = make_model()
    out = representation_statistics(model, torch.randn(64, D_IN, generator=gen(3)))
    for li, h in enumerate(HIDDEN):
        assert 0.0 <= out[f"dead_frac/L{li}"] <= 1.0 and 1.0 <= out[f"effective_rank/L{li}"] <= h + 1e-6
        assert out[f"stable_rank/L{li}"] <= out[f"effective_rank/L{li}"] + 1e-6 or out[f"stable_rank/L{li}"] <= h
    # kill every unit of the first hidden layer: all dead, rank 0 there
    with torch.no_grad():
        model.hidden[0].theta.zero_(); model.hidden[0].theta_bias.fill_(-1.0)
    out = representation_statistics(model, torch.randn(64, D_IN, generator=gen(3)))
    assert out["dead_frac/L0"] == 1.0 and out["inactive_frac/L0"] == 1.0 and out["effective_rank/L0"] == 0.0


# ------------------------------------------------------------------------------- weights
def test_weight_statistics_keys_standard_vs_sin():
    std = weight_statistics(make_model())
    n_layers = len(HIDDEN) + 1
    for li in range(n_layers):
        assert f"w_abs_mean/L{li}" in std and f"w_fro_norm/L{li}" in std and f"w_abs_max/L{li}" in std
    for k in ("w_abs_mean/all", "w_fro_norm/all", "w_abs_max/all", "theta_abs_mean/all"):
        assert k in std
    assert not any(k.startswith(("w_over_A", "sat_frac", "jac_mean", "jac_sq_mean")) for k in std)

    sin_model = make_model("sin", "sin", gamma=1.5)
    s = weight_statistics(sin_model)
    for k in ("w_over_A_mean/all", "sat_frac/all", "jac_mean/all", "jac_sq_mean/all"):
        assert k in s
    for li in range(n_layers):
        assert f"w_over_A_mean/L{li}" in s and f"sat_frac/L{li}" in s and f"jac_sq_mean/L{li}" in s
    # matched initialisation: the effective-weight statistics coincide with the standard model
    for k in ("w_abs_mean/all", "w_fro_norm/all", "w_abs_max/all"):
        assert s[k] == pytest.approx(std[k], abs=1e-5)
    assert s["w_fro_norm/all"] == pytest.approx(math.sqrt(sum(s[f"w_fro_norm/L{li}"] ** 2 for li in range(n_layers))))
    # at init |W0| <= b = A / gamma, so |W|/A <= 1/gamma and nothing is saturated
    assert 0.0 < s["w_over_A_mean/all"] <= 1 / 1.5 + 1e-6
    assert s["sat_frac/all"] == 0.0 and 0.0 < s["jac_sq_mean/all"] <= 1.0
    # mixed: reparametrised hidden layers, standard output -> per-layer keys only for hidden layers
    mixed = weight_statistics(make_model("sin", "standard"))
    assert "w_over_A_mean/L0" in mixed and f"w_over_A_mean/L{n_layers - 1}" not in mixed and "w_over_A_mean/all" in mixed


def test_weight_statistics_saturation_at_the_bound():
    model = make_model("sin", "sin")
    with torch.no_grad():
        for lin in model.linear_layers:  # push every angle to pi/2 (amplitude scale: Phi = A * pi/2)
            lin.theta.fill_(float(lin.amplitude) * math.pi / 2)
    s = weight_statistics(model)
    assert s["w_over_A_mean/all"] == pytest.approx(1.0, abs=1e-6)
    assert s["sat_frac/all"] == pytest.approx(1.0) and s["jac_sq_mean/all"] == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------------------------------------- gradients / Fisher
def _probe(n=8, seed=5):
    x = torch.randn(n, D_IN, generator=gen(seed))
    y = torch.randint(0, N_CLS, (n,), generator=gen(seed + 1))
    return x, y


def test_gradient_fisher_standard_model_coordinates_coincide():
    model = make_model()
    x, y = _probe()
    out = gradient_fisher_statistics(model, x, y, n_samples=4, generator=gen(0))
    assert out["grad_norm_theta"] == out["grad_norm_w"] > 0
    assert out["fisher_trace_theta"] == out["fisher_trace_w"] > 0
    assert out["fisher_erank_theta"] == out["fisher_erank_w"]
    assert 1.0 <= out["fisher_erank_w"] <= 4.0 + 1e-6
    assert out["jac_fro_ratio"] == pytest.approx(1.0)
    # model state restored: capture off, gradients cleared, training flag kept
    assert all(lin.last_weight is None for lin in model.linear_layers)
    assert all(p.grad is None for p in model.parameters()) and model.training
    # the generator makes the Monte-Carlo Fisher deterministic
    again = gradient_fisher_statistics(model, x, y, n_samples=4, generator=gen(0))
    assert again == out


def test_gradient_fisher_matches_manual_per_sample_gradient():
    model = make_model()
    x, y = _probe(n=3)
    out = gradient_fisher_statistics(model, x, y, n_samples=3, generator=gen(0))
    norms = []
    for i in range(3):
        model.zero_grad()
        torch.nn.functional.cross_entropy(model(x[i : i + 1]), y[i : i + 1]).backward()
        norms.append(math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters())))
    model.zero_grad(set_to_none=True)
    assert out["grad_norm_w"] == pytest.approx(sum(norms) / 3, rel=1e-5)


@pytest.mark.parametrize("theta_scale", ("unit", "amplitude"))
def test_gradient_fisher_sin_model_jacobian_scaling(theta_scale):
    model = make_model("sin", "sin", theta_scale=theta_scale)
    x, y = _probe()
    out = gradient_fisher_statistics(model, x, y, n_samples=4, generator=gen(0))
    assert out["grad_norm_theta"] != out["grad_norm_w"]
    assert out["fisher_trace_theta"] != out["fisher_trace_w"]
    assert out["jac_fro_ratio"] == pytest.approx(out["grad_norm_theta"] / out["grad_norm_w"])
    assert out["jac_fro_ratio"] < 1.0  # |dW/dTheta| = A|cos| < 1 (unit) resp. |cos| <= 1 (amplitude)
    if theta_scale == "unit":
        # the Theta-gradient is shrunk by at least the largest amplitude (gamma * largest init bound)
        a_max = max(float(lin.amplitude) for lin in model.linear_layers)
        assert a_max < 1.0 and out["jac_fro_ratio"] <= a_max + 1e-6
    # W-coordinates are comparable across parametrisations: matched init -> same W-gradient as standard
    std = gradient_fisher_statistics(make_model(), x, y, n_samples=4, generator=gen(0))
    assert out["grad_norm_w"] == pytest.approx(std["grad_norm_w"], rel=1e-4)
    assert out["fisher_trace_w"] == pytest.approx(std["fisher_trace_w"], rel=1e-4)


def test_compute_mechanism_metrics_respects_flags():
    model = make_model("sin", "sin")
    x, y = _probe()
    full = compute_mechanism_metrics(model, x, y, {"fisher_samples": 4}, generator=gen(0))
    assert {"w_abs_mean/all", "dead_frac/all", "fisher_trace_w", "sat_frac/all"} <= set(full)
    only_f = compute_mechanism_metrics(model, x, y, {"weights": False, "representation": False, "fisher_samples": 4}, generator=gen(0))
    assert "fisher_trace_w" in only_f and "w_abs_mean/all" not in only_f and "dead_frac/all" not in only_f
    none = compute_mechanism_metrics(model, x, y, {"weights": False, "representation": False, "fisher": False})
    assert none == {}


# ------------------------------------------------------------------------------- summaries
def test_summarize_curve_known_sequence():
    perf = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    s = summarize_curve(perf, window=0.2)
    assert s["n_tasks"] == 10 and s["window"] == 2
    assert s["auc_norm"] == pytest.approx(0.45)
    assert s["early_window_mean"] == pytest.approx(0.85) and s["final_window_mean"] == pytest.approx(0.05)
    assert s["retention_ratio"] == pytest.approx(0.05 / 0.85) and s["drop"] == pytest.approx(0.8)
    assert s["slope_per_100"] == pytest.approx(-10.0)
    assert s["min_task_perf"] == 0.0 and s["last_task_perf"] == 0.0
    # default window = 10 % of the stream; integer windows count tasks
    d = summarize_curve(perf)
    assert d["window"] == 1 and d["early_window_mean"] == pytest.approx(0.9) and d["final_window_mean"] == pytest.approx(0.0)
    i3 = summarize_curve(perf, window=3)
    assert i3["window"] == 3 and i3["early_window_mean"] == pytest.approx(0.8) and i3["final_window_mean"] == pytest.approx(0.1)
    assert summarize_curve(perf, window=50)["window"] == 10  # clipped to the stream length
    assert summarize_curve([]) == {}
    one = summarize_curve([0.5])
    assert one["auc_norm"] == 0.5 and math.isnan(one["slope_per_100"]) and one["retention_ratio"] == 1.0
    flat = summarize_curve([0.7] * 5)
    assert flat["retention_ratio"] == 1.0 and flat["drop"] == 0.0 and flat["slope_per_100"] == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(summarize_curve([0.0, 0.0, 0.5])["retention_ratio"])  # early mean 0 -> undefined


def test_summarize_run_handles_fresh_gaps_and_mechanism_tails():
    T = 10
    rows = []
    for t in range(T):
        r = {"task": t, "online_accuracy": 0.9 - 0.05 * t, "dead_frac/all": 0.1 * t, "grad_norm_w": None if t == 0 else 1.0 + t}
        if t in (0, 4, 9):
            r["fresh_accuracy"] = 0.9  # fresh model keeps the task-0 level
        rows.append(r)
    shuffled = [rows[i] for i in (3, 9, 0, 7, 1, 8, 2, 6, 4, 5)]
    s = summarize_run(shuffled, metric="online_accuracy", window=0.2)
    assert s["metric"] == "online_accuracy" and s["n_tasks"] == T and s["window"] == 2
    assert s["auc_norm"] == pytest.approx(np.mean([r["online_accuracy"] for r in rows]))
    assert s["early_window_mean"] == pytest.approx(0.875) and s["final_window_mean"] == pytest.approx(0.475)
    assert s["fresh_points"] == [0, 4, 9]
    gaps = [0.0, 0.2, 0.45]
    assert s["fresh_gap_mean"] == pytest.approx(np.mean(gaps))
    assert s["fresh_gap_final"] == pytest.approx(0.45)  # final window = tasks 8-9 -> only the point at task 9
    s7 = summarize_run(rows, metric="online_accuracy", window=0.7)
    assert s7["fresh_gap_final"] == pytest.approx(np.mean(gaps[-2:]))  # final window = tasks 3-9 -> points 4 and 9
    s_int = summarize_run(rows, metric="online_accuracy", window=6)
    assert s_int["window"] == 6 and s_int["fresh_gap_final"] == pytest.approx(np.mean(gaps[-2:]))  # int window, tasks 4-9
    # mechanism tails: mean over the last / first `window` rows, None values skipped
    assert s["final_dead_frac/all"] == pytest.approx(0.85) and s["early_dead_frac/all"] == pytest.approx(0.05)
    assert s["final_grad_norm_w"] == pytest.approx(9.5) and s["early_grad_norm_w"] == pytest.approx(2.0)
    assert "final_w_over_A_mean/all" not in s  # absent metrics stay absent
    # no fresh points at all -> no fresh keys, and rows with a None fresh value are ignored
    plain = summarize_run([{"task": t, "online_accuracy": 0.5, "fresh_accuracy": None} for t in range(4)], metric="online_accuracy")
    assert "fresh_gap_final" not in plain and "fresh_points" not in plain and plain["retention_ratio"] == 1.0
    assert DEFAULT_WINDOW == 0.1
