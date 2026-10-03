"""Tests for the paired-design statistics (plasticity/metrics/stats.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from plasticity.metrics import stats as S


def _shifted(n=10, shift=1.0, noise=0.1, seed=1):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=n)
    a = b + shift + noise * rng.normal(size=n)
    return a, b


# ---------------------------------------------------------------- paired_difference / sign-flip test
def test_identical_samples_give_p_one_and_zero_effect():
    x = np.random.default_rng(0).normal(size=12)
    r = S.paired_difference(x, x.copy())
    assert r["n"] == 12
    assert r["p_perm"] == 1.0 and r["perm_method"] == "exact"
    assert r["mean_diff"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert np.isnan(r["p_wilcoxon"]) and np.isnan(r["cohen_dz"]) and np.isnan(r["t_p"])


def test_clear_shift_is_detected():
    a, b = _shifted(n=10)
    r = S.paired_difference(a, b, seed=3)
    assert r["perm_method"] == "exact" and r["n_perm"] == 2 ** 10
    assert r["p_perm"] == pytest.approx(2 / 2 ** 10)  # smallest attainable two-sided exact p
    assert r["p_wilcoxon"] < 0.01 and r["t_p"] < 1e-6
    assert r["ci_low"] > 0.8 and r["ci_high"] < 1.2 and r["ci_low"] < r["mean_diff"] < r["ci_high"]
    assert r["mean_diff"] == pytest.approx(a.mean() - b.mean())
    assert r["cohen_dz"] == pytest.approx(np.mean(a - b) / np.std(a - b, ddof=1))
    assert r["mean_a"] == pytest.approx(a.mean()) and r["mean_b"] == pytest.approx(b.mean())
    # swapping the arguments flips the sign but keeps the p-values
    r2 = S.paired_difference(b, a, seed=3)
    assert r2["mean_diff"] == pytest.approx(-r["mean_diff"]) and r2["p_perm"] == r["p_perm"]
    assert r2["ci_low"] == pytest.approx(-r["ci_high"]) and r2["ci_high"] == pytest.approx(-r["ci_low"])


def test_exact_enumeration_matches_monte_carlo():
    rng = np.random.default_rng(5)
    b = rng.normal(size=12)
    a = b + 0.4 + 0.5 * rng.normal(size=12)  # moderate effect -> p in the interior
    exact = S.sign_flip_test(a - b)
    mc = S.sign_flip_test(a - b, force_monte_carlo=True, n_perm=20000, seed=0)
    assert exact["method"] == "exact" and mc["method"] == "monte_carlo"
    assert 0.005 < exact["p"] < 0.5
    assert abs(exact["p"] - mc["p"]) < 0.015  # Monte-Carlo SE at p~0.1 is ~0.002
    # exact enumeration reproduces the brute-force definition on a tiny sample
    d = np.array([0.3, -0.1, 0.5, 0.2, -0.4])
    t_obs = abs(d.mean())
    count = sum(abs((d * s).mean()) >= t_obs - 1e-12 for s in np.array(np.meshgrid(*[[-1, 1]] * 5)).T.reshape(-1, 5))
    assert S.sign_flip_test(d)["p"] == pytest.approx(count / 32)


def test_monte_carlo_uses_plus_one_correction_and_is_seeded():
    a, b = _shifted(n=20, shift=2.0, noise=0.05)
    r = S.paired_difference(a, b, n_perm=1000)
    assert r["perm_method"] == "monte_carlo" and r["n_perm"] == 1000
    assert r["p_perm"] == pytest.approx(1 / 1001)  # no null draw is as extreme as the observed shift
    r_same = S.paired_difference(a, b, n_perm=1000)
    assert r_same == r
    # zero differences do not change an exact p (sign flips of 0 are no-ops)
    d = np.array([0.5, 0.2, -0.1, 0.4])
    assert S.sign_flip_test(d)["p"] == S.sign_flip_test(np.concatenate([d, [0.0, 0.0]]))["p"]


def test_nan_pairs_are_dropped_and_length_mismatch_raises():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.0, 1.0, 1.0, np.nan])
    assert S.paired_difference(a, b)["n"] == 2
    with pytest.raises(ValueError):
        S.paired_difference([1, 2, 3], [1, 2])


# ---------------------------------------------------------------- confidence intervals
def test_bootstrap_ci_contains_true_mean_deterministic():
    x = np.random.default_rng(11).normal(loc=0.0, scale=1.0, size=30)
    m, lo, hi = S.mean_ci(x, n_boot=5000, seed=0)
    assert m == pytest.approx(x.mean()) and lo < 0.0 < hi and lo < m < hi
    assert S.mean_ci(x, n_boot=5000, seed=0) == (m, lo, hi)  # deterministic
    # coverage over a small simulation is close to the nominal 95 %
    rng = np.random.default_rng(2)
    hits = 0
    for k in range(120):
        xs = rng.normal(size=15)
        _, lo_k, hi_k = S.mean_ci(xs, n_boot=1000, seed=k)
        hits += lo_k < 0 < hi_k
    assert 0.85 <= hits / 120 <= 1.0


def test_t_ci_matches_scipy_and_edge_cases():
    x = np.array([1.0, 2.5, 0.3, 1.7, 2.2])
    m, lo, hi = S.t_ci(x, alpha=0.05)
    ref = sps.t.interval(0.95, df=4, loc=x.mean(), scale=sps.sem(x))
    assert (m, lo, hi) == pytest.approx((x.mean(), ref[0], ref[1]))
    assert S.mean_ci([3.0]) == (3.0, 3.0, 3.0)
    assert all(np.isnan(v) for v in S.mean_ci([]))
    assert np.isnan(S.t_ci([3.0])[1])


# ---------------------------------------------------------------- Holm
def test_holm_hand_computed_example():
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone -> 0.06
    adj = S.holm_correction([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert S.holm_correction([0.5, 0.6, 0.7]) == pytest.approx([1.0, 1.0, 1.0])  # clipped at 1
    assert S.holm_correction([0.02]) == pytest.approx([0.02])
    nan_adj = S.holm_correction([0.01, np.nan, 0.04])
    assert np.isnan(nan_adj[1]) and nan_adj[[0, 2]] == pytest.approx([0.02, 0.04])  # NaN excluded from m
    s = S.holm_correction(pd.Series([0.03, 0.001], index=["x", "y"]))
    assert isinstance(s, pd.Series) and list(s.index) == ["x", "y"] and s.to_numpy() == pytest.approx([0.03, 0.002])
    assert S.holm_correction([]).size == 0
    with pytest.raises(ValueError):
        S.holm_correction([0.5, 1.5])


# ---------------------------------------------------------------- DataFrame helpers
def _runs_df():
    rng = np.random.default_rng(7)
    seeds = np.arange(8)
    ref = rng.normal(size=8)
    rows = [{"label": "ref", "seed": s, "acc": v, "drop": -v} for s, v in zip(seeds, ref)]
    rows += [{"label": "A", "seed": s, "acc": ref[s] + 1.0, "drop": -(ref[s] + 1.0)} for s in seeds]  # exactly +1 per seed
    rows += [{"label": "A", "seed": 99, "acc": 50.0, "drop": -50.0}]  # unpaired seed: must be dropped
    rows += [{"label": "B", "seed": s, "acc": ref[s] + 0.01 * rng.normal(), "drop": 0.0} for s in seeds]
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)  # shuffle rows
    return df, ref


def test_compare_to_reference_pairs_by_seed():
    df, ref = _runs_df()
    out = S.compare_to_reference(df, "acc", "ref", n_boot=2000, seed=0)
    assert list(out["label"]) == sorted(out["label"], key=list(pd.unique(df["label"])).index)
    assert "ref" not in set(out["label"]) and len(out) == 2
    a = out.set_index("label").loc["A"]
    assert a["n_pairs"] == 8  # seeds 0..7 only (99 has no reference partner, so the 50.0 outlier is ignored)
    assert a["mean_diff"] == pytest.approx(1.0) and a["ci_low"] == pytest.approx(1.0) and a["ci_high"] == pytest.approx(1.0)
    assert a["mean_ref"] == pytest.approx(ref.mean()) and a["mean_label"] == pytest.approx(ref.mean() + 1.0)
    assert a["p_perm"] == pytest.approx(2 / 256) and a["p_holm"] == pytest.approx(4 / 256) and a["cohen_dz"] > 1e6  # sd of d is round-off only
    assert bool(a["better"]) and not bool(a["worse"])
    b = out.set_index("label").loc["B"]
    assert b["n_pairs"] == 8 and abs(b["mean_diff"]) < 0.05 and not bool(b["better"]) and not bool(b["worse"])
    assert out["p_holm"].to_numpy() == pytest.approx(S.holm_correction(out["p_perm"].to_numpy()))
    # lower-is-better metric flips the verdict
    out_drop = S.compare_to_reference(df, "drop", "ref", n_boot=500, higher_is_better=False).set_index("label")
    assert bool(out_drop.loc["A", "better"]) and not bool(out_drop.loc["A", "worse"])
    assert bool(S.compare_to_reference(df, "drop", "ref", n_boot=500).set_index("label").loc["A", "worse"])
    with pytest.raises(ValueError):
        S.compare_to_reference(df, "acc", "nope")
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        S.compare_to_reference(dup, "acc", "ref", n_boot=200)
    assert len(S.compare_to_reference(dup, "acc", "ref", n_boot=200, duplicates="mean")) == 2


def test_summary_table_and_pairwise_matrix():
    df, ref = _runs_df()
    tab = S.summary_table(df, ["acc", "drop"], n_boot=1000, seed=0)
    assert list(tab.columns) == ["label", "metric", "n", "mean", "ci_low", "ci_high", "std", "sem"]
    assert len(tab) == 6 and set(tab["label"]) == {"ref", "A", "B"}
    r = tab.set_index(["label", "metric"]).loc[("ref", "acc")]
    assert r["n"] == 8 and r["mean"] == pytest.approx(ref.mean()) and r["std"] == pytest.approx(ref.std(ddof=1))
    assert r["ci_low"] < r["mean"] < r["ci_high"]
    assert tab.set_index(["label", "metric"]).loc[("A", "acc"), "n"] == 9  # 8 paired seeds + the unpaired seed 99
    pairs = S.pairwise_comparisons(df, "acc", n_boot=500)
    assert len(pairs) == 3 and pairs["p_holm"].to_numpy() == pytest.approx(S.holm_correction(pairs["p_perm"].to_numpy()))
    mat = S.pairwise_matrix(df, "acc", labels=["ref", "A", "B"], n_boot=500)
    assert list(mat.index) == ["ref", "A", "B"] and np.isnan(np.diag(mat)).all()
    assert np.allclose(mat.to_numpy(), mat.to_numpy().T, equal_nan=True)
    diff = S.pairwise_matrix(df, "acc", labels=["ref", "A", "B"], value="mean_diff", n_boot=500)
    assert diff.loc["A", "ref"] == pytest.approx(1.0) and diff.loc["ref", "A"] == pytest.approx(-1.0)
