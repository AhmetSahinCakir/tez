"""Regression tests for the ``load_suite`` cache: the cache must hold the whole suite, never a label-filtered
subset, and must be invalidated when the suite on disk changes (new / re-run / removed runs)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from plasticity.analysis import aggregate as A
from plasticity.metrics.summary import summarize_run
from plasticity.utils import append_jsonl, dump_json

N_TASKS = 6
LABELS = ("sin", "tanh", "baseline")


def _write_run(d: Path, label: str, seed: int, finished: bool = True, acc0: float = 0.8) -> None:
    d.mkdir(parents=True, exist_ok=True)
    dump_json({"seed": seed, "method": {"name": "baseline"}, "model": {"reparam": {"hidden": label if label != "baseline" else "standard"}},
               "optimizer": {"lr": 0.01}, "stream": {"name": "pmnist", "n_tasks": N_TASKS}}, d / "config.json")
    rng = np.random.default_rng(seed + 10 * len(label))
    rows = [{"task": t, "online_accuracy": float(acc0 - 0.01 * t + 0.001 * rng.standard_normal()), "n_train": 10} for t in range(N_TASKS)]
    if (d / "tasks.jsonl").exists():
        (d / "tasks.jsonl").unlink()
    for r in rows:
        append_jsonl(r, d / "tasks.jsonl")
    if finished:
        s = summarize_run(rows, "online_accuracy", window=0.5)
        s.update({"seed": seed, "method": "baseline", "stream": "pmnist"})
        dump_json(s, d / "summary.json")


@pytest.fixture
def suite(tmp_path):
    root = tmp_path / "suite"
    for lab in LABELS:
        for s in (0, 1):
            _write_run(root / lab / f"seed{s}", lab, s)
    return root


def _bump_mtime(path: Path) -> None:
    """Make a file look modified even on coarse-grained filesystems."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def test_filtered_load_never_shrinks_the_cache(suite, tmp_path):
    cache = tmp_path / "cache"
    r1, c1 = A.load_suite(suite, labels=["sin"], cache=cache, verbose=False)
    assert set(r1["label"]) == {"sin"} == set(c1["label"]) and len(r1) == 2
    assert (cache / "suite_manifest.json").is_file()
    # the cache holds the whole suite although only 'sin' was requested ...
    r_all, c_all = A.load_suite(suite, cache=cache, verbose=False)
    assert sorted(r_all["label"].unique()) == sorted(LABELS) and len(r_all) == 6
    assert sorted(c_all["label"].unique()) == sorted(LABELS) and len(c_all) == 6 * N_TASKS
    # ... and another label set is served from it correctly
    r2, c2 = A.load_suite(suite, labels=["tanh", "baseline", "ghost"], cache=cache, verbose=False)
    assert sorted(r2["label"].unique()) == ["baseline", "tanh"] and set(c2["label"]) == {"baseline", "tanh"}
    assert r2.attrs["unmatched_labels"] == ["ghost"]
    assert r2.attrs["root"] == str(suite.resolve())


def test_cache_hit_is_used_and_matches_raw_load(suite, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    runs, curves = A.load_suite(suite, cache=cache, verbose=False)
    calls = []
    orig = A.load_run
    monkeypatch.setattr(A, "load_run", lambda *a, **k: (calls.append(a), orig(*a, **k))[1])
    r2, c2 = A.load_suite(suite, cache=cache, verbose=False)
    assert calls == []  # served from the cache: no run directory was re-read
    assert len(r2) == len(runs) and len(c2) == len(curves)
    assert np.allclose(np.sort(r2["auc_norm"].to_numpy()), np.sort(runs["auc_norm"].to_numpy()))
    assert "base_label" in r2.columns and "base_label" in c2.columns
    # refresh=True re-reads the raw files and rewrites the cache
    A.load_suite(suite, cache=cache, refresh=True, verbose=False)
    assert len(calls) == 6


def test_cache_invalidated_by_new_rerun_and_removed_runs(suite, tmp_path):
    cache = tmp_path / "cache"
    r0, _ = A.load_suite(suite, cache=cache, verbose=False)
    assert len(r0) == 6
    # a run that finishes after the cache was written is picked up without refresh=True
    _write_run(suite / "sin" / "seed2", "sin", 2)
    r1, c1 = A.load_suite(suite, cache=cache, verbose=False)
    assert len(r1) == 7 and (r1["label"] == "sin").sum() == 3 and len(c1) == 7 * N_TASKS
    # a re-run of an existing directory (new summary.json content / mtime) is picked up too
    _write_run(suite / "tanh" / "seed0", "tanh", 0, acc0=0.3)
    _bump_mtime(suite / "tanh" / "seed0" / "summary.json")
    r2, _ = A.load_suite(suite, cache=cache, verbose=False)
    v = r2.loc[(r2.label == "tanh") & (r2.seed == 0), "auc_norm"].iloc[0]
    assert v < 0.35
    # a removed run disappears
    for f in (suite / "baseline" / "seed1").iterdir():
        f.unlink()
    (suite / "baseline" / "seed1").rmdir()
    r3, c3 = A.load_suite(suite, cache=cache, verbose=False)
    assert len(r3) == 6 and (r3["label"] == "baseline").sum() == 1 and set(c3["label"]) == set(LABELS)


def test_unfinished_runs_skipped_and_include_unfinished_separate_cache(suite, tmp_path):
    cache = tmp_path / "cache"
    _write_run(suite / "partial" / "seed0", "partial", 0, finished=False)
    r, c, skipped = A.load_suite(suite, cache=cache, verbose=False, return_skipped=True)
    assert skipped == ["partial/seed0"] and "partial" not in set(c["label"]) and len(r) == 6
    # the cached skipped list survives a cache hit and is filtered by `labels`
    r_hit, _, skipped_hit = A.load_suite(suite, cache=cache, verbose=False, return_skipped=True)
    assert skipped_hit == ["partial/seed0"] and r_hit.attrs["skipped"] == ["partial/seed0"]
    r_sin, _, skipped_sin = A.load_suite(suite, labels=["sin"], cache=cache, verbose=False, return_skipped=True)
    assert skipped_sin == [] and set(r_sin["label"]) == {"sin"}
    # include_unfinished changes what the tables hold -> different fingerprint -> rebuilt, curves now hold 'partial'
    _, c2 = A.load_suite(suite, cache=cache, include_unfinished=True, verbose=False)
    assert "partial" in set(c2["label"])
    # an unfinished run that finishes later invalidates the cache
    _write_run(suite / "partial" / "seed0", "partial", 0, finished=True)
    r3, _ = A.load_suite(suite, cache=cache, include_unfinished=True, verbose=False)
    assert "partial" in set(r3["label"]) and len(r3) == 7


def test_legacy_or_broken_cache_is_rebuilt(suite, tmp_path):
    cache = tmp_path / "cache"
    runs, curves = A.load_suite(suite, verbose=False)
    # tables written by save_tables alone (no manifest, e.g. an analysis output dir) are not trusted blindly
    A.save_tables(runs[runs.label == "sin"], curves[curves.label == "sin"], cache, fmt="csv")
    r, c = A.load_suite(suite, cache=cache, verbose=False)
    assert len(r) == 6 and sorted(r["label"].unique()) == sorted(LABELS)
    # a corrupt manifest is ignored and rewritten
    (cache / "suite_manifest.json").write_text("{not json", encoding="utf-8")
    r2, _ = A.load_suite(suite, cache=cache, verbose=False)
    assert len(r2) == 6
    man = json.loads((cache / "suite_manifest.json").read_text(encoding="utf-8"))
    assert man["fingerprint"]["root"] == str(suite.resolve()) and len(man["fingerprint"]["runs"]) == 12
    # a manifest whose table file vanished -> rebuilt
    (cache / man["files"]["curves"]).unlink()
    r3, c3 = A.load_suite(suite, cache=cache, verbose=False)
    assert len(r3) == 6 and len(c3) == 6 * N_TASKS and (cache / man["files"]["curves"]).is_file()
