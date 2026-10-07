"""Tests for scripts/run_suite.py and scripts/select_hparams.py (suite grid runner + hyper-parameter selection)."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plasticity.utils import apply_overrides  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve postponed annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


run_suite_mod = _load("run_suite")
select_mod = _load("select_hparams")

TINY_COMMON = {"stream.n_tasks": 2, "stream.samples_per_task": 200, "metrics.every_n_tasks": 1, "metrics.fisher": False,
               "fresh_reference.every_n_tasks": 0}


def _write_tiny_suite(tmp_path: Path) -> Path:
    suite = {
        "name": "tiny",
        "base_config": "configs/pmnist_pilot.yaml",
        "workers": 2,
        "seeds": [0],
        "stream_seed_offset": 1000,
        "results_dir": str(tmp_path / "results"),
        "common": TINY_COMMON,
        "runs": [
            {"label": "baseline", "overrides": {"method.name": "baseline"}},
            {"label": "sin", "overrides": {"model.reparam.hidden": "sin"}},
        ],
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(suite, sort_keys=False))
    return path


# ------------------------------------------------------------------------------------------------
# override serialisation round trip
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0.01, 1e-05, 0.1, 1.0, 3, 0, -2, True, False, "sin", "yes", "null", "0.5", "a b", "x=y",
                                   [256, 256], [0.1, 0.2], ["sin", "tanh"], [], None])
def test_serialise_round_trip(value):
    text = run_suite_mod.serialise_value(value)
    cfg = apply_overrides({"a": {"b": "old"}}, [f"a.b={text}"])
    back = cfg["a"]["b"]
    assert back == value
    assert type(back) is type(value)


def test_serialise_sci_string_and_nested_dict():
    assert run_suite_mod.serialise_value("1e-4") == "0.0001"  # '1e-4' is a str in YAML 1.1 -> coerced to float
    with pytest.raises(TypeError):
        run_suite_mod.serialise_value({"a": 1})


# ------------------------------------------------------------------------------------------------
# grid expansion / commands (dry run, no subprocesses)
# ------------------------------------------------------------------------------------------------
def test_expand_sweep_and_commands(tmp_path):
    suite = {
        "name": "grid", "base_config": "configs/pmnist_pilot.yaml", "n_seeds": 2, "stream_seed_offset": 1000,
        "common": {"metrics": {"fisher": False}},          # nested -> flattened to metrics.fisher
        "runs": [
            {"label": "baseline", "overrides": {"method.name": "baseline"}},
            {"label": "wc", "overrides": {"method.name": "weight_clipping"}, "seeds": [7],
             "sweep": {"optimizer.lr": [0.01, 0.03], "method.kappa": [1.0, 2.0]}},
        ],
    }
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(suite, sort_keys=False))  # sweep keys expand in file order
    res = run_suite_mod.run_suite(path, dry_run=True, results_root=tmp_path / "out", verbose=False)
    labels = [(j.label, j.seed) for j in res.jobs]
    assert labels == [("baseline", 0), ("baseline", 1),
                      ("wc/lr=0.01/kappa=1.0", 7), ("wc/lr=0.01/kappa=2.0", 7),
                      ("wc/lr=0.03/kappa=1.0", 7), ("wc/lr=0.03/kappa=2.0", 7)]
    assert all(j.status == "dry-run" for j in res.jobs)
    assert not (tmp_path / "out").exists()  # dry run writes nothing
    job = res.jobs[3]
    assert job.out_dir == tmp_path / "out" / "wc/lr=0.01/kappa=2.0" / "seed7"
    assert job.overrides == {"metrics.fisher": False, "method.name": "weight_clipping", "optimizer.lr": 0.01, "method.kappa": 2.0}
    cmd = job.cmd
    assert cmd[1:3] == ["-m", "plasticity.run"]
    assert cmd[cmd.index("--config") + 1] == str(PROJECT_ROOT / "configs" / "pmnist_pilot.yaml")
    sets = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--set"]
    assert sets[:3] == ["seed=7", "stream_seed_offset=1000", "threads=1"]
    assert "optimizer.lr=0.01" in sets and "method.kappa=2.0" in sets and "metrics.fisher=false" in sets
    # the command's overrides reproduce the job's overrides through the real config loader
    cfg = apply_overrides({}, sets)
    assert cfg["optimizer"]["lr"] == 0.01 and cfg["method"]["kappa"] == 2.0 and cfg["metrics"]["fisher"] is False
    assert cfg["seed"] == 7 and cfg["stream_seed_offset"] == 1000

    # --only / --max-runs filtering
    res = run_suite_mod.run_suite(path, dry_run=True, results_root=tmp_path / "out", only="kappa=2.0", max_runs=1, verbose=False)
    assert [j.label for j in res.launched] == ["wc/lr=0.01/kappa=2.0"]
    assert sum(j.status == "pending" for j in res.jobs) == 5


def test_selected_hparams_merge(tmp_path):
    sel = tmp_path / "selected.yaml"
    sel.write_text(yaml.safe_dump({"metric": "auc_norm", "selected": {"sin": {"optimizer.lr": 0.03}, "baseline": {"optimizer.lr": 0.1}}}))
    suite = {"name": "s", "base_config": "configs/pmnist_pilot.yaml", "selected_hparams": str(sel),
             "runs": [{"label": "sin/ablation", "overrides": {"model.reparam.hidden": "sin"}},
                      {"label": "baseline", "overrides": {"optimizer.lr": 0.01}}]}   # explicit override wins
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(suite))
    res = run_suite_mod.run_suite(path, dry_run=True, results_root=tmp_path / "o", verbose=False)
    assert res.jobs[0].overrides == {"optimizer.lr": 0.03, "model.reparam.hidden": "sin"}
    assert res.jobs[1].overrides == {"optimizer.lr": 0.01}


# ------------------------------------------------------------------------------------------------
# end-to-end: two tiny runs in parallel, resumable
# ------------------------------------------------------------------------------------------------
def test_run_suite_end_to_end_and_resume(tmp_path):
    suite_path = _write_tiny_suite(tmp_path)
    res = run_suite_mod.run_suite(suite_path, workers=2, verbose=False)
    root = tmp_path / "results"
    assert res.results_root == root
    assert res.ok, [(j.name, j.status, j.returncode) for j in res.jobs]
    assert [j.status for j in res.jobs] == ["ok", "ok"]
    for label in ("baseline", "sin"):
        d = root / label / "seed0"
        for fname in ("config.json", "tasks.jsonl", "summary.json", "stdout.log"):
            assert (d / fname).exists(), f"{label}: {fname} missing"
        cfg = json.loads((d / "config.json").read_text())
        assert cfg["seed"] == 0 and cfg["stream_seed_offset"] == 1000 and cfg["threads"] == 1
        assert cfg["stream"]["n_tasks"] == 2 and cfg["stream"]["samples_per_task"] == 200 and cfg["metrics"]["fisher"] is False
        summary = json.loads((d / "summary.json").read_text())
        assert summary["n_tasks"] == 2 and summary["stream_seed"] == 1000
    assert json.loads((root / "sin/seed0/config.json").read_text())["model"]["reparam"]["hidden"] == "sin"
    assert (root / "suite.json").exists()

    index_path = root / "index.csv"
    assert res.index_path == index_path and index_path.exists()
    with open(index_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["label"], r["seed"], r["status"]) for r in rows] == [("baseline", "0", "ok"), ("sin", "0", "ok")]
    assert all(0.0 <= float(r["auc_norm"]) <= 1.0 for r in rows)
    assert all(float(r["wall_time"]) > 0 and float(r["elapsed"]) > 0 for r in rows)
    for col in ("metric", "early_window_mean", "final_window_mean", "retention_ratio", "run_dir"):
        assert col in rows[0]

    # re-running skips the completed runs (no subprocess launched, summaries untouched)
    mtimes = {label: (root / label / "seed0" / "summary.json").stat().st_mtime_ns for label in ("baseline", "sin")}
    res2 = run_suite_mod.run_suite(suite_path, workers=2, verbose=False)
    assert [j.status for j in res2.jobs] == ["skipped", "skipped"]
    assert res2.launched == [] and res2.ok
    assert mtimes == {label: (root / label / "seed0" / "summary.json").stat().st_mtime_ns for label in ("baseline", "sin")}
    with open(index_path, newline="") as f:
        rows2 = list(csv.DictReader(f))
    assert [r["status"] for r in rows2] == ["skipped", "skipped"]
    assert [r["wall_time"] for r in rows2] == [r["wall_time"] for r in rows] and all(r["elapsed"] == "" for r in rows2)
    assert [r["auc_norm"] for r in rows2] == [r["auc_norm"] for r in rows]  # metrics re-read from summary.json

    # --force on a filtered job re-runs just that job
    res3 = run_suite_mod.run_suite(suite_path, workers=1, force=True, only="baseline", dry_run=True, verbose=False)
    assert [j.name for j in res3.launched] == ["baseline/seed0"]


def test_failed_run_sets_status_and_retries(tmp_path):
    suite = {"name": "bad", "base_config": "configs/pmnist_pilot.yaml", "seeds": [0], "common": TINY_COMMON,
             "results_dir": str(tmp_path / "r"),
             "runs": [{"label": "broken", "overrides": {"stream.name": "no_such_stream"}}]}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(suite, sort_keys=False))
    # unknown stream -> the child exits non-zero (before loading data); its traceback is captured in stdout.log
    res = run_suite_mod.run_suite(path, workers=1, retries=0, verbose=False)
    job = res.jobs[0]
    assert job.status == "failed" and job.returncode != 0 and job.attempts == 1 and not res.ok
    log = (tmp_path / "r" / "broken" / "seed0" / "stdout.log").read_text()
    assert "attempt 1" in log and "Traceback" in log and "no_such_stream" in log
    assert not (tmp_path / "r" / "broken" / "seed0" / "summary.json").exists()
    with open(tmp_path / "r" / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "failed" and rows[0]["returncode"] not in ("", "0")
    # retries: an interpreter that cannot be launched fails every attempt (cheap: no subprocess start-up)
    res = run_suite_mod.run_suite(path, workers=1, retries=2, python=str(tmp_path / "no-such-python"), verbose=False)
    assert res.jobs[0].status == "failed" and res.jobs[0].attempts == 3 and res.jobs[0].returncode == -1
    assert "attempt 3" in (tmp_path / "r" / "broken" / "seed0" / "stdout.log").read_text()
    assert run_suite_mod.main([str(path), "--dry-run"]) == 0
    assert run_suite_mod.main([str(path), "--python", str(tmp_path / "no-such-python"), "--retries", "0"]) == 1


# ------------------------------------------------------------------------------------------------
# select_hparams on a synthetic results tree
# ------------------------------------------------------------------------------------------------
def _fake_run(root: Path, label: str, seed: int, auc: float, lr: float, kappa=None, method="baseline"):
    d = root / label / f"seed{seed}"
    d.mkdir(parents=True)
    cfg = {"seed": seed, "stream_seed_offset": 1000, "threads": 1, "optimizer": {"name": "sgd", "lr": lr, "weight_decay": 0.0},
           "method": {"name": method}, "model": {"hidden_sizes": [100, 100, 100]}}
    if kappa is not None:
        cfg["method"]["kappa"] = kappa
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "summary.json").write_text(json.dumps({"metric": "online_accuracy", "auc_norm": auc, "seed": seed, "wall_time": 1.0,
                                                "final_window_mean": auc - 0.1}))


def test_select_hparams_synthetic(tmp_path):
    root = tmp_path / "dev"
    for seed, auc in ((0, 0.70), (1, 0.72)):
        _fake_run(root, "baseline/lr=0.01", seed, auc, lr=0.01)
    for seed, auc in ((0, 0.80), (1, 0.78)):
        _fake_run(root, "baseline/lr=0.1", seed, auc, lr=0.1)
    _fake_run(root, "baseline/lr=0.3", 0, 0.99, lr=0.3)                          # single seed -> excluded by --min-seeds 2
    for seed, auc in ((0, 0.81), (1, 0.83)):
        _fake_run(root, "weight_clipping/lr=0.01/kappa=1.0", seed, auc, lr=0.01, kappa=1.0, method="weight_clipping")
    for seed, auc in ((0, 0.85), (1, 0.87)):
        _fake_run(root, "weight_clipping/lr=0.01/kappa=2.0", seed, auc, lr=0.01, kappa=2.0, method="weight_clipping")
    (root / "weight_clipping/lr=0.03/kappa=2.0/seed0").mkdir(parents=True)        # unfinished run (no summary) -> ignored
    (root / "weight_clipping/lr=0.03/kappa=2.0/seed0/config.json").write_text("{}")

    selected, details, variants, problems = select_mod.select_hparams(root, metric="auc_norm", min_seeds=2)
    assert selected == {"baseline": {"optimizer.lr": 0.1}, "weight_clipping": {"method.kappa": 2.0, "optimizer.lr": 0.01}}
    assert details["baseline"]["variant"] == "baseline/lr=0.1" and details["baseline"]["n"] == 2
    assert details["baseline"]["mean"] == pytest.approx(0.79) and details["baseline"]["std"] == pytest.approx(0.01414, abs=1e-4)
    assert details["weight_clipping"]["variant"] == "weight_clipping/lr=0.01/kappa=2.0"
    assert [v.label for v in variants if v.chosen] == ["baseline/lr=0.1", "weight_clipping/lr=0.01/kappa=2.0"]
    assert len(variants) == 5 and problems == []

    # min_seeds=1 lets the single-seed variant win; lower-is-better flips the choice
    sel1, _, _, _ = select_mod.select_hparams(root, min_seeds=1)
    assert sel1["baseline"] == {"optimizer.lr": 0.3}
    sel_min, _, _, _ = select_mod.select_hparams(root, min_seeds=2, higher_is_better=False)
    assert sel_min["baseline"] == {"optimizer.lr": 0.01} and sel_min["weight_clipping"]["method.kappa"] == 1.0

    out = tmp_path / "selected.yaml"
    yaml_path, table_path = select_mod.write_selection(out, selected, details, variants, "auc_norm", root, min_seeds=2)
    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["selected"] == selected and doc["metric"] == "auc_norm" and doc["details"]["baseline"]["seeds"] == [0, 1]
    table = table_path.read_text()
    assert "| baseline | baseline/lr=0.1 | 0.7900 ± 0.0141 | 2 | 0,1 | * |" in table
    assert "| weight_clipping | weight_clipping/lr=0.01/kappa=1.0 | 0.8200 ± 0.0141 | 2 | 0,1 |  |" in table

    # the YAML feeds run_suite's selected_hparams
    assert run_suite_mod._load_selected(str(yaml_path), tmp_path) == selected

    # CLI
    assert select_mod.main(["--results", str(root), "--out", str(tmp_path / "cli.yaml"), "--min-seeds", "2"]) == 0
    assert (tmp_path / "cli.yaml").exists() and (tmp_path / "cli.md").exists()
    assert select_mod.base_label("mnist/sin/lr=0.01", group="sweep") == "mnist/sin"
    assert select_mod.base_label("mnist/sin/lr=0.01") == "mnist"


# ------------------------------------------------------------------------------------------------
# selected_hparams: longest-prefix matching, unused keys warn
# ------------------------------------------------------------------------------------------------
def test_selected_hparams_longest_prefix(tmp_path):
    sel = tmp_path / "selected.yaml"
    sel.write_text(yaml.safe_dump({"selected": {"mnist": {"optimizer.lr": 0.1}, "mnist/sin": {"optimizer.lr": 0.01},
                                                "unused_key": {"optimizer.lr": 9.0}}}))
    suite = {"name": "st", "base_config": "configs/pmnist_pilot.yaml", "selected_hparams": str(sel),
             "runs": [{"label": "mnist/sin", "overrides": {"model.reparam.hidden": "sin"}, "sweep": {"model.gain": [1.0, 2.0]}},
                      {"label": "mnist/tanh", "overrides": {"model.reparam.hidden": "tanh"}},
                      {"label": "cifar/sin", "overrides": {}}]}
    path = tmp_path / "st.yaml"
    path.write_text(yaml.safe_dump(suite, sort_keys=False))
    with pytest.warns(UserWarning, match="unused_key"):
        res = run_suite_mod.run_suite(path, dry_run=True, results_root=tmp_path / "o", verbose=False)
    by_label = {j.label: j.overrides for j in res.jobs}
    assert by_label["mnist/sin/gain=1.0"] == {"optimizer.lr": 0.01, "model.reparam.hidden": "sin", "model.gain": 1.0}  # longest prefix
    assert by_label["mnist/tanh"] == {"optimizer.lr": 0.1, "model.reparam.hidden": "tanh"}                            # 'mnist' prefix
    assert by_label["cifar/sin"] == {}                                                                                # no match
    assert run_suite_mod.match_selected("a/b/c", {"a": 1, "a/b": 2}) == "a/b"
    assert run_suite_mod.match_selected("ab/c", {"a": 1}) is None   # '/' boundaries only, not string prefixes


def test_relative_paths_resolve_against_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)   # the cwd must not matter
    suite = {"name": "rel", "base_config": "configs/pmnist_pilot.yaml", "results_dir": "results/_rel_test",
             "runs": [{"label": "a", "overrides": {}}]}
    path = tmp_path / "rel.yaml"
    path.write_text(yaml.safe_dump(suite))
    res = run_suite_mod.run_suite(path, dry_run=True, verbose=False)
    assert res.results_root == PROJECT_ROOT / "results" / "_rel_test"
    assert res.jobs[0].cmd[res.jobs[0].cmd.index("--config") + 1] == str(PROJECT_ROOT / "configs" / "pmnist_pilot.yaml")
    res = run_suite_mod.run_suite(path, dry_run=True, results_root="results/_rel_test2", verbose=False)
    assert res.results_root == PROJECT_ROOT / "results" / "_rel_test2"
    # a base_config next to the suite file is found through the suite-dir fallback
    (tmp_path / "local.yaml").write_text("base: " + str(PROJECT_ROOT / "configs" / "pmnist_pilot.yaml") + "\n")
    suite["base_config"] = "local.yaml"
    path.write_text(yaml.safe_dump(suite))
    assert run_suite_mod.load_suite(path)["base_config"] == str(tmp_path / "local.yaml")
    suite["base_config"] = "does/not/exist.yaml"
    path.write_text(yaml.safe_dump(suite))
    with pytest.raises(FileNotFoundError):
        run_suite_mod.load_suite(path)


# ------------------------------------------------------------------------------------------------
# interrupt / signal / lock handling with fake interpreters (no training)
# ------------------------------------------------------------------------------------------------
def _fake_python(tmp_path: Path, name: str, body: str) -> str:
    """An executable standing in for the interpreter; $6 is the run dir (argv: -m plasticity.run --config C --out D ...)."""
    script = tmp_path / name
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(0o755)
    return str(script)


def _fake_suite(tmp_path: Path, name: str, n_runs: int = 2, retries: int = 1) -> Path:
    suite = {"name": name, "base_config": "configs/pmnist_pilot.yaml", "seeds": [0], "results_dir": str(tmp_path / name),
             "workers": 2, "runs": [{"label": f"r{i}", "overrides": {}} for i in range(n_runs)]}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(suite, sort_keys=False))
    return path


def test_interrupt_terminates_children_and_does_not_retry(tmp_path):
    import os
    import signal
    import threading
    import time

    # each child records its pid, then sleeps "forever" (exec: the pid of the sh script *is* the sleep)
    sleeper = _fake_python(tmp_path, "sleepy", 'echo $$ > "$6/child.pid"; exec sleep 60')
    suite_path = _fake_suite(tmp_path, "intr", n_runs=3)     # 3 jobs, 2 workers -> one job is still pending
    root = tmp_path / "intr"
    pid_files = [root / f"r{i}" / "seed0" / "child.pid" for i in range(2)]

    def send_sigint_when_children_run():
        deadline = time.time() + 20
        while time.time() < deadline and not all(p.exists() and p.read_text().strip() for p in pid_files):
            time.sleep(0.05)
        time.sleep(0.2)
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)

    threading.Thread(target=send_sigint_when_children_run, daemon=True).start()
    t0 = time.time()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_suite_mod.run_suite(suite_path, workers=2, retries=3, python=sleeper, verbose=False)
    finally:  # never leave sleepers behind if the runner misbehaves
        for p in pid_files:
            try:
                os.kill(int(p.read_text().strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
    assert time.time() - t0 < 15, "the interrupt must not wait for the (60 s) children"
    # the children were terminated (pids are gone) and the locks released
    time.sleep(0.1)
    for p in pid_files:
        pid = int(p.read_text().strip())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not (p.parent / run_suite_mod.LOCK_NAME).exists()
    # no retry was launched after the kill; the pending job was never started; index.csv records 'interrupted'
    for i in range(2):
        log = (root / f"r{i}" / "seed0" / "stdout.log").read_text()
        assert "attempt 1" in log and "attempt 2" not in log and "killed by signal" in log
    assert not (root / "r2" / "seed0" / "stdout.log").exists()
    with open(root / "index.csv", newline="") as f:
        rows = {r["label"]: r for r in csv.DictReader(f)}
    assert [rows[f"r{i}"]["status"] for i in range(3)] == ["interrupted"] * 3
    assert rows["r0"]["attempts"] == "1" and int(rows["r0"]["returncode"]) < 0
    # resuming afterwards relaunches everything (nothing was completed, nothing is locked)
    res = run_suite_mod.run_suite(suite_path, dry_run=True, verbose=False)
    assert [j.name for j in res.launched] == ["r0/seed0", "r1/seed0", "r2/seed0"]


def test_signal_killed_child_is_not_retried_but_exit_codes_are(tmp_path):
    suicidal = _fake_python(tmp_path, "suicidal", "kill -TERM $$")
    suite_path = _fake_suite(tmp_path, "sig", n_runs=1)
    res = run_suite_mod.run_suite(suite_path, workers=1, retries=2, python=suicidal, verbose=False)
    job = res.jobs[0]
    assert job.status == "failed" and job.returncode == -15 and job.attempts == 1 and not res.ok
    assert not job.lock_path.exists()
    failing = _fake_python(tmp_path, "failing", "exit 3")
    res = run_suite_mod.run_suite(suite_path, workers=1, retries=2, python=failing, verbose=False)
    assert res.jobs[0].status == "failed" and res.jobs[0].returncode == 3 and res.jobs[0].attempts == 3


def test_running_lock_skips_jobs_owned_by_a_live_process(tmp_path):
    import os
    import subprocess

    writer = _fake_python(tmp_path, "writer", 'echo \'{"auc_norm": 0.5, "wall_time": 1.0, "metric": "x"}\' > "$6/summary.json"')
    suite_path = _fake_suite(tmp_path, "lock", n_runs=2)
    root = tmp_path / "lock"
    live = root / "r0" / "seed0"
    live.mkdir(parents=True)
    (live / run_suite_mod.LOCK_NAME).write_text(f"pid={os.getpid()}\nhost={__import__('socket').gethostname()}\nrole=child\n")
    proc = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
    proc.wait()
    dead_pid = proc.pid                                       # reaped -> no live process has this pid
    stale = root / "r1" / "seed0"
    stale.mkdir(parents=True)
    (stale / run_suite_mod.LOCK_NAME).write_text(f"pid={dead_pid}\nhost={__import__('socket').gethostname()}\nrole=child\n")

    res = run_suite_mod.run_suite(suite_path, workers=2, python=writer, verbose=False)
    j0, j1 = res.jobs
    assert j0.status == "locked" and str(os.getpid()) in j0.note and not (live / "stdout.log").exists()
    assert (live / run_suite_mod.LOCK_NAME).exists()          # a live lock is never touched
    assert j1.status == "ok" and not (stale / run_suite_mod.LOCK_NAME).exists()   # the stale lock was taken over
    assert res.n_locked == 1 and not res.ok
    with open(root / "index.csv", newline="") as f:
        rows = {r["label"]: r["status"] for r in csv.DictReader(f)}
    assert rows == {"r0": "locked", "r1": "ok"}
    assert run_suite_mod.main([str(suite_path), "--python", writer]) == 1   # locked -> non-zero exit
    (live / run_suite_mod.LOCK_NAME).unlink()
    res = run_suite_mod.run_suite(suite_path, workers=2, python=writer, verbose=False)
    assert [j.status for j in res.jobs] == ["ok", "skipped"] and res.ok
    assert run_suite_mod._pid_alive(os.getpid()) and not run_suite_mod._pid_alive(dead_pid)


# ------------------------------------------------------------------------------------------------
# select_hparams: label-segment -> dotted key resolution
# ------------------------------------------------------------------------------------------------
def _fake_run_cfg(root: Path, label: str, seed: int, auc: float, cfg: dict):
    d = root / label / f"seed{seed}"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "summary.json").write_text(json.dumps({"metric": "online_accuracy", "auc_norm": auc, "seed": seed, "wall_time": 1.0}))


def test_select_hparams_label_key_resolution(tmp_path):
    assert select_mod._parse_label_value("1e-05") == 1e-05 and isinstance(select_mod._parse_label_value("1e-05"), float)
    assert select_mod._parse_label_value("0.01") == 0.01 and select_mod._parse_label_value("sin") == "sin"
    assert select_mod._parse_label_value("true") is True

    def cfg(lr, shrink=None, mlr=None):
        c = {"seed": 0, "optimizer": {"name": "sgd", "lr": lr}, "method": {"name": "sp"}}
        if shrink is not None:
            c["method"]["shrink"] = shrink
        if mlr is not None:
            c["method"]["lr"] = mlr   # a second key ending in '.lr' with the same value
        return c

    # (a) ambiguous 'lr' (optimizer.lr == method.lr) is resolved to the key that varies across the group;
    #     method.lr is constant and must NOT be written into the selection
    root = tmp_path / "a"
    _fake_run_cfg(root, "sp/lr=0.003", 0, 0.8, cfg(0.003, shrink=1e-05, mlr=0.003))
    _fake_run_cfg(root, "sp/lr=0.01", 0, 0.7, cfg(0.01, shrink=1e-05, mlr=0.003))
    sel, _, _, problems = select_mod.select_hparams(root)
    assert sel == {"sp": {"optimizer.lr": 0.003}} and problems == []

    # (b) single variant, nothing varies, two equal '.lr' keys, no suite.json -> warning, no guess, no bare key
    root = tmp_path / "b"
    _fake_run_cfg(root, "sp/lr=0.003/shrink=1e-05", 0, 0.8, cfg(0.003, shrink=1e-05, mlr=0.003))
    sel, _, _, problems = select_mod.select_hparams(root)
    assert sel == {"sp": {"method.shrink": 1e-05}}          # sci-notation label value matched against config.json
    assert len(problems) == 1 and "lr=0.003" in problems[0] and "method.lr" in problems[0]
    assert all("." in k for k in sel["sp"])

    # (c) the same tree with suite.json (written by run_suite.py) resolves 'lr' from the recorded sweep keys
    (root / "suite.json").write_text(json.dumps({"jobs": [
        {"label": "sp/lr=0.003/shrink=1e-05", "seed": 0, "sweep": {"optimizer.lr": 0.003, "method.shrink": 1e-05},
         "overrides": {"method.name": "sp", "optimizer.lr": 0.003, "method.shrink": 1e-05}}]}))
    sel, _, _, problems = select_mod.select_hparams(root)
    assert sel == {"sp": {"optimizer.lr": 0.003, "method.shrink": 1e-05}} and problems == []

    # (d) a key absent from config.json: dotted label keys pass through, bare ones are dropped with a warning
    root = tmp_path / "d"
    _fake_run_cfg(root, "m/foo=2/model.extra=3", 0, 0.5, {"seed": 0, "optimizer": {"lr": 0.1}})
    sel, _, _, problems = select_mod.select_hparams(root)
    assert sel == {"m": {"model.extra": 3}} and len(problems) == 1 and "foo=2" in problems[0]

    # (e) run_suite's suite.json round-trips through select_hparams end to end (dry grid, synthetic summaries)
    suite = {"name": "rt", "base_config": "configs/pmnist_pilot.yaml", "seeds": [0], "results_dir": str(tmp_path / "rt"),
             "runs": [{"label": "sp", "overrides": {"method.name": "shrink_perturb"},
                       "sweep": {"optimizer.lr": [0.003, 0.01], "method.shrink": [0.00001]}}]}
    (tmp_path / "rt.yaml").write_text(yaml.safe_dump(suite, sort_keys=False))
    jobs = run_suite_mod.expand_jobs(run_suite_mod.load_suite(tmp_path / "rt.yaml"))
    assert [j.label for j in jobs] == ["sp/lr=0.003/shrink=1e-05", "sp/lr=0.01/shrink=1e-05"]
    assert jobs[0].sweep == {"optimizer.lr": 0.003, "method.shrink": 1e-05}
    (tmp_path / "rt").mkdir()
    (tmp_path / "rt" / "suite.json").write_text(json.dumps({"jobs": [
        {"label": j.label, "seed": j.seed, "sweep": j.sweep, "overrides": j.overrides} for j in jobs]}))
    for j, auc in zip(jobs, (0.6, 0.9)):
        _fake_run_cfg(tmp_path / "rt", j.label, 0, auc, cfg(j.sweep["optimizer.lr"], shrink=1e-05, mlr=j.sweep["optimizer.lr"]))
    sel, details, _, problems = select_mod.select_hparams(tmp_path / "rt")
    assert sel == {"sp": {"optimizer.lr": 0.01, "method.shrink": 1e-05}} and problems == []
    assert details["sp"]["variant"] == "sp/lr=0.01/shrink=1e-05"


def test_lock_from_a_previous_boot_or_a_reused_pid_is_stale(tmp_path):
    """After a container replacement pids restart from 1: a lock whose pid is alive again but that was written
    under another boot id, or whose pid now belongs to an unrelated process, must count as stale."""
    import os, subprocess, sys as _sys
    run_dir = tmp_path / "r0"
    run_dir.mkdir()
    lock = run_dir / run_suite_mod.LOCK_NAME
    host = __import__("socket").gethostname()
    boot = run_suite_mod._boot_id()
    if not boot:
        pytest.skip("no /proc boot id on this platform")
    # 1) alive pid (this pytest process) but written under a different boot -> stale
    lock.write_text(f"pid={os.getpid()}\nhost={host}\nrole=child\nboot=not-this-boot\n")
    assert not run_suite_mod._lock_alive(lock, run_suite_mod._read_lock(lock))
    # 2) same boot, alive pid, but the process is not the run of this directory (pytest) -> stale
    lock.write_text(f"pid={os.getpid()}\nhost={host}\nrole=child\nboot={boot}\n")
    assert not run_suite_mod._lock_alive(lock, run_suite_mod._read_lock(lock))
    # 3) a legacy lock without boot id keeps the old semantics (pid alive -> live)
    lock.write_text(f"pid={os.getpid()}\nhost={host}\nrole=child\n")
    assert run_suite_mod._lock_alive(lock, run_suite_mod._read_lock(lock))
    # 4) a genuine child: alive process whose command line names the run directory -> live; dead -> stale
    proc = subprocess.Popen([_sys.executable, "-c", "import time; time.sleep(60)", "--out", str(run_dir)])
    try:
        lock.write_text(run_suite_mod._lock_text(proc.pid, "child"))
        info = run_suite_mod._read_lock(lock)
        assert info["boot"] == boot
        assert run_suite_mod._lock_alive(lock, info)
    finally:
        proc.kill()
        proc.wait()
    assert not run_suite_mod._lock_alive(lock, run_suite_mod._read_lock(lock))
    # 5) the runner's own (short-lived) lock is judged by pid + boot only
    lock.write_text(run_suite_mod._lock_text(os.getpid(), "runner"))
    assert run_suite_mod._lock_alive(lock, run_suite_mod._read_lock(lock))
