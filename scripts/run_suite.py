#!/usr/bin/env python
"""Parallel grid runner for experiment suites (method x hyper-parameter x seed grids).

A *suite* is a YAML file (see ``suites/README.md``)::

    name: pmnist_main
    base_config: configs/pmnist_pilot.yaml
    workers: 4                      # concurrent subprocesses (each child runs with threads=1)
    seeds: [0, 1, 2, 3, 4]          # or  n_seeds: 10  (-> 0..9); a run may override 'seeds' / 'n_seeds'
    stream_seed_offset: 0           # dev suites use e.g. 1000 so that dev streams differ from test streams
    common: {metrics.every_n_tasks: 5}         # dotted overrides applied to every run
    selected_hparams: suites/selected_pmnist.yaml   # optional: per-base-label overrides (select_hparams.py)
    runs:
      - label: baseline
        overrides: {method.name: baseline, optimizer.lr: 0.01}
      - label: sin
        overrides: {model.reparam.hidden: sin, model.reparam.output: sin}
        sweep: {optimizer.lr: [0.003, 0.01, 0.03]}   # cartesian expansion -> labels 'sin/lr=0.01', ...

Every (label, seed) pair becomes one independent subprocess::

    python -m plasticity.run --config <base_config> --out results/<name>/<label>/seed<k> \
        --set seed=<k> --set stream_seed_offset=<offset> --set threads=1 --set <dotted>=<value> ...

Override precedence (later wins): ``common`` < ``selected_hparams[label]`` < run ``overrides`` < ``sweep``
values.  A ``selected_hparams`` entry applies to every run whose label starts with the entry's key on a
``/`` boundary; the *longest* matching key wins (``mnist/sin`` beats ``mnist`` for the run ``mnist/sin/lr=0.01``),
so both ``select_hparams.py --group first`` (keys like ``sin``) and ``--group sweep`` (keys like
``mnist/sin``) outputs are consumed; keys that match no run raise a warning.  Values are serialised with
YAML so that :func:`plasticity.utils._parse_scalar` reads them back with the same type (floats, ints,
bools, strings, lists); nested dict values are flattened to dotted keys.

Relative paths in a suite (``base_config``, ``selected_hparams``, ``results_dir``) and on the command line
(``--results-root``) are resolved against the project root (``base_config`` / ``selected_hparams`` fall back
to the suite's directory), never against the current working directory.

Features: completed runs (``summary.json`` present) are skipped so a suite is resumable (``--force``
re-runs them), ``--dry-run`` prints the command list, ``--only SUBSTR`` filters labels, ``--max-runs N``
caps the number of launched runs, failed runs are retried (``--retries``, default 1; a child killed by a
signal is *not* retried), a progress line with an ETA is printed, each child's stdout/stderr goes to
``<run dir>/stdout.log`` and at the end ``results/<name>/index.csv`` lists every run (label, seed,
status, wall time, main summary metrics).  Children are launched from a ``ThreadPoolExecutor`` with
``subprocess.Popen`` (one OS process per run, with its own thread count).  While a child runs, its run
directory holds ``running.lock`` (pid + host of the child); a job whose lock belongs to a live process
is reported as ``locked`` and not launched, so two invocations of the same suite never write the same
run directory.  On Ctrl-C the runner stops launching, terminates its running children (SIGTERM, then
SIGKILL after a grace period), waits for the workers, writes ``index.csv`` (those jobs are
``interrupted``) and re-raises ``KeyboardInterrupt`` (exit code 130).  The exit code is non-zero if any
run failed, was interrupted or was locked by another invocation.

Python API: ``run_suite(suite_path, **flags) -> SuiteResult`` (used by ``tests/test_suite_runner.py``).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/run_suite.py` from anywhere
    sys.path.insert(0, str(PROJECT_ROOT))

from plasticity.utils import _parse_scalar, flatten_dict  # noqa: E402

__all__ = ["Job", "SuiteResult", "LOCK_NAME", "load_suite", "results_root_of", "expand_jobs", "build_command",
           "serialise_value", "match_selected", "run_suite", "main"]

# summary.json keys copied into index.csv (those present in at least one summary, in this order)
INDEX_METRICS: Sequence[str] = (
    "metric", "auc_norm", "early_window_mean", "final_window_mean", "retention_ratio", "drop", "slope_per_100",
    "fresh_gap_final", "fresh_gap_mean", "final_test_accuracy", "final_train_accuracy", "epochs", "n_tasks",
    "global_steps", "stream_seed", "method", "stream",
)
_SCI_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+")
LOCK_NAME = "running.lock"          # marker written into a run directory while a child is running
LOCK_FRESH_SECONDS = 60.0           # an unreadable/empty lock younger than this is assumed to be live
TERMINATE_GRACE_SECONDS = 5.0       # SIGTERM -> SIGKILL grace period on interrupt


# ------------------------------------------------------------------------------------------------
# value helpers
# ------------------------------------------------------------------------------------------------
def _coerce(v: Any) -> Any:
    """Normalise a suite value: tuples -> lists, '1e-4'-style strings (a str in YAML 1.1) -> float."""
    if isinstance(v, tuple):
        return [_coerce(x) for x in v]
    if isinstance(v, list):
        return [_coerce(x) for x in v]
    if isinstance(v, str) and _SCI_NUMBER.fullmatch(v.strip()):
        return float(v)
    return v


def serialise_value(v: Any) -> str:
    """Render an override value so that ``plasticity.utils._parse_scalar`` (YAML) reads it back unchanged.

    ``0.01 -> '0.01'``, ``1e-05 -> '1.0e-05'``, ``True -> 'true'``, ``'sin' -> 'sin'``, ``'yes' -> "'yes'"``,
    ``[256, 256] -> '[256, 256]'``, ``None -> 'null'``.
    """
    v = _coerce(v)
    if isinstance(v, dict):
        raise TypeError("nested dict values must be flattened to dotted keys before serialisation")
    text = yaml.safe_dump(v, default_flow_style=True, width=1 << 20, allow_unicode=True)
    if text.endswith("\n...\n"):
        text = text[:-5]
    text = text.strip()
    back = _parse_scalar(text)
    if back != v or type(back) is not type(v):
        raise ValueError(f"override value {v!r} does not survive the YAML round trip (got {back!r})")
    return text


def _label_value(v: Any) -> str:
    """Compact, file-system safe rendering of a sweep value for run labels (``lr=0.01``, ``kappa=2.0``)."""
    v = _coerce(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return "[" + ",".join(_label_value(x) for x in v) + "]"
    if v is None:
        return "null"
    return re.sub(r"[\s/\\]+", "_", str(v))


def _fmt_secs(s: float) -> str:
    s = max(0, int(round(s)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# ------------------------------------------------------------------------------------------------
# suite spec -> jobs
# ------------------------------------------------------------------------------------------------
@dataclass
class Job:
    label: str                      # full label incl. sweep segments, e.g. 'sin/lr=0.01'
    seed: int
    out_dir: Path
    overrides: Dict[str, Any]       # dotted key -> value (common + selected + run + sweep)
    stream_seed_offset: int = 0
    sweep: Dict[str, Any] = field(default_factory=dict)   # the sweep keys/values of this job (dotted keys)
    cmd: List[str] = field(default_factory=list)
    status: str = "pending"         # pending | dry-run | skipped | ok | failed | interrupted | locked
    wall_time: float = float("nan")
    returncode: Optional[int] = None
    attempts: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)
    note: str = ""                  # e.g. which process holds the lock of a 'locked' job

    @property
    def name(self) -> str:
        return f"{self.label}/seed{self.seed}"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "summary.json"

    @property
    def lock_path(self) -> Path:
        return self.out_dir / LOCK_NAME

    @property
    def complete(self) -> bool:
        return self.summary_path.exists()


@dataclass
class SuiteResult:
    name: str
    results_root: Path
    jobs: List[Job]                 # every job of the grid (in suite order)
    launched: List[Job]             # jobs executed (or printed) by this invocation
    index_path: Optional[Path] = None
    dry_run: bool = False
    elapsed: float = 0.0

    def count(self, status: str) -> int:
        return sum(1 for j in self.jobs if j.status == status)

    @property
    def n_ok(self) -> int:
        return self.count("ok")

    @property
    def n_failed(self) -> int:
        return self.count("failed")

    @property
    def n_interrupted(self) -> int:
        return self.count("interrupted")

    @property
    def n_locked(self) -> int:
        return self.count("locked")

    @property
    def n_skipped(self) -> int:
        return self.count("skipped")

    @property
    def ok(self) -> bool:
        """True when no launched job failed, was interrupted or was locked by another invocation."""
        return self.n_failed == 0 and self.n_interrupted == 0 and self.n_locked == 0


def _resolve(path: str | os.PathLike, suite_dir: Path) -> Path:
    """Resolve a path given in a suite file: absolute, else relative to the project root, else to the suite dir.

    The current working directory is deliberately *not* used, so that ``results_dir`` / ``--results-root``
    (see :func:`results_root_of`) and ``base_config`` / ``selected_hparams`` resolve the same way wherever
    the script is started from.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (PROJECT_ROOT, Path(suite_dir)):
        if (base / p).exists():
            return (base / p).resolve()
    raise FileNotFoundError(f"{path} not found (relative paths are resolved against {PROJECT_ROOT} and {suite_dir})")


def _seeds(node: Dict[str, Any], default: Optional[List[int]] = None) -> Optional[List[int]]:
    if "seeds" in node and node["seeds"] is not None:
        seeds = node["seeds"]
        if isinstance(seeds, (int, str)):
            seeds = [int(s) for s in str(seeds).split(",") if str(s).strip()]
        return [int(s) for s in seeds]
    if "n_seeds" in node and node["n_seeds"] is not None:
        return list(range(int(node["n_seeds"])))
    return default


def _load_selected(path: Optional[str], suite_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load a ``select_hparams.py`` output (``{selected: {base_label: {dotted: value}}}`` or a bare mapping)."""
    if not path:
        return {}
    with open(_resolve(path, suite_dir), "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    mapping = doc.get("selected", doc) if isinstance(doc, dict) else {}
    return {str(k).strip().strip("/"): flatten_dict(v or {}) for k, v in mapping.items() if isinstance(v, dict) or v is None}


def match_selected(label: str, selected: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Key of ``selected`` that applies to ``label``: the longest key equal to the label or to one of its
    leading ``/`` segments (``'mnist/sin/lr=0.01'`` tries ``mnist/sin/lr=0.01``, ``mnist/sin``, ``mnist``)."""
    parts = label.split("/")
    for n in range(len(parts), 0, -1):
        key = "/".join(parts[:n])
        if key in selected:
            return key
    return None


def load_suite(suite_path: str | os.PathLike) -> Dict[str, Any]:
    """Read and validate a suite YAML; returns the spec with ``_suite_path`` / ``_suite_dir`` / ``base_config`` resolved."""
    suite_path = Path(suite_path).resolve()
    with open(suite_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    if not isinstance(spec, dict):
        raise ValueError(f"{suite_path}: suite file must be a mapping")
    spec.setdefault("name", suite_path.stem)
    if "base_config" not in spec:
        raise ValueError(f"{suite_path}: 'base_config' is required")
    if not spec.get("runs"):
        raise ValueError(f"{suite_path}: 'runs' must be a non-empty list")
    spec["_suite_path"] = str(suite_path)
    spec["_suite_dir"] = str(suite_path.parent)
    spec["base_config"] = str(_resolve(spec["base_config"], suite_path.parent))
    return spec


def results_root_of(spec: Dict[str, Any], results_root: Optional[str | os.PathLike] = None) -> Path:
    """Output root of a suite: the explicit argument, else the suite's ``results_dir``, else ``results/<name>``.

    Relative paths are resolved against the project root (like every other path of a suite)."""
    root = Path(results_root) if results_root is not None else Path(spec.get("results_dir") or PROJECT_ROOT / "results" / spec["name"])
    return root if root.is_absolute() else (PROJECT_ROOT / root)


def expand_jobs(spec: Dict[str, Any], results_root: Optional[str | os.PathLike] = None,
                seeds: Optional[Iterable[int]] = None) -> List[Job]:
    """Expand the suite spec into the full (label, seed) grid (no commands yet)."""
    suite_dir = Path(spec.get("_suite_dir", PROJECT_ROOT / "suites"))
    root = results_root_of(spec, results_root)
    common = {k: _coerce(v) for k, v in flatten_dict(spec.get("common") or {}).items()}
    selected = _load_selected(spec.get("selected_hparams"), suite_dir)
    used_selected = set()
    suite_seeds = list(seeds) if seeds is not None else (_seeds(spec, [0]) or [0])
    offset = int(spec.get("stream_seed_offset", 0) or 0)
    jobs: List[Job] = []
    seen = set()
    for run in spec["runs"]:
        if "label" not in run:
            raise ValueError(f"every run needs a 'label': {run}")
        label = str(run["label"]).strip().strip("/")
        if not label or any(part in ("", ".", "..") for part in label.split("/")):
            raise ValueError(f"invalid run label {run['label']!r}")
        run_over = {k: _coerce(v) for k, v in flatten_dict(run.get("overrides") or {}).items()}
        sweep = run.get("sweep") or {}
        keys = list(sweep)
        values = [list(sweep[k]) if isinstance(sweep[k], (list, tuple)) else [sweep[k]] for k in keys]
        run_seeds = list(seeds) if seeds is not None else (_seeds(run, suite_seeds) or suite_seeds)
        run_offset = int(run.get("stream_seed_offset", offset))
        for combo in (itertools.product(*values) if keys else [()]):
            full_label = label + "".join(f"/{k.split('.')[-1]}={_label_value(v)}" for k, v in zip(keys, combo))
            sweep_vals = {k: _coerce(v) for k, v in zip(keys, combo)}
            ov: Dict[str, Any] = dict(common)
            sel_key = match_selected(full_label, selected)
            if sel_key is not None:
                used_selected.add(sel_key)
                ov.update(selected[sel_key])
            ov.update(run_over)
            ov.update(sweep_vals)
            for seed in run_seeds:
                key = (full_label, int(seed))
                if key in seen:
                    raise ValueError(f"duplicate run {full_label}/seed{seed} in suite {spec['name']}")
                seen.add(key)
                jobs.append(Job(label=full_label, seed=int(seed), out_dir=root / full_label / f"seed{seed}",
                                overrides=ov, stream_seed_offset=run_offset, sweep=sweep_vals))
    unused = sorted(set(selected) - used_selected)
    if unused:
        warnings.warn(f"suite {spec['name']}: selected_hparams keys match no run label: {', '.join(unused)}", stacklevel=2)
    return jobs


def build_command(job: Job, base_config: str | os.PathLike, python: Optional[str] = None, threads: int = 1,
                  quiet: bool = False) -> List[str]:
    cmd = [python or sys.executable, "-m", "plasticity.run", "--config", str(base_config), "--out", str(job.out_dir)]
    if quiet:
        cmd.append("--quiet")
    cmd += ["--set", f"seed={job.seed}", "--set", f"stream_seed_offset={job.stream_seed_offset}", "--set", f"threads={int(threads)}"]
    for k, v in job.overrides.items():
        cmd += ["--set", f"{k}={serialise_value(v)}"]
    return cmd


# ------------------------------------------------------------------------------------------------
# run-directory lock (one process per run directory)
# ------------------------------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _boot_id() -> str:
    """Kernel boot id (Linux); changes when the container / machine is replaced, so that a lock written
    before the replacement can never be mistaken for a live one even if its pid was reused."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _proc_cmdline(pid: int) -> Optional[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", "replace") if raw else None


def _lock_text(pid: int, role: str) -> str:
    boot = _boot_id()
    return (f"pid={pid}\nhost={socket.gethostname()}\nrole={role}\nstarted={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            + (f"boot={boot}\n" if boot else ""))


def _read_lock(path: Path) -> Dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    info: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def _lock_alive(path: Path, info: Dict[str, str]) -> bool:
    """Is the process named by a lock file still alive?  Locks of other hosts cannot be checked -> alive.

    A pid alone is not proof of life: after a container replacement pids restart from 1 and a stale lock's
    pid is soon reused by an unrelated process.  Locks therefore carry the kernel boot id (a different boot
    id means the owner is gone), and for a child lock the live process must also be the run of this very
    directory (its command line names the run directory)."""
    if not info.get("pid"):  # being written right now (or unreadable): trust it while it is fresh
        try:
            return time.time() - path.stat().st_mtime < LOCK_FRESH_SECONDS
        except OSError:
            return False
    if info.get("host") and info["host"] != socket.gethostname():
        return True
    try:
        pid = int(info["pid"])
    except ValueError:
        return False
    if not _pid_alive(pid):
        return False
    boot = info.get("boot")
    if boot:
        current = _boot_id()
        if current and current != boot:
            return False  # written before a reboot / container replacement: the pid has been reused
        if info.get("role") == "child":
            cmd = _proc_cmdline(pid)
            if cmd is not None and str(path.parent) not in cmd and str(path.parent.resolve()) not in cmd:
                return False  # the pid is alive but belongs to a different process now
    return True


def _acquire_lock(out_dir: Path) -> Optional[Dict[str, str]]:
    """Create ``running.lock`` in ``out_dir`` atomically (O_EXCL).

    Returns ``None`` on success; otherwise the lock info of the *live* owner.  A stale lock (owner dead)
    is taken over: it is renamed away first (atomic) so that only one of several contenders removes it.
    """
    path = out_dir / LOCK_NAME
    for _ in range(3):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            info = _read_lock(path)
            if _lock_alive(path, info):
                return info or {"pid": "?"}
            stale = path.with_name(f"{LOCK_NAME}.stale.{os.getpid()}.{threading.get_ident()}")
            try:
                os.rename(path, stale)
                stale.unlink()
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_lock_text(os.getpid(), "runner"))
        return None
    return _read_lock(path) or {"pid": "?"}


def _release_lock(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------
# execution
# ------------------------------------------------------------------------------------------------
class _RunContext:
    """Shared state of one ``run_suite`` invocation: the stop flag and the running child processes."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.children: Dict[str, subprocess.Popen] = {}

    def register(self, name: str, proc: subprocess.Popen) -> None:
        with self.lock:
            self.children[name] = proc

    def unregister(self, name: str) -> None:
        with self.lock:
            self.children.pop(name, None)

    def snapshot(self) -> List[subprocess.Popen]:
        with self.lock:
            return list(self.children.values())

    def terminate_children(self, grace: float = TERMINATE_GRACE_SECONDS) -> None:
        """SIGTERM every running child, then SIGKILL those still registered after ``grace`` seconds.

        The worker threads own ``Popen.wait``; they unregister a child once it has exited, so this only
        signals and polls the registry (never calls ``wait`` from the main thread)."""
        for proc in self.snapshot():
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass
        deadline = time.monotonic() + max(0.0, grace)
        while self.snapshot() and time.monotonic() < deadline:
            time.sleep(0.05)
        for proc in self.snapshot():
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass


def _child_env(threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(int(threads))
    return env


def _run_job(job: Job, env: Dict[str, str], retries: int, ctx: Optional[_RunContext] = None, force: bool = False) -> Job:
    """Run one job (with retries) in the calling worker thread; returns the job with its status filled in."""
    ctx = ctx or _RunContext()
    if ctx.stop.is_set():
        job.status = "interrupted"
        return job
    job.out_dir.mkdir(parents=True, exist_ok=True)
    if job.complete and not force:  # finished by another invocation since the grid was expanded
        job.status = "skipped"
        return job
    owner = _acquire_lock(job.out_dir)
    if owner is not None:
        job.status = "locked"
        job.note = f"pid {owner.get('pid', '?')} on {owner.get('host', '?')}"
        return job
    t0 = time.time()
    killed = False
    try:
        if job.summary_path.exists():  # --force: never leave a stale summary behind a failed re-run
            job.summary_path.unlink()
        with open(job.out_dir / "stdout.log", "w", encoding="utf-8") as log:
            for attempt in range(1, max(0, int(retries)) + 2):
                if ctx.stop.is_set():
                    log.write("# interrupted before attempt %d\n" % attempt)
                    break
                job.attempts = attempt
                log.write(f"# attempt {attempt}: {' '.join(shlex.quote(c) for c in job.cmd)}\n")
                log.flush()
                killed = False
                try:
                    proc = subprocess.Popen(job.cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT), env=env)
                except OSError as exc:  # e.g. interpreter not found
                    job.returncode = -1
                    log.write(f"# launch error: {exc}\n")
                else:
                    ctx.register(job.name, proc)
                    try:
                        job.lock_path.write_text(_lock_text(proc.pid, "child"), encoding="utf-8")
                        if ctx.stop.is_set():   # interrupted between launch and registration
                            proc.terminate()
                        job.returncode = proc.wait()
                    finally:
                        ctx.unregister(job.name)
                    killed = job.returncode is not None and job.returncode < 0
                if job.returncode == 0 and job.summary_path.exists():
                    break
                if killed:
                    try:
                        sig = signal.Signals(-job.returncode).name
                    except ValueError:
                        sig = str(-job.returncode)
                    log.write(f"# attempt {attempt} killed by signal {sig}; not retried\n")
                    log.flush()
                    break
                log.write(f"# attempt {attempt} failed (exit code {job.returncode})\n")
                log.flush()
    finally:
        _release_lock(job.lock_path)
    job.wall_time = time.time() - t0
    if job.returncode == 0 and job.summary_path.exists():
        job.status = "ok"
        job.summary = _read_json(job.summary_path)
    elif ctx.stop.is_set() or (killed and job.returncode == -int(signal.SIGINT)):
        job.status = "interrupted"
    else:
        job.status = "failed"
    return job


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_index(jobs: Sequence[Job], results_root: Path) -> Path:
    """Write ``index.csv`` (one row per job of the grid) and return its path."""
    for job in jobs:
        if not job.summary and job.complete:
            job.summary = _read_json(job.summary_path)
    metrics = [k for k in INDEX_METRICS if any(k in j.summary for j in jobs)]
    # wall_time = training time recorded in summary.json (same meaning for skipped and fresh runs);
    # elapsed   = end-to-end subprocess time measured by the runner (only for runs launched by this invocation)
    cols = ["label", "seed", "status", "wall_time", "elapsed", "attempts", "returncode", "run_dir"] + metrics
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / "index.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for job in sorted(jobs, key=lambda j: (j.label, j.seed)):
            wall = job.summary.get("wall_time", "")
            elapsed = f"{job.wall_time:.1f}" if job.wall_time == job.wall_time else ""   # NaN -> not launched
            try:
                run_dir = str(job.out_dir.relative_to(results_root))
            except ValueError:
                run_dir = str(job.out_dir)
            row: List[Any] = [job.label, job.seed, job.status, f"{wall:.2f}" if isinstance(wall, float) else wall, elapsed,
                              job.attempts or "", "" if job.returncode is None else job.returncode, run_dir]
            row += [job.summary.get(k, "") for k in metrics]
            w.writerow(row)
    return path


def run_suite(suite_path: str | os.PathLike, workers: Optional[int] = None, dry_run: bool = False, force: bool = False,
              only: Optional[str | Sequence[str]] = None, max_runs: Optional[int] = None,
              results_root: Optional[str | os.PathLike] = None, retries: int = 1, python: Optional[str] = None,
              quiet: bool = False, seeds: Optional[Iterable[int]] = None, verbose: bool = True) -> SuiteResult:
    """Run (or list) every job of a suite; see the module docstring for the flags."""
    spec = load_suite(suite_path)
    jobs = expand_jobs(spec, results_root=results_root, seeds=seeds)
    root = results_root_of(spec, results_root)
    threads = int(spec.get("threads", 1) or 1)
    n_workers = int(workers or spec.get("workers") or 1)
    for job in jobs:
        job.cmd = build_command(job, spec["base_config"], python=python, threads=threads, quiet=quiet)

    # ---- select the jobs to launch -------------------------------------------------------------
    patterns = [only] if isinstance(only, str) else list(only or [])
    selected: List[Job] = []
    for job in jobs:
        if patterns and not any(p in job.name for p in patterns):
            job.status = "skipped" if job.complete else "pending"
            continue
        if job.complete and not force:
            job.status = "skipped"
            continue
        selected.append(job)
    if max_runs is not None:
        selected = selected[: max(0, int(max_runs))]
    launched = list(selected)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    result = SuiteResult(name=spec["name"], results_root=root, jobs=jobs, launched=launched, dry_run=dry_run)
    log(f"[{spec['name']}] {len(jobs)} runs in grid, {sum(j.complete for j in jobs)} complete, "
        f"{len(launched)} to launch with {n_workers} worker(s) -> {root}")

    if dry_run:
        for job in launched:
            job.status = "dry-run"
            log(" ".join(shlex.quote(c) for c in job.cmd))
        return result
    if not launched:
        result.index_path = write_index(jobs, root)
        log(f"[{spec['name']}] nothing to do; index -> {result.index_path}")
        return result

    # ---- execute --------------------------------------------------------------------------------
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "suite.json", "w", encoding="utf-8") as f:
        json.dump({"suite": {k: v for k, v in spec.items() if not k.startswith("_")}, "suite_path": spec["_suite_path"],
                   "jobs": [{"label": j.label, "seed": j.seed, "out_dir": str(j.out_dir), "overrides": j.overrides,
                             "sweep": j.sweep, "stream_seed_offset": j.stream_seed_offset} for j in jobs]}, f, indent=1, default=str)
    env = _child_env(threads)
    ctx = _RunContext()
    t0 = time.time()
    n_done = n_failed = 0
    executor = ThreadPoolExecutor(max_workers=max(1, n_workers), thread_name_prefix="run_suite")
    futures = {}
    try:
        for job in launched:
            futures[executor.submit(_run_job, job, env, retries, ctx, force)] = job
        for fut in as_completed(futures):
            job = fut.result()
            n_done += 1
            n_failed += job.status != "ok"
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(launched) - n_done)
            extra = f", {job.attempts} attempts" if job.attempts > 1 else ""
            extra += f" [{job.note}]" if job.note else ""
            log(f"[{spec['name']}] {n_done}/{len(launched)} done | {n_failed} failed | elapsed {_fmt_secs(elapsed)} | "
                f"ETA {_fmt_secs(eta)} | {job.name}: {job.status} ({job.wall_time:.1f}s{extra})")
    except KeyboardInterrupt:
        log(f"[{spec['name']}] interrupted; cancelling pending runs and terminating running children ...")
        ctx.stop.set()                                   # no further attempts / retries in the workers
        executor.shutdown(wait=False, cancel_futures=True)
        ctx.terminate_children()                         # SIGTERM, then SIGKILL after the grace period
        executor.shutdown(wait=True)                     # workers reap their children and release the locks
        for job in launched:
            if job.status == "pending":                  # cancelled before it started
                job.status = "interrupted"
        result.elapsed = time.time() - t0
        result.index_path = write_index(jobs, root)
        log(f"[{spec['name']}] stopped after {_fmt_secs(result.elapsed)}: {result.n_ok} ok, {result.n_failed} failed, "
            f"{result.n_interrupted} interrupted; index -> {result.index_path}")
        raise
    executor.shutdown(wait=True)
    result.elapsed = time.time() - t0
    result.index_path = write_index(jobs, root)
    locked = f", {result.n_locked} locked by another process" if result.n_locked else ""
    log(f"[{spec['name']}] finished: {result.n_ok} ok, {result.n_failed} failed, {result.n_skipped} skipped{locked} "
        f"in {_fmt_secs(result.elapsed)}; index -> {result.index_path}")
    return result


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run every (run label, seed) job of a suite YAML as parallel subprocesses.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("suite", nargs="?", default=None, help="suite YAML (see suites/README.md)")
    ap.add_argument("--suite", dest="suite_opt", default=None, help=argparse.SUPPRESS)   # alias of the positional
    ap.add_argument("--workers", type=int, default=None, help="concurrent subprocesses (default: suite 'workers' or 1)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands that would run and exit")
    ap.add_argument("--force", action="store_true", help="re-run jobs whose summary.json already exists")
    ap.add_argument("--only", action="append", default=None, metavar="SUBSTR",
                    help="only jobs whose '<label>/seed<k>' contains SUBSTR (repeatable; any match)")
    ap.add_argument("--max-runs", type=int, default=None, help="launch at most N jobs this invocation")
    ap.add_argument("--results-root", default=None,
                    help="output root (default: results/<suite name>; a relative path is taken from the project root)")
    ap.add_argument("--retries", type=int, default=1, help="retries per failed job (default 1)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds overriding the suite's seeds")
    ap.add_argument("--python", default=None, help="interpreter for the children (default: this one)")
    ap.add_argument("--quiet", action="store_true", help="pass --quiet to plasticity.run (no per-task progress)")
    args = ap.parse_args(argv)
    suite = args.suite or args.suite_opt
    if not suite:
        ap.error("a suite YAML is required")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else None
    try:
        res = run_suite(suite, workers=args.workers, dry_run=args.dry_run, force=args.force, only=args.only,
                        max_runs=args.max_runs, results_root=args.results_root, retries=args.retries, python=args.python,
                        quiet=args.quiet, seeds=seeds, verbose=True)
    except KeyboardInterrupt:
        return 130
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
