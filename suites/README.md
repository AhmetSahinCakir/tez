# Experiment suites

A suite is a YAML file describing a grid of runs (method × hyper-parameters × seeds) that
`scripts/run_suite.py` executes as independent subprocesses of `python -m plasticity.run`.

```yaml
name: pmnist_main                   # results go to results/<name>/   (default: file stem)
base_config: configs/pmnist_pilot.yaml   # relative to the project root (fallback: the suite's directory)
workers: 4                          # concurrent subprocesses; every child runs with threads=1
seeds: [0, 1, 2, 3, 4]              # or  n_seeds: 10  (-> 0..9)
stream_seed_offset: 0               # dev suites use 1000 so dev task streams differ from the test streams
common:                             # dotted overrides applied to every run (nested dicts are flattened)
  metrics.every_n_tasks: 5
selected_hparams: suites/selected_pmnist.yaml   # optional, written by scripts/select_hparams.py
runs:
  - label: baseline
    overrides: {method.name: baseline, optimizer.lr: 0.01}
  - label: sin
    overrides: {model.reparam.hidden: sin, model.reparam.output: sin}
    sweep: {optimizer.lr: [0.003, 0.01, 0.03]}   # cartesian product -> labels sin/lr=0.003, sin/lr=0.01, ...
    seeds: [0]                                   # per-run seeds / n_seeds / stream_seed_offset override
```

* Each `(label, seed)` pair becomes one run directory `results/<name>/<label>/seed<k>/` holding
  `config.json`, `tasks.jsonl`, `summary.json` and the child's `stdout.log`.
* Sweep keys append `/<last key component>=<value>` segments to the label (`sin/lr=0.01`,
  `weight_clipping/lr=0.01/kappa=2.0`). The *base label* of a run is the text before the first `/`.
* Override precedence (later wins): `common` < `selected_hparams[label]` < run `overrides` < `sweep`.
  A `selected_hparams` key applies to every run whose label equals it or starts with it on a `/` boundary,
  and the **longest** matching key wins: with keys `mnist` and `mnist/sin`, the run `mnist/sin/lr=0.01`
  takes `mnist/sin` and `mnist/tanh` takes `mnist`. This consumes both `select_hparams.py` groupings
  (`--group first` -> keys like `sin`; `--group sweep` -> keys like `mnist/sin`). Keys that match no run
  raise a warning.
* **Paths**: every relative path -- `base_config`, `selected_hparams`, `results_dir` in the suite and
  `--results-root` on the command line -- is resolved against the project root, never against the current
  working directory (`base_config` / `selected_hparams` additionally fall back to the suite's directory), so
  the script behaves the same from any directory.
* The child command is
  `python -m plasticity.run --config <base_config> --out <run dir> --set seed=<k> --set stream_seed_offset=<off> --set threads=1 --set <dotted>=<value> ...`;
  values are YAML-serialised so that `plasticity.utils._parse_scalar` reads back the same type
  (`0.01`, `1.0e-05`, `true`, `sin`, `'yes'`, `[256, 256]`, `null`).

## Running

```bash
python scripts/run_suite.py suites/pmnist_dev.yaml                 # resumable: finished runs are skipped
python scripts/run_suite.py suites/pmnist_dev.yaml --dry-run       # print the command list only
python scripts/run_suite.py suites/pmnist_dev.yaml --only sin --workers 2 --max-runs 3
python scripts/run_suite.py suites/pmnist_dev.yaml --force --only baseline/lr=0.01/seed0
python scripts/run_suite.py suites/pmnist_dev.yaml --seeds 0,1 --results-root /tmp/scratch --retries 0
```

Flags: `--workers`, `--dry-run`, `--force` (re-run completed jobs), `--only SUBSTR` (repeatable, matched
against `<label>/seed<k>`), `--max-runs N`, `--results-root DIR`, `--retries N` (default 1 retry on a
non-zero exit; a child killed by a signal is never retried), `--seeds a,b,c`, `--python`, `--quiet`. A
progress line with an ETA is printed per finished run. The exit code is non-zero if any run failed, was
interrupted or was locked by another invocation (130 after Ctrl-C). From Python:
`run_suite(path, workers=2, only=..., ...)` returns a `SuiteResult` with the `Job` list
(`status` in `ok | failed | skipped | locked | interrupted | pending | dry-run`).

* **Concurrency**: while a child runs, its run directory holds `running.lock` (`pid=`, `host=` of the
  child). A job whose lock belongs to a live process is reported as `locked` and not launched, so two
  invocations of the same suite (or a resumed one next to a still-running child) never write the same
  directory; a lock left by a dead process is taken over. Completion (`summary.json`) is re-checked at
  launch time, so a job finished by another invocation in the meantime is `skipped`.
* **Ctrl-C**: the runner stops launching new attempts, terminates its running children (SIGTERM, SIGKILL
  after 5 s), waits for the worker threads, writes `index.csv` (those jobs are `interrupted`, never
  retried) and exits with 130. Re-running the suite afterwards resumes exactly the unfinished jobs.

Outputs in `results/<name>/`:

* `index.csv` – one row per job of the grid: `label, seed, status, wall_time` (training time from `summary.json`), `elapsed` (subprocess time measured by the runner), `attempts, returncode, run_dir`
  and the main `summary.json` metrics (`metric, auc_norm, early_window_mean, final_window_mean,
  retention_ratio, drop, slope_per_100, fresh_gap_final, final_test_accuracy, ...`).
* `suite.json` – the resolved suite and the expanded job list with each job's dotted `overrides` and
  `sweep` keys (provenance; `select_hparams.py` uses it to map label segments back to config keys).

## Hyper-parameter selection (dev suites)

```bash
python scripts/select_hparams.py --results results/pmnist_dev --out suites/selected_pmnist.yaml \
    --metric auc_norm --min-seeds 1
```

Groups variants by base label, averages the metric over seeds, picks the best variant per base label and
writes `{metric, results, selected: {base: {dotted.key: value}}, details: {...}}` and a markdown table of all
variants (`<out>.md`). The overrides are recovered from `config.json`: the dotted keys that vary across the
group (restricted to the sweep/override keys of `suite.json` when present) plus the keys named by the
label's `k=v` segments. A segment such as `lr=0.01` is mapped to a dotted key using, in order, the sweep key
recorded in `suite.json`, the unique matching key that varies across the group, the unique one with the same
value, or the only candidate; an ambiguous segment (e.g. `optimizer.lr` and `method.lr` both equal) is
reported as a warning and left out rather than guessed, and a bare non-dotted key is never emitted.
Point a final suite at the YAML with `selected_hparams:` and the chosen overrides are merged into every run
whose label matches (longest prefix, see above).

Options: `--metric` (default `auc_norm`), `--lower-is-better`, `--min-seeds N`, `--table PATH`, and
`--group first|sweep`. `--group sweep` strips only the trailing `k=v` segments, which is the right grouping for
suites whose labels nest the method under a dataset, e.g. `suites/stationary.yaml` (`mnist/sin/lr=0.01` ->
base `mnist/sin`); the default `first` grouping would merge `mnist/standard`, `mnist/sin` and `mnist/tanh`
into one base `mnist`.
