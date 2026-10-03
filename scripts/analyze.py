#!/usr/bin/env python
"""Turn a results suite into tables and figures.

    python scripts/analyze.py --results results/pmnist_canonical --out reports/pmnist_canonical \
        [--reference baseline] [--metric auto|online_accuracy|test_accuracy] [--labels a,b,c] [--smooth 5] [--lang tr]
        [--sweep cfg.optimizer.lr[:group_key]] [--mechanism-keys k1,k2] [--format csv|parquet|auto] [--n-boot 10000]
        [--figures all|performance,mechanism,summary,fresh_gap,sweep] [--no-figures]

Writes ``runs.csv``, ``curves.csv`` (or ``.parquet``), ``summary.md/csv``, ``comparison.md/csv`` and the figures
``performance``, ``mechanism``, ``summary``, ``fresh_gap`` (when fresh-reference data exist) and ``sweep``
(when ``--sweep`` is given) as png + pdf, then prints the summary table.  Missing metrics never abort the
report: each product is produced independently and failures are reported on stderr.

``--labels`` entries match a full label or a leading path of a sweep label (``sin`` selects ``sin/lr=0.01``,
``sin/lr=0.03`` ...); entries matching nothing are reported.  ``--sweep X[:GROUP]`` draws the run-level
metric (``auc_norm`` else ``final_<metric>`` else ``final_window_mean``) against the config column ``X``
with one line per ``GROUP``: by default the *base label* (the label without its ``/lr=...`` sweep
segments, so each method is one sensitivity curve); ``label`` uses the full label and any other column
of ``runs`` (e.g. ``cfg.method.name``) is accepted.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plasticity.analysis import aggregate as A  # noqa: E402
from plasticity.analysis import plots as P  # noqa: E402
from plasticity.analysis import tables as Tb  # noqa: E402


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="suite directory: results/<suite>")
    ap.add_argument("--out", required=True, help="output directory: reports/<suite>")
    ap.add_argument("--reference", default=None, help="reference label for the paired comparison (default: baseline-like label, else first)")
    ap.add_argument("--metric", default="auto", help="auto | online_accuracy | test_accuracy | any per-task key")
    ap.add_argument("--labels", default=None, help="comma-separated labels to keep (display order); a base label such as 'sin' "
                    "also selects its sweep variants 'sin/lr=...'")
    ap.add_argument("--smooth", type=int, default=None, help="moving-average window (tasks) for the curves")
    ap.add_argument("--lang", default="tr", choices=sorted(P.LABELS), help="figure/table language")
    ap.add_argument("--format", default="auto", choices=("auto", "csv", "parquet"), help="runs/curves table format")
    ap.add_argument("--mechanism-keys", default=None, help="comma-separated mechanism keys for the panel figure")
    ap.add_argument("--sweep", default=None, help="x config key for a sensitivity figure, optionally ':<group column>' "
                    "(default base_label; e.g. cfg.optimizer.lr, cfg.optimizer.lr:label, cfg.optimizer.lr:cfg.method.name)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--figures", default="all", help="comma-separated subset of performance,mechanism,summary,fresh_gap,sweep (default all)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def _pick_reference(runs, labels):
    modes = P.infer_modes(runs, labels)
    for lab in labels:
        if modes.get(lab) == "standard":
            return lab
    return labels[0] if labels else None


def _resolve_reference(requested, present, runs, order, quiet):
    """The reference label: exact match, else the single sweep label below the request, else a baseline-like label."""
    if requested is not None:
        if requested in present:
            return requested
        hits, _ = A.match_labels([requested], present)
        if len(hits) == 1:
            return hits[0]
        if not quiet:
            why = "not found" if not hits else f"ambiguous ({len(hits)} labels match: {hits})"
            print(f"[analyze] reference {requested!r} {why}", file=sys.stderr)
    return _pick_reference(runs, order)


def _run_level_metric(runs, metric):
    """A column of ``runs`` to summarise the per-task ``metric``: auc_norm, else final_<metric>, else final_window_mean."""
    for key in ("auc_norm", f"final_{metric}", "final_window_mean", "last_task_perf"):
        if key in runs.columns:
            return key
    return metric  # absent as well -> the figure shows its 'no data' placeholder instead of crashing


def _guard(name, fn, errors, quiet):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - the report must survive a single failing product
        errors.append((name, e))
        if not quiet:
            print(f"[analyze] {name}: skipped ({type(e).__name__}: {e})", file=sys.stderr)
            traceback.print_exc(limit=2, file=sys.stderr)
        return None


def main(argv=None) -> int:
    args = _parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()] if args.labels else None
    runs, curves, skipped = A.load_suite(args.results, labels=labels, verbose=not args.quiet, return_skipped=True)
    unmatched = runs.attrs.get("unmatched_labels", [])
    if len(runs) == 0:
        print(f"[analyze] no finished runs under {args.results} (skipped: {skipped}; requested labels matching nothing: {unmatched})",
              file=sys.stderr)
        return 1
    errors = []
    paths = A.save_tables(runs, curves, out, fmt=args.format)
    present = [str(l) for l in runs["label"].unique()]
    order = A.match_labels(labels, present)[0] if labels else P.order_labels(present, runs=runs)
    P.make_palette(order, runs=runs)  # fix the colour of every label once, for all figures
    metric = A.auto_metric(runs, curves) if args.metric == "auto" else args.metric
    reference = _resolve_reference(args.reference, present, runs, order, args.quiet)
    if reference != args.reference and not args.quiet:
        print(f"[analyze] reference: using {reference!r}", file=sys.stderr)
    if not args.quiet:
        print(f"[analyze] {len(runs)} runs, {len(order)} labels, metric={metric}, reference={reference}, "
              f"skipped={len(skipped)}; tables -> {paths['runs']}, {paths['curves']}")

    summary = _guard("summary tables", lambda: Tb.write_summary_tables(runs, out, labels=order, lang=args.lang), errors, args.quiet)
    comparison = _guard("comparison tables", lambda: Tb.write_comparison_tables(
        runs, out, reference=reference, labels=order, lang=args.lang, n_boot=args.n_boot), errors, args.quiet)

    if not args.no_figures:
        def perf():
            fig = P.plot_performance_curves(curves, metric, labels=order, smooth=args.smooth, lang=args.lang, runs=runs)
            return P.save_figure(fig, out / "performance", dpi=args.dpi, close=True)

        def mech():
            keys = [k.strip() for k in args.mechanism_keys.split(",")] if args.mechanism_keys else None
            fig = P.plot_mechanism_panels(curves, keys=keys, labels=order, smooth=args.smooth, lang=args.lang, runs=runs)
            return P.save_figure(fig, out / "mechanism", dpi=args.dpi, close=True)

        def summ():
            fig = P.plot_summary_dots(runs, _run_level_metric(runs, metric), labels=order, reference=reference, lang=args.lang)
            return P.save_figure(fig, out / "summary", dpi=args.dpi, close=True)

        def fresh():
            if not A.has_fresh_data(curves):
                return None
            fig = P.plot_fresh_gap(curves, metric=metric, labels=order, lang=args.lang, runs=runs)
            return P.save_figure(fig, out / "fresh_gap", dpi=args.dpi, close=True)

        def sweep():
            if not args.sweep:
                return None
            xkey, _, gkey = args.sweep.partition(":")
            fig = P.plot_sweep(runs, xkey, _run_level_metric(runs, metric), group_key=gkey or "auto", labels=order, lang=args.lang)
            return P.save_figure(fig, out / "sweep", dpi=args.dpi, close=True)

        wanted = None if args.figures.strip().lower() == "all" else {f.strip() for f in args.figures.split(",") if f.strip()}
        for key, name, fn in (("performance", "performance figure", perf), ("mechanism", "mechanism figure", mech),
                              ("summary", "summary figure", summ), ("fresh_gap", "fresh-gap figure", fresh), ("sweep", "sweep figure", sweep)):
            if wanted is None or key in wanted:
                _guard(name, fn, errors, args.quiet)
        plt.close("all")

    if summary is not None:
        print((out / "summary.md").read_text(encoding="utf-8"))
    if comparison is not None and not args.quiet:
        print((out / "comparison.md").read_text(encoding="utf-8"))
    if errors and not args.quiet:
        print(f"[analyze] finished with {len(errors)} skipped product(s): " + ", ".join(n for n, _ in errors), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
