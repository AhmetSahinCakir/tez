"""Analysis tooling: turn a suite of run directories into tables and figures.

* :mod:`plasticity.analysis.aggregate` -- ``load_suite`` (runs + curves DataFrames), ``mean_curves``
* :mod:`plasticity.analysis.plots`     -- matplotlib figures with a stable, colour-blind-safe palette
* :mod:`plasticity.analysis.tables`    -- markdown / csv summary and paired-comparison tables

The CLI entry point is ``scripts/analyze.py``.
"""
from .aggregate import load_suite, mean_curves, discover_runs, load_run, auto_metric, has_fresh_data, save_tables  # noqa: F401
from .tables import summary_table, comparison_table, write_summary_tables, write_comparison_tables, df_to_markdown  # noqa: F401
from .plots import (  # noqa: F401
    LABELS, label_text, make_palette, reset_palette, save_figure,
    plot_performance_curves, plot_mechanism_panels, plot_summary_dots, plot_fresh_gap, plot_sweep,
)

__all__ = [
    "load_suite", "mean_curves", "discover_runs", "load_run", "auto_metric", "has_fresh_data", "save_tables",
    "summary_table", "comparison_table", "write_summary_tables", "write_comparison_tables", "df_to_markdown",
    "LABELS", "label_text", "make_palette", "reset_palette", "save_figure",
    "plot_performance_curves", "plot_mechanism_panels", "plot_summary_dots", "plot_fresh_gap", "plot_sweep",
]
