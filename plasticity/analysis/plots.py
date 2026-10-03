"""Matplotlib figures for the thesis (Agg backend, pure functions returning ``Figure``).

Backend: ``matplotlib.use('Agg')`` is forced at import unless the caller chose a backend through the
``MPLBACKEND`` environment variable.  Style: the thesis rc settings (:data:`STYLE`) are applied with
``matplotlib.rc_context`` *inside* every plot / save function only, so importing this module never
changes the global ``rcParams``; call :func:`apply_style` explicitly to make them global.

Colour policy (dataviz skill, ``references/palette.md``): one validated 8-slot categorical palette,
assigned in **fixed order and never cycled**; the validator (``validate_palette.js``) passes it in light
mode (worst adjacent CVD dE 9.1, normal-vision dE 19.6; the aqua / yellow / magenta slots sit below 3:1
contrast, which is relieved because every figure is shipped with its table twin and a legend).  The
three parametrisations of the thesis get *pinned* slots so they look the same in every figure:

    sin -> slot 1 (blue)   tanh -> slot 2 (orange)   standard/baseline -> slot 3 (aqua)

The three pinned colours are *reserved*: other labels take the remaining five slots in order of first
appearance and only spill into a pinned slot once those are exhausted, so a suite without a ``tanh``
label never draws something else in orange.  A base label (``sin``) shares the pinned colour of its own
sweep variants (``sin/lr=0.01`` ...) so the sensitivity figure keeps the same colour code.  A label seen
once keeps its colour for the rest of the process (module-level registry, see :func:`make_palette` /
:func:`reset_palette`).
Past eight labels the palette is *not* extended with generated hues: extra labels share a muted grey and
are told apart by line style (fold / facet instead when possible).

Marks: 2 px lines, CI bands as a ~15 % wash of the series hue, hairline solid grid, no top/right
spines, legend always present for >= 2 series (moved outside the axes, with a wider figure, when there
are many), the "mean ± 95 % CI" note below the x-axis label, text in ink tokens (never in the series
colour).  All user-facing text goes through :data:`LABELS` (Turkish default, ``lang='en'`` available).
"""
from __future__ import annotations

import functools
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

if os.environ.get("MPLBACKEND") is None:  # the caller may choose another backend through MPLBACKEND
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .aggregate import base_label, match_labels, mean_curves, t_half_width  # noqa: E402

__all__ = ["LABELS", "label_text", "PALETTE", "MODE_SLOTS", "STYLE", "make_palette", "reset_palette", "infer_modes",
           "order_labels", "apply_style", "style_context", "save_figure", "plot_performance_curves", "plot_mechanism_panels",
           "plot_summary_dots", "plot_fresh_gap", "plot_sweep", "DEFAULT_MECHANISM_PANELS"]

# ------------------------------------------------------------------------------------------------
# palette (reference instance of the dataviz skill; validated, fixed order)
# ------------------------------------------------------------------------------------------------
PALETTE: Tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
MODE_SLOTS: Dict[str, int] = {"sin": 0, "tanh": 1, "standard": 2}
OVERFLOW_COLOR = "#898781"          # muted ink for labels beyond the 8 slots (with distinct line styles)
OVERFLOW_STYLES = ("--", ":", "-.", (0, (5, 2, 1, 2)))
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BAND_ALPHA = 0.15

_REGISTRY: Dict[str, str] = {}      # label -> hex, stable for the process
_STYLE_REGISTRY: Dict[str, Any] = {}

_MODE_PATTERNS = (
    ("sin", re.compile(r"(^|[^a-z])sin([^a-z]|$)", re.I)),
    ("tanh", re.compile(r"(^|[^a-z])tanh([^a-z]|$)", re.I)),
    ("standard", re.compile(r"(^|[^a-z])(std|standard|baseline|base)([^a-z]|$)", re.I)),
)


def _mode_from_name(label: str) -> Optional[str]:
    for mode, pat in _MODE_PATTERNS:
        if pat.search(str(label)):
            return mode
    return None


def infer_modes(runs: Optional[pd.DataFrame], labels: Iterable[str], label_col: str = "label") -> Dict[str, Optional[str]]:
    """``{label: 'sin'|'tanh'|'standard'|None}`` from the config columns of ``runs`` (fallback: the label text).

    'standard' is only assigned to the plain baseline (``cfg.method.name == 'baseline'`` with the standard
    parametrisation), so that the published methods, which also use the standard parametrisation, are not
    pinned to its colour.  ``label_col`` is the column of ``runs`` the labels refer to (``'label'`` or
    ``'base_label'``).
    """
    out: Dict[str, Optional[str]] = {}
    col_mode = "cfg.model.reparam.hidden"
    col_method = "cfg.method.name"
    for lab in labels:
        mode: Optional[str] = None
        if runs is not None and col_mode in runs.columns and label_col in runs.columns:
            sub = runs.loc[runs[label_col].astype(str) == str(lab)]
            if len(sub):
                m = sub[col_mode].dropna().astype(str).mode()
                m = str(m.iloc[0]) if len(m) else None
                if m in ("sin", "tanh"):
                    mode = m
                elif m == "standard":
                    meth = sub[col_method].dropna().astype(str).mode() if col_method in sub.columns else pd.Series(dtype=str)
                    if (len(meth) == 0 or str(meth.iloc[0]) == "baseline") and _mode_from_name(lab) in (None, "standard"):
                        mode = "standard"
        if mode is None:
            mode = _mode_from_name(lab)
        out[lab] = mode
    return out


def reset_palette() -> None:
    """Forget every label -> colour assignment made so far in this process."""
    _REGISTRY.clear()
    _STYLE_REGISTRY.clear()


_PINNED_COLORS = frozenset(PALETTE[i] for i in MODE_SLOTS.values())


def make_palette(labels: Iterable[str], modes: Optional[Dict[str, Optional[str]]] = None, runs: Optional[pd.DataFrame] = None,
                 colors: Optional[Dict[str, str]] = None, label_col: str = "label") -> Dict[str, str]:
    """Stable ``{label: hex}`` map.  Labels already registered keep their colour; pinned modes get their slot
    (first label of that mode wins; a base label such as ``sin`` shares the slot already held by one of its
    sweep variants ``sin/...``); the rest take the lowest free *unreserved* slots in the order given and use
    the three pinned slots only once the other five are taken.  ``colors`` overrides explicitly (and is
    registered).  ``label_col`` tells :func:`infer_modes` which column of ``runs`` the labels refer to."""
    labels = [str(l) for l in labels]
    if colors:
        _REGISTRY.update({str(k): v for k, v in colors.items()})
    if modes is None:
        modes = infer_modes(runs, labels, label_col=label_col)
    used = set(_REGISTRY.values())
    # 1) pinned modes
    for lab in labels:
        if lab in _REGISTRY:
            continue
        slot = MODE_SLOTS.get(modes.get(lab) or "")
        if slot is None:
            continue
        colour = PALETTE[slot]
        if colour not in used:
            _REGISTRY[lab] = colour
            used.add(colour)
        elif "/" not in lab and any(v == colour and base_label(k) == lab for k, v in _REGISTRY.items()):
            _REGISTRY[lab] = colour  # base label of a sweep: same colour as its (first) variant
    # 2) everything else: next free unreserved slot in fixed order (reserved slots last);
    #    beyond 8 -> muted grey + distinct line style
    for lab in labels:
        if lab in _REGISTRY:
            continue
        free = [c for c in PALETTE if c not in used and c not in _PINNED_COLORS] or [c for c in PALETTE if c not in used]
        if free:
            _REGISTRY[lab] = free[0]
            used.add(free[0])
        else:
            n_over = sum(1 for v in _REGISTRY.values() if v == OVERFLOW_COLOR)
            if n_over == 0:
                warnings.warn("more than 8 labels: extra labels share a muted grey and are distinguished by line style; "
                              "prefer folding or faceting (dataviz rule: never generate a 9th hue)")
            _REGISTRY[lab] = OVERFLOW_COLOR
            _STYLE_REGISTRY[lab] = OVERFLOW_STYLES[n_over % len(OVERFLOW_STYLES)]
    return {lab: _REGISTRY[lab] for lab in labels}


def _linestyle(label: str) -> Any:
    return _STYLE_REGISTRY.get(str(label), "-")


def order_labels(labels: Iterable[str], modes: Optional[Dict[str, Optional[str]]] = None, runs: Optional[pd.DataFrame] = None,
                 label_col: str = "label") -> List[str]:
    """Pinned modes first (sin, tanh, standard), then the rest in sorted order."""
    labels = [str(l) for l in dict.fromkeys(labels)]
    if modes is None:
        modes = infer_modes(runs, labels, label_col=label_col)
    rank = {"sin": 0, "tanh": 1, "standard": 2}
    return sorted(labels, key=lambda l: (rank.get(modes.get(l) or "", 3), l))


# ------------------------------------------------------------------------------------------------
# text
# ------------------------------------------------------------------------------------------------
LABELS: Dict[str, Dict[str, str]] = {
    "tr": {
        "task": "Görev", "online_accuracy": "Çevrimiçi doğruluk", "test_accuracy": "Test doğruluğu",
        "online_loss": "Çevrimiçi kayıp", "fresh_accuracy": "Taze model doğruluğu", "time": "Süre (s)",
        "continual": "sürekli", "fresh": "taze model", "accuracy": "Doğruluk",
        "dead_frac/all": "Ölü birim oranı", "inactive_frac/all": "Etkisiz birim oranı",
        "w_abs_mean/all": "Ortalama |w|", "w_fro_norm/all": "‖W‖_F", "w_abs_max/all": "En büyük |w|",
        "theta_abs_mean/all": "Ortalama |θ|",
        "stable_rank/last": "Kararlı rank (son gizli katman)", "effective_rank/last": "Etkin rank (son gizli katman)",
        "stable_rank_c/last": "Kararlı rank, merkezlenmiş", "effective_rank_c/last": "Etkin rank, merkezlenmiş",
        "jac_sq_mean/all": "Ortalama (∂W/∂Θ)²", "jac_mean/all": "Ortalama ∂W/∂Θ", "w_over_A_mean/all": "Ortalama |W|/A",
        "sat_frac/all": "Doyma oranı", "grad_norm_theta": "‖∇_Θ L‖", "grad_norm_w": "‖∇_W L‖",
        "fisher_trace_theta": "Fisher izi (Θ)", "fisher_trace_w": "Fisher izi (W)",
        "fisher_erank_theta": "Fisher etkin rank (Θ)", "fisher_erank_w": "Fisher etkin rank (W)",
        "jac_fro_ratio": "Jakobiyen Frobenius oranı", "step_grad_norm_theta": "Adım gradyan normu (Θ)",
        "step_dw_norm": "Adım ‖ΔW‖", "step_dtheta_norm": "Adım ‖ΔΘ‖", "act_abs_mean/all": "Ortalama |aktivasyon|",
        "auc_norm": "Normalize AUC", "early_window_mean": "Erken pencere ort.", "final_window_mean": "Son pencere ort.",
        "retention_ratio": "Koruma oranı", "drop": "Düşüş", "slope_per_100": "Eğim / 100 görev",
        "fresh_gap_final": "Taze model farkı (son)", "fresh_gap_mean": "Taze model farkı (ort.)",
        "min_task_perf": "En düşük görev başarımı", "last_task_perf": "Son görev başarımı",
        "final_test_accuracy": "Son test doğruluğu", "wall_time": "Duvar süresi (s)", "global_steps": "Toplam adım",
        "n_tasks": "Görev sayısı", "n_seeds": "Tohum sayısı", "n_pairs": "Eşleşme sayısı",
        "label": "Yöntem", "seed": "Tohum", "mean": "Ortalama", "std": "Std. sapma", "ci95": "%95 GA",
        "reference": "Referans", "mean_label": "Ortalama (yöntem)", "mean_ref": "Ortalama (referans)",
        "mean_diff": "Fark (yöntem − referans)", "p_perm": "p (permütasyon)", "p_holm": "p (Holm)",
        "cohen_dz": "Cohen d_z", "metric": "Ölçüt", "better": "daha iyi", "worse": "daha kötü",
        "final_prefix": "Son: ", "early_prefix": "Erken: ", "lr": "Öğrenme oranı", "gamma": "γ",
        "mean_ci_note": "ortalama ± %95 GA (tohumlar üzerinden, t-dağılımı)",
        "performance_title": "Görev başarımı", "mechanism_title": "Mekanizma ölçütleri", "summary_title": "Özet ölçütler",
        "fresh_title": "Sürekli öğrenme vs. taze model", "sweep_title": "Duyarlılık",
        "reference_line": "referans", "value_axis": "Değer", "seed_points": "tohumlar",
        "single_seed_note": "tek tohum, GA yok", "n_seeds_short": "tohum",
    },
    "en": {
        "task": "Task", "online_accuracy": "Online accuracy", "test_accuracy": "Test accuracy",
        "online_loss": "Online loss", "fresh_accuracy": "Fresh-model accuracy", "time": "Time (s)",
        "continual": "continual", "fresh": "fresh model", "accuracy": "Accuracy",
        "dead_frac/all": "Dead unit fraction", "inactive_frac/all": "Inactive unit fraction",
        "w_abs_mean/all": "Mean |w|", "w_fro_norm/all": "‖W‖_F", "w_abs_max/all": "Max |w|",
        "theta_abs_mean/all": "Mean |θ|",
        "stable_rank/last": "Stable rank (last hidden layer)", "effective_rank/last": "Effective rank (last hidden layer)",
        "stable_rank_c/last": "Stable rank, centred", "effective_rank_c/last": "Effective rank, centred",
        "jac_sq_mean/all": "Mean (∂W/∂Θ)²", "jac_mean/all": "Mean ∂W/∂Θ", "w_over_A_mean/all": "Mean |W|/A",
        "sat_frac/all": "Saturated fraction", "grad_norm_theta": "‖∇_Θ L‖", "grad_norm_w": "‖∇_W L‖",
        "fisher_trace_theta": "Fisher trace (Θ)", "fisher_trace_w": "Fisher trace (W)",
        "fisher_erank_theta": "Fisher effective rank (Θ)", "fisher_erank_w": "Fisher effective rank (W)",
        "jac_fro_ratio": "Jacobian Frobenius ratio", "step_grad_norm_theta": "Step gradient norm (Θ)",
        "step_dw_norm": "Step ‖ΔW‖", "step_dtheta_norm": "Step ‖ΔΘ‖", "act_abs_mean/all": "Mean |activation|",
        "auc_norm": "Normalised AUC", "early_window_mean": "Early-window mean", "final_window_mean": "Final-window mean",
        "retention_ratio": "Retention ratio", "drop": "Drop", "slope_per_100": "Slope / 100 tasks",
        "fresh_gap_final": "Fresh-model gap (final)", "fresh_gap_mean": "Fresh-model gap (mean)",
        "min_task_perf": "Min task performance", "last_task_perf": "Last task performance",
        "final_test_accuracy": "Final test accuracy", "wall_time": "Wall time (s)", "global_steps": "Total steps",
        "n_tasks": "Tasks", "n_seeds": "Seeds", "n_pairs": "Pairs",
        "label": "Method", "seed": "Seed", "mean": "Mean", "std": "Std", "ci95": "95% CI",
        "reference": "Reference", "mean_label": "Mean (method)", "mean_ref": "Mean (reference)",
        "mean_diff": "Diff (method − reference)", "p_perm": "p (permutation)", "p_holm": "p (Holm)",
        "cohen_dz": "Cohen's d_z", "metric": "Metric", "better": "better", "worse": "worse",
        "final_prefix": "Final: ", "early_prefix": "Early: ", "lr": "Learning rate", "gamma": "γ",
        "mean_ci_note": "mean ± 95% CI (over seeds, Student t)",
        "performance_title": "Task performance", "mechanism_title": "Mechanism metrics", "summary_title": "Summary metrics",
        "fresh_title": "Continual vs. fresh model", "sweep_title": "Sensitivity",
        "reference_line": "reference", "value_axis": "Value", "seed_points": "seeds",
        "single_seed_note": "single seed, no CI", "n_seeds_short": "seeds",
    },
}


def label_text(key: str, lang: str = "tr") -> str:
    """Human-readable text for a metric / column / config key (falls back to the key itself)."""
    table = LABELS.get(lang, LABELS["tr"])
    key = str(key)
    if key in table:
        return table[key]
    for prefix in ("final_", "early_"):
        if key.startswith(prefix) and key[len(prefix):] in table:
            return table[prefix + "prefix"] + table[key[len(prefix):]]
    if key.startswith("cfg."):
        last = key.split(".")[-1]
        return table.get(last, last)
    if key.startswith("method/"):
        return key[len("method/"):]
    return key


# ------------------------------------------------------------------------------------------------
# style and saving
# ------------------------------------------------------------------------------------------------
STYLE: Dict[str, Any] = {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
}
"""Quiet thesis style: hairline solid grid, no top/right spines, ink text tokens, sans-serif."""


def apply_style() -> None:
    """Make :data:`STYLE` the *global* matplotlib style (not done at import; the plot functions use
    :func:`style_context` instead so that importing this module leaves ``rcParams`` untouched)."""
    plt.rcParams.update(STYLE)


def style_context():
    """``matplotlib.rc_context`` with :data:`STYLE`; every plot / save function of this module runs inside it."""
    return matplotlib.rc_context(STYLE)


def _styled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with style_context():
            return fn(*args, **kwargs)
    return wrapper


@_styled
def save_figure(fig: Figure, path: str | os.PathLike, formats: Sequence[str] = ("png", "pdf"), dpi: int = 200, close: bool = False) -> List[Path]:
    """Save ``fig`` as ``<path>.<fmt>`` for every format (an extension in ``path`` is replaced), tight bounding
    box (the outside legend and the CI note are included).  Returns the paths."""
    p = Path(path)
    if p.suffix.lower() in (".png", ".pdf", ".svg", ".jpg"):
        p = p.with_suffix("")
    p.parent.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for fmt in formats:
        target = p.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, format=fmt, bbox_inches="tight")
        out.append(target)
    if close:
        plt.close(fig)
    return out


def _T(lang: str):
    return lambda k: label_text(k, lang)


def _fig_ax(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig._plasticity_owned = True  # created here -> we may widen it / lay it out
        return fig, ax
    return ax.figure, ax


def _finish(fig: Figure) -> None:
    """Lay out a figure created by this module so the outside legend and the bottom note fit on the canvas."""
    if getattr(fig, "_plasticity_owned", False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fig.tight_layout()
            except Exception:  # pragma: no cover - layout is cosmetic
                pass


def _present_labels(df: pd.DataFrame, labels: Optional[Sequence[str]], runs: Optional[pd.DataFrame] = None) -> List[str]:
    present = [str(l) for l in pd.unique(df["label"])] if "label" in df.columns else []
    if labels is None:
        return order_labels(present, runs=runs)
    return [str(l) for l in labels if str(l) in present]


LEGEND_OUTSIDE_FROM = 5   # with this many series or more, legends move outside the axes (never over the data)
LEGEND_EXTRA_WIDTH = 1.6  # inches added to a figure we created when its legend goes outside
NOTE_OFFSET_PT = 34.0     # the CI note sits this far below the axes (under the tick labels and the x label)


def _widen_for_outside_legend(ax) -> None:
    """Widen a figure created here so an outside legend does not squeeze the axes (the axes keep their size)."""
    fig = ax.figure
    if not getattr(fig, "_plasticity_owned", False) or getattr(fig, "_plasticity_widened", False) or len(fig.axes) != 1:
        return
    old_w = fig.get_figwidth()
    new_w = old_w + LEGEND_EXTRA_WIDTH
    sp = fig.subplotpars
    fig.set_figwidth(new_w)
    fig.subplots_adjust(left=sp.left * old_w / new_w, right=sp.right * old_w / new_w)
    fig._plasticity_widened = True


def _place_legend(ax, handles, labels, loc="best", outside: Optional[bool] = None, anchor=(1.01, 1.0), outside_loc="upper left", **kw):
    """Legend for >= 2 series (a single series needs none).  Outside the axes (right, figure widened) when there
    are many series."""
    if len(labels) < 2:
        return None
    if outside is None:
        outside = len(labels) >= LEGEND_OUTSIDE_FROM
    kw.setdefault("labelcolor", INK2)
    kw.setdefault("handlelength", 2.2)
    if outside:
        _widen_for_outside_legend(ax)
        return ax.legend(handles, labels, loc=outside_loc, bbox_to_anchor=anchor, borderaxespad=0.0, **kw)
    return ax.legend(handles, labels, loc=loc, **kw)


def _ci_note(T, n_max: int) -> str:
    """'mean ± 95 % CI (...)' when at least two seeds exist, otherwise 'single seed, no CI'."""
    if n_max >= 2:
        return f"{T('mean_ci_note')}, n = {n_max} {T('n_seeds_short')}"
    return T("single_seed_note")


def _note(ax, text: str) -> None:
    """Small muted note (CI description) right-aligned *below* the x-axis label, where it cannot collide with
    the left-aligned title.  Tagged ``gid='ci_note'``."""
    if not text:
        return
    ax.annotate(text, xy=(1.0, 0.0), xycoords="axes fraction", xytext=(0.0, -NOTE_OFFSET_PT), textcoords="offset points",
                ha="right", va="top", fontsize=7, color=MUTED, annotation_clip=False, gid="ci_note")


# ------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------
def _draw_mean_band(ax, mc: pd.DataFrame, label: str, color: str, ci: bool = True, ls="-", marker=None, lw=None, ms=4):
    sub = mc[mc["label"] == label]
    if len(sub) == 0:
        return None
    x, m = sub["task"].to_numpy(), sub["mean"].to_numpy()
    if ci and sub["ci_low"].notna().any():
        ax.fill_between(x, sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy(), color=color, alpha=BAND_ALPHA, linewidth=0)
    (line,) = ax.plot(x, m, color=color, linestyle=ls, marker=marker, markersize=ms, markeredgecolor=SURFACE, markeredgewidth=0.8,
                      linewidth=lw if lw is not None else plt.rcParams["lines.linewidth"], label=label)
    return line


@_styled
def plot_performance_curves(curves: pd.DataFrame, metric: str, labels: Optional[Sequence[str]] = None, smooth: Optional[int] = None,
                            ax=None, title: Optional[str] = None, ylabel: Optional[str] = None, lang: str = "tr",
                            colors: Optional[Dict[str, str]] = None, runs: Optional[pd.DataFrame] = None, ci: bool = True,
                            legend: bool = True, figsize=(6.0, 3.6)) -> Figure:
    """Mean curve of ``metric`` per label over seeds with a t-based 95 % CI band."""
    T = _T(lang)
    fig, ax = _fig_ax(ax, figsize)
    labels = _present_labels(curves, labels, runs)
    pal = make_palette(labels, runs=runs, colors=colors)
    mc = mean_curves(curves, metric, smooth=smooth, labels=labels)
    handles: List[Any] = []
    for lab in labels:
        h = _draw_mean_band(ax, mc, lab, pal[lab], ci=ci, ls=_linestyle(lab))
        if h is not None:
            handles.append(h)
    ax.set_xlabel(T("task"))
    ax.set_ylabel(ylabel if ylabel is not None else T(metric))
    if title is None:
        title = T("performance_title")
    if title:
        ax.set_title(title, loc="left")
    if len(mc) == 0:
        ax.text(0.5, 0.5, f"{metric}: no data", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    else:
        _note(ax, _ci_note(T, int(mc["n"].max())) if ci else "")
    if legend:
        _place_legend(ax, handles, [h.get_label() for h in handles])
    _finish(fig)
    return fig


DEFAULT_MECHANISM_PANELS: Tuple[str, ...] = ("dead_frac/all", "w_abs_mean/all", "effective_rank/last", "stable_rank/last",
                                              "grad_norm_w", "fisher_trace_w", "w_over_A_mean/all", "sat_frac/all", "jac_sq_mean/all")


@_styled
def plot_mechanism_panels(curves: pd.DataFrame, keys: Optional[Sequence[str]] = None, labels: Optional[Sequence[str]] = None,
                          smooth: Optional[int] = None, ncols: int = 3, lang: str = "tr", colors: Optional[Dict[str, str]] = None,
                          runs: Optional[pd.DataFrame] = None, ci: bool = True, skip_missing: bool = True, title: Optional[str] = None,
                          panel_size=(3.0, 2.3), sharex: bool = True) -> Figure:
    """Grid of small panels, one per mechanism key (keys absent from ``curves`` are skipped), shared legend."""
    T = _T(lang)
    labels = _present_labels(curves, labels, runs)
    pal = make_palette(labels, runs=runs, colors=colors)
    keys = list(keys) if keys is not None else [k for k in DEFAULT_MECHANISM_PANELS if k in curves.columns]
    if skip_missing:
        keys = [k for k in keys if k in curves.columns and pd.to_numeric(curves[k], errors="coerce").notna().any()]
    n = max(len(keys), 1)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows + 0.5), sharex=sharex, squeeze=False)
    handles: Dict[str, Any] = {}
    for i, ax in enumerate(axes.flat):
        if i >= len(keys):
            ax.set_visible(False)
            continue
        key = keys[i]
        mc = mean_curves(curves, key, smooth=smooth, labels=labels)
        for lab in labels:
            h = _draw_mean_band(ax, mc, lab, pal[lab], ci=ci, ls=_linestyle(lab), lw=1.4)
            if h is not None:
                handles.setdefault(lab, h)
        ax.set_title(T(key), loc="left", fontsize=9)
        if i // ncols == nrows - 1 or not sharex:
            ax.set_xlabel(T("task"))
    if not keys:
        axes[0, 0].set_visible(True)
        axes[0, 0].text(0.5, 0.5, "no mechanism metrics", transform=axes[0, 0].transAxes, ha="center", va="center", color=MUTED)
    if len(handles) >= 2:
        fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=min(len(handles), 4),
                   bbox_to_anchor=(0.5, 0.0), labelcolor=INK2, handlelength=2.2)
    fig.suptitle(title if title is not None else T("mechanism_title"), x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06 if len(handles) >= 2 else 0, 1, 0.97))
    return fig


@_styled
def plot_summary_dots(runs: pd.DataFrame, metric: str, labels: Optional[Sequence[str]] = None, reference: Optional[str] = None,
                      lang: str = "tr", colors: Optional[Dict[str, str]] = None, ax=None, title: Optional[str] = None,
                      alpha: float = 0.05, show_seeds: bool = True, figsize=None) -> Figure:
    """Horizontal dot (mean) + 95 % t-CI per label for a run-level ``metric``; faint seed points; reference hairline."""
    T = _T(lang)
    labels = _present_labels(runs, labels, runs)
    pal = make_palette(labels, runs=runs, colors=colors)
    if figsize is None:
        figsize = (5.0, 0.45 * max(len(labels), 1) + 1.2)
    fig, ax = _fig_ax(ax, figsize)
    if metric not in runs.columns:
        ax.text(0.5, 0.5, f"{metric}: no data", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        ax.set_title(title if title is not None else T(metric), loc="left")
        return fig
    vals = pd.to_numeric(runs[metric], errors="coerce")
    ys = np.arange(len(labels))[::-1]
    n_max = 0
    for y, lab in zip(ys, labels):
        x = vals[runs["label"] == lab].dropna().to_numpy(dtype=float)
        if x.size == 0:
            continue
        m = float(x.mean())
        n_max = max(n_max, int(x.size))
        half = float(t_half_width(np.array([x.size]), np.array([x.std(ddof=1) if x.size > 1 else np.nan]), alpha)[0])
        if show_seeds:
            ax.scatter(x, np.full(x.size, y), s=14, color=pal[lab], alpha=0.35, linewidths=0, zorder=2)
        if np.isfinite(half):
            ax.plot([m - half, m + half], [y, y], color=pal[lab], linewidth=1.8, solid_capstyle="round", zorder=3)
        ax.scatter([m], [y], s=42, color=pal[lab], edgecolor=SURFACE, linewidths=1.0, zorder=4)
    if reference is not None and reference in labels:
        ref = vals[runs["label"] == reference].dropna()
        if len(ref):
            ax.axvline(float(ref.mean()), color=AXIS, linewidth=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlabel(T(metric))
    ax.grid(axis="y", visible=False)
    ax.set_title(title if title is not None else T(metric), loc="left")
    _note(ax, _ci_note(T, n_max))
    _finish(fig)
    return fig


@_styled
def plot_fresh_gap(curves: pd.DataFrame, metric: Optional[str] = None, labels: Optional[Sequence[str]] = None, lang: str = "tr",
                   colors: Optional[Dict[str, str]] = None, runs: Optional[pd.DataFrame] = None, ax=None, title: Optional[str] = None,
                   fresh_key: str = "fresh_accuracy", ci: bool = True, figsize=(6.0, 3.6)) -> Figure:
    """Continual accuracy (solid, filled markers) vs. fresh-model accuracy (dashed, open markers) at the reference tasks.

    Two legends inside the axes (labels lower left, line styles lower right) for a few labels; a single merged
    legend outside the (widened) figure for >= ``LEGEND_OUTSIDE_FROM`` labels.  Raises ``ValueError`` when no
    fresh-reference data exist (check :func:`aggregate.has_fresh_data` first).
    """
    T = _T(lang)
    if fresh_key not in curves.columns or not pd.to_numeric(curves[fresh_key], errors="coerce").notna().any():
        raise ValueError(f"no {fresh_key!r} data in curves")
    if metric is None:
        metric = "test_accuracy" if "test_accuracy" in curves.columns and curves["test_accuracy"].notna().any() else "online_accuracy"
    pts = curves[pd.to_numeric(curves[fresh_key], errors="coerce").notna()]
    labels = _present_labels(pts, labels, runs)
    pal = make_palette(labels, runs=runs, colors=colors)
    fig, ax = _fig_ax(ax, figsize)
    mc_c = mean_curves(pts, metric, labels=labels)
    mc_f = mean_curves(pts, fresh_key, labels=labels)
    for lab in labels:
        _draw_mean_band(ax, mc_c, lab, pal[lab], ci=ci, ls="-", marker="o", ms=5)
        sub = mc_f[mc_f["label"] == lab]
        if len(sub):
            if ci and sub["ci_low"].notna().any():
                ax.fill_between(sub["task"], sub["ci_low"], sub["ci_high"], color=pal[lab], alpha=BAND_ALPHA * 0.6, linewidth=0)
            ax.plot(sub["task"], sub["mean"], color=pal[lab], linestyle="--", linewidth=1.4, marker="o", markersize=5,
                    markerfacecolor=SURFACE, markeredgecolor=pal[lab], markeredgewidth=1.2)
    ax.set_xlabel(T("task"))
    ax.set_ylabel(T(metric))
    ax.set_title(title if title is not None else T("fresh_title"), loc="left")
    handles = [Line2D([], [], color=pal[l], linewidth=1.8, marker="o", markersize=5, markeredgecolor=SURFACE, label=l) for l in labels]
    style = [Line2D([], [], color=INK2, linewidth=1.8, marker="o", markersize=5, label=T("continual")),
             Line2D([], [], color=INK2, linewidth=1.4, linestyle="--", marker="o", markersize=5, markerfacecolor=SURFACE, label=T("fresh"))]
    outside = len(labels) >= LEGEND_OUTSIDE_FROM
    if outside:  # many labels: one legend outside (labels, blank row, line styles) so the two blocks never collide
        spacer = Line2D([], [], alpha=0.0, label=" ")
        _place_legend(ax, handles + [spacer] + style, [h.get_label() for h in handles] + [" "] + [h.get_label() for h in style], outside=True)
    else:
        leg1 = _place_legend(ax, handles, [h.get_label() for h in handles], loc="lower left", outside=False)
        if leg1 is not None:
            ax.add_artist(leg1)
            leg1.set_clip_on(False)  # add_artist clips to the axes patch, which would drop the legend from the tight bbox / layout
        ax.legend(handles=style, loc="lower right", labelcolor=INK2)
    n_max = int(max(mc_c["n"].max() if len(mc_c) else 0, mc_f["n"].max() if len(mc_f) else 0))
    _note(ax, _ci_note(T, n_max) if ci else "")
    _finish(fig)
    return fig


def _resolve_sweep_group(df: pd.DataFrame, group_key: Optional[str]) -> Optional[str]:
    """Column to group the sensitivity curves by.

    ``'auto'`` (default) -> ``base_label`` when present else ``label``.  An explicit ``'label'`` falls back to
    ``base_label`` when every label holds a single x value (sweep suites label runs ``<base>/lr=<v>``, so the
    full label would give one isolated dot per series) and ``base_label`` has several.  None / an absent
    column -> a single group.
    """
    if group_key is None:
        return None
    if group_key == "auto":
        group_key = "base_label" if "base_label" in df.columns else "label"
    if group_key not in df.columns:
        return None
    if group_key == "label" and "base_label" in df.columns and len(df):
        per_label = df.groupby("label")["_x"].nunique()
        per_base = df.groupby("base_label")["_x"].nunique()
        if per_label.max() <= 1 and per_base.max() > 1:
            return "base_label"
    return group_key


@_styled
def plot_sweep(runs: pd.DataFrame, x_cfg_key: str, metric: str, group_key: Optional[str] = "auto", labels: Optional[Sequence[str]] = None,
               lang: str = "tr", colors: Optional[Dict[str, str]] = None, ax=None, title: Optional[str] = None, logx: Optional[bool] = None,
               alpha: float = 0.05, xlabel: Optional[str] = None, figsize=(5.5, 3.6)) -> Figure:
    """Sensitivity curve: mean ± 95 % t-CI of ``metric`` over seeds against a config value (e.g. ``cfg.optimizer.lr``),
    one line per ``group_key`` value.

    ``group_key='auto'`` (default) groups by ``base_label`` (the label without its sweep segments, so the runs
    ``sin/lr=0.001, sin/lr=0.003, ...`` form one ``sin`` line); ``'label'`` groups by the full label but falls
    back to ``base_label`` when that would leave every series with a single point; any other column of
    ``runs`` (e.g. ``cfg.method.name``) works too; None pools everything into one line.  ``labels`` restricts
    the runs to those labels (a base label selects its sweep variants) and, for label-like grouping, orders
    the lines.  Log x-axis when the x values span > 1 decade."""
    T = _T(lang)
    fig, ax = _fig_ax(ax, figsize)
    if x_cfg_key not in runs.columns or metric not in runs.columns:
        ax.text(0.5, 0.5, f"{x_cfg_key} / {metric}: no data", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        return fig
    df = runs.copy()
    df["_x"] = pd.to_numeric(df[x_cfg_key], errors="coerce")
    df["_y"] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["_x", "_y"])
    gcol = _resolve_sweep_group(df, group_key)
    if gcol is None:
        df["_g"] = "all"
    else:
        df["_g"] = df[gcol].astype(str)
    label_like = gcol in ("label", "base_label")
    if labels is not None and "label" in df.columns:  # restrict the runs to the requested labels (base labels select their variants)
        keep, _ = match_labels([str(l) for l in labels], df["label"].astype(str))
        df = df[df["label"].astype(str).isin(keep)]
    groups = [str(g) for g in pd.unique(df["_g"])]
    if labels is not None and label_like:
        wanted = [str(l) for l in labels]
        if gcol == "base_label":
            wanted = list(dict.fromkeys(base_label(l) for l in wanted))
        groups = [g for g in wanted if g in groups] + [g for g in groups if g not in wanted]
    elif label_like:
        groups = order_labels(groups, runs=runs, label_col=gcol)
    pal = make_palette(groups, runs=runs if label_like else None, colors=colors, label_col=gcol if label_like else "label")
    agg = df.groupby(["_g", "_x"])["_y"].agg(n="count", mean="mean", std=lambda s: s.std(ddof=1)).reset_index()
    agg["half"] = t_half_width(agg["n"].to_numpy(), agg["std"].to_numpy(), alpha)
    handles = []
    for g in groups:
        sub = agg[agg["_g"] == g].sort_values("_x")
        if len(sub) == 0:
            continue
        if sub["half"].notna().any():
            ax.fill_between(sub["_x"], sub["mean"] - sub["half"], sub["mean"] + sub["half"], color=pal[g], alpha=BAND_ALPHA, linewidth=0)
        (h,) = ax.plot(sub["_x"], sub["mean"], color=pal[g], marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                       linestyle=_linestyle(g), label=g)
        handles.append(h)
    xs = agg["_x"].to_numpy()
    if logx is None:
        logx = bool(xs.size and xs.min() > 0 and xs.max() / xs.min() > 10)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel if xlabel is not None else T(x_cfg_key))
    ax.set_ylabel(T(metric))
    ax.set_title(title if title is not None else f"{T('sweep_title')}: {T(x_cfg_key)}", loc="left")
    _place_legend(ax, handles, [h.get_label() for h in handles])
    _note(ax, _ci_note(T, int(agg["n"].max()) if len(agg) else 0))
    _finish(fig)
    return fig
