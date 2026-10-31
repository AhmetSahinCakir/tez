#!/usr/bin/env python3
"""Figure for RAPOR §13.4: γ (bound width) sensitivity of the triangle-wave model on PMNIST, 200 tasks × 5000.
(a) online accuracy per γ with the standard, Weight Clipping and sine references; (b) normalised AUC and final
window vs γ; (c) fraction of weights that crossed the bound at least once (cross_frac) per γ.

    python3 scripts/tri_gamma_figure.py   -> reports/pmnist_tri_gamma_vs_main.png
"""
import glob, json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plasticity.utils import read_jsonl  # noqa: E402

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
COL = {"baseline": "#2a78d6", "sin": "#eb6834", "tri": "#eda100", "wc": "#e87ba4"}
GAMMAS = ["1.0", "1.25", "1.5", "2.0", "3.0"]
GDIR = {"1.0": "results/pmnist_main/tri", **{g: f"results/pmnist_tri_gamma/tri/gamma/gamma={g}" for g in GAMMAS[1:]}}
SEEDS = range(5)


def shade(i, n=len(GAMMAS)):
    """γ=1.0 full tri colour, fading towards grey for wider bounds."""
    a = np.array(to_rgb(COL["tri"])); g = np.array(to_rgb("#9a9a94")); f = i / (n - 1)
    return tuple(a * (1 - f) + g * f)


def runs(d):
    return [read_jsonl(f"{d}/seed{s}/tasks.jsonl") for s in SEEDS if Path(f"{d}/seed{s}/tasks.jsonl").exists()]


def summaries(d):
    return [json.load(open(f"{d}/seed{s}/summary.json")) for s in SEEDS if Path(f"{d}/seed{s}/summary.json").exists()]


def series(rs, key):
    n = min(len(r) for r in rs)
    m = np.array([[r[t].get(key, np.nan) for t in range(n)] for r in rs], dtype=float)
    v = np.nanmean(m, axis=0); ok = ~np.isnan(v)
    return np.arange(n)[ok], v[ok]


def blocks(t, v, b=10):
    n = len(v) // b * b
    return t[:n].reshape(-1, b).mean(axis=1), v[:n].reshape(-1, b).mean(axis=1)


fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax in axs:
    ax.grid(True, color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)

ax = axs[0]
for name, d, c, ls in [("Standart", "results/pmnist_main/baseline", COL["baseline"], "-"),
                       ("Weight Clipping", "results/pmnist_main/weight_clipping", COL["wc"], "-"),
                       ("Sinüs (γ=1.5)", "results/pmnist_main/sin", COL["sin"], "-")]:
    ax.plot(*blocks(*series(runs(d), "online_accuracy")), color=c, ls=ls, lw=1.5, label=name)
for i, g in enumerate(GAMMAS):
    ax.plot(*blocks(*series(runs(GDIR[g]), "online_accuracy")), color=shade(i), lw=2.0, label=f"Üçgen γ={g}")
ax.set_xlabel("Görev"); ax.set_ylabel("Çevrimiçi doğruluk (10 görevlik blok ort.)")
ax.set_title("(a) PMNIST, 200 görev × 5.000, 5 tohum", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.2, loc="lower left", ncol=2)

ax = axs[1]
gx = [float(g) for g in GAMMAS]
for key, lab, mk in [("auc_norm", "Normalize AUC", "o"), ("final_window_mean", "Son pencere (son 20 görev)", "s")]:
    m = [np.mean([s[key] for s in summaries(GDIR[g])]) for g in GAMMAS]
    sd = [np.std([s[key] for s in summaries(GDIR[g])]) for g in GAMMAS]
    ax.errorbar(gx, m, yerr=sd, color=COL["tri"], marker=mk, lw=1.8, capsize=3, label=f"Üçgen: {lab}",
                ls="-" if key == "auc_norm" else "--")
for name, d, c in [("Standart", "results/pmnist_main/baseline", COL["baseline"]),
                   ("Weight Clipping", "results/pmnist_main/weight_clipping", COL["wc"])]:
    ax.axhline(np.mean([s["auc_norm"] for s in summaries(d)]), color=c, lw=1.4, ls=":", label=f"{name} AUC")
ax.set_xlabel("γ (sınır genişliği, A = γ·b)"); ax.set_ylabel("Doğruluk")
ax.set_xticks(gx); ax.set_title("(b) Sınır genişliğine duyarlılık", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.2, loc="lower left")

ax = axs[2]
for i, g in enumerate(GAMMAS[1:], start=1):
    ax.plot(*series(runs(GDIR[g]), "cross_frac/all"), color=shade(i), lw=2.0, label=f"γ={g}")
ax.set_xlabel("Görev"); ax.set_ylabel("Sınırı en az bir kez aşan ağırlık oranı")
ax.set_title("(c) Yansıma sıklığı (cross_frac)", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.01, "Not: γ=1.0 koşuları ana paketten (cross_frac o koşularda kaydedilmemiştir); tohum ortalamaları.",
         fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = ROOT / "reports" / "pmnist_tri_gamma_vs_main.png"
fig.savefig(out, dpi=160); print("wrote", out)
