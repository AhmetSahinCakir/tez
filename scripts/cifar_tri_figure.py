#!/usr/bin/env python3
"""Figure for RAPOR §7.4: triangle-wave reparametrisation on the CIFAR-100 binary online stream.
Panels: (a) online accuracy (block means) of the main methods plus tri γ=1.0 / 1.5 and the sin γ=1.2 ablation;
(b) fraction of tri weights that crossed the bound at least once (cross_frac), per layer;
(c) output layer of the two tight-bound models: sin γ=1.2 saturation vs tri γ=1.0 crossing.

    python3 scripts/cifar_tri_figure.py   -> reports/cifar_tri_vs_main.png
"""
import glob, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plasticity.utils import read_jsonl  # noqa: E402

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
COL = {"baseline": "#2a78d6", "sin": "#eb6834", "tanh": "#1baf7a", "tri": "#eda100", "wc": "#e87ba4"}
SRC = {
    "Standart": ("results/cifar_main/baseline", COL["baseline"], "-"),
    "Weight Clipping (κ=1)": ("results/cifar_main/weight_clipping", COL["wc"], "-"),
    "Sinüs (γ=5)": ("results/cifar_main/sin", COL["sin"], "-"),
    "Sinüs (γ=1.2)": ("results/cifar_ablation/sin/gamma/gamma=1.2", COL["sin"], "--"),
    "Üçgen (γ=1.0)": ("results/cifar_tri/tri", COL["tri"], "-"),
    "Üçgen (γ=1.5)": ("results/cifar_tri/tri_gamma1.5", COL["tri"], "--"),
}


def runs(d):
    return [read_jsonl(f) for f in sorted(glob.glob(f"{d}/seed*/tasks.jsonl"))]


def mean_key(rs, key):
    """Mean over seeds per task; mechanism keys are logged only at some tasks, so rows without the key are dropped."""
    n = min(len(r) for r in rs)
    m = np.array([[r[t].get(key, np.nan) for t in range(n)] for r in rs], dtype=float)
    return np.nanmean(m, axis=0)


def series(rs, key):
    v = mean_key(rs, key); ok = ~np.isnan(v)
    return np.arange(len(v))[ok], v[ok]


def blocks(v, b=50):
    n = len(v) // b * b
    return (np.arange(n // b) * b + b / 2), v[:n].reshape(-1, b).mean(axis=1)


fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax in axs:
    ax.grid(True, color="#e6e5e1", lw=0.8); ax.set_axisbelow(True); ax.set_xlabel("Görev")

ax = axs[0]
for name, (d, c, ls) in SRC.items():
    rs = runs(d)
    if not rs:
        continue
    t, v = blocks(mean_key(rs, "online_accuracy"))
    ax.plot(t, v, color=c, ls=ls, lw=2.0 if "Üçgen" in name else 1.6, label=f"{name} (n={len(rs)})")
ax.set_ylabel("Çevrimiçi doğruluk (50 görevlik blok ort.)")
ax.set_title("(a) CIFAR-100 ikili akış, 1000 görev", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

ax = axs[1]
tri = runs("results/cifar_tri/tri")
for li, (lab, alpha) in enumerate([("giriş (L0)", 1.0), ("gizli (L1)", 0.65), ("çıkış (L2)", 0.4)]):
    ax.plot(*series(tri, f"cross_frac/L{li}"), color=COL["tri"], alpha=alpha, lw=2.0, label=f"üçgen γ=1.0, {lab}")
tri15 = runs("results/cifar_tri/tri_gamma1.5")
ax.plot(*series(tri15, "cross_frac/all"), color=COL["tri"], ls="--", lw=1.6, label="üçgen γ=1.5, tüm ağ")
ax.set_ylabel("Sınırı en az bir kez aşan ağırlık oranı")
ax.set_title("(b) Yansıma sıklığı (cross_frac)", loc="left", fontweight="bold")
ax.set_ylim(0, 0.7); ax.legend(frameon=False, fontsize=8.5, loc="upper left")

ax = axs[2]
sin12 = runs("results/cifar_ablation/sin/gamma/gamma=1.2")
ax.plot(*series(sin12, "sat_frac/L2"), color=COL["sin"], ls="--", lw=2.0, label="sinüs γ=1.2: doymuş oran (|J| < 0.1)")
ax.plot(*series(tri, "cross_frac/L2"), color=COL["tri"], lw=2.0, label="üçgen γ=1.0: sınırı aşan oran (|J| = 1)")
ax.set_ylabel("Çıkış katmanı ağırlık oranı")
ax.set_title("(c) Çıkış katmanı: sıkı sınır altında", loc="left", fontweight="bold")
ax.set_ylim(0, 1.0); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.text(0.01, 0.01, "Not: sinüs γ=1.2 bileşen deneyi 3 tohum, diğerleri 5 tohum; tohum ortalamaları.", fontsize=8.5, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = ROOT / "reports" / "cifar_tri_vs_main.png"
fig.savefig(out, dpi=160); print("wrote", out)
