# Figures of the paper (English labels, vector PDF).  Usage: python3 scripts/paper_figures.py
# Needs reports/pmnist_fullstream/curves.csv (written by scripts/analyze.py).  Writes paper/figures/*.pdf.
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
OUT = "paper/figures"
COL = {"baseline": "#2a78d6", "sin": "#eb6834", "tanh": "#1baf7a", "tri": "#eda100", "weight_clipping": "#d6498a",
       "ln_wd": "#7b61c9", "sin_refl": "#9c3d12"}
NAME = {"baseline": "Standard", "sin": "Sine", "tanh": "Tanh", "tri": "Triangle wave (ours)", "weight_clipping": "Weight clipping",
        "ln_wd": "LayerNorm + weight decay", "sin_refl": "Reflected sine (3 seeds)"}
LS = {"weight_clipping": (0, (4, 2)), "sin_refl": (0, (1, 1))}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False})
d = pd.read_csv("reports/pmnist_fullstream/curves.csv")

def band(lab, key, win=1):
    """Mean over seeds and the 95% t-interval, per task (rolling mean of width `win` applied per seed first)."""
    p = d[d.label == lab].dropna(subset=[key]).pivot(index="task", columns="seed", values=key).sort_index()
    if win > 1: p = p.rolling(win, min_periods=1, center=True).mean()
    m, n = p.mean(axis=1), p.shape[1]
    h = stats.t.ppf(0.975, n - 1) * p.std(axis=1, ddof=1) / np.sqrt(n)
    return p.index.values, m.values, h.values

def grid(ax): ax.grid(True, color="#e6e5e1", linewidth=0.6); ax.set_axisbelow(True)

# ---- Figure 1: the three bounded maps and the magnitude of their derivative
x = np.linspace(-3.2, 3.2, 1301)
maps = [("sin", "Sine", np.sin(x), np.abs(np.cos(x))), ("tanh", "Tanh", np.tanh(x), 1 - np.tanh(x) ** 2),
        ("tri", "Triangle wave", 2 / np.pi * np.arcsin(np.sin(np.pi * x / 2)), np.ones_like(x))]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3))
for lab, name, f, df in maps:
    axs[0].plot(x, f, color=COL[lab], lw=1.6, label=name); axs[1].plot(x, df, color=COL[lab], lw=1.6, label=name)
for ax, yl in zip(axs, [r"$W/A = f(\Theta/A)$", r"$|\partial W/\partial\Theta| = |f'(\Theta/A)|$"]):
    grid(ax); ax.set_xlabel(r"$\Theta/A$"); ax.set_ylabel(yl)
axs[0].axhline(1, color="#b8b7b2", ls="--", lw=0.8); axs[0].axhline(-1, color="#b8b7b2", ls="--", lw=0.8)
axs[1].set_ylim(-0.05, 1.15); axs[0].legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_maps.pdf"); plt.close(fig)

# ---- Figure 2: online accuracy over the 800-task stream
fig, ax = plt.subplots(figsize=(6.6, 3.2)); grid(ax)
for lab in ["tri", "weight_clipping", "ln_wd", "sin_refl", "baseline", "tanh", "sin"]:
    t, m, h = band(lab, "online_accuracy", 10)
    ax.fill_between(t, m - h, m + h, color=COL[lab], alpha=0.18, lw=0)
    ax.plot(t, m, color=COL[lab], lw=1.5, ls=LS.get(lab, "-"), label=NAME[lab])
ax.set_xlim(0, 800); ax.set_ylim(0.68, 0.96); ax.set_xlabel("Task"); ax.set_ylabel("Online accuracy on the current task")
ax.legend(loc="lower left", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_curves.pdf"); plt.close(fig)

# ---- Figure 3: mechanism
panels = [("sat_frac/all", r"Saturated fraction ($|f'| < 0.1$)", ["sin", "tanh", "tri", "sin_refl"]),
          ("jac_sq_mean/all", r"Step multiplier $\langle f'^2\rangle$", ["sin", "tanh", "tri", "sin_refl"]),
          ("w_over_A_mean/all", r"Distance to the bound $\langle|W|\rangle/A$", ["sin", "tanh", "tri", "sin_refl"]),
          ("effective_rank/last", "Effective rank (last hidden layer)", ["baseline", "sin", "tanh", "tri", "weight_clipping", "sin_refl"])]
fig, axs = plt.subplots(2, 2, figsize=(6.6, 4.3)); axs = axs.ravel()
for ax, (key, title, labs) in zip(axs, panels):
    grid(ax)
    for lab in labs:
        t, m, h = band(lab, key)
        ax.fill_between(t, m - h, m + h, color=COL[lab], alpha=0.18, lw=0)
        ax.plot(t, m, color=COL[lab], lw=1.4, ls=LS.get(lab, "-"), label=NAME[lab].replace(" (ours)", "").replace(" (3 seeds)", ""))
    ax.set_title(title, fontsize=9); ax.set_xlabel("Task"); ax.set_xlim(0, 800); ax.tick_params(labelsize=8)
h, l = axs[3].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{OUT}/fig_mechanism.pdf"); plt.close(fig)
print("wrote", OUT)
