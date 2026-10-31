# Saturation ("frozen parameter") fraction per layer over tasks, aligned with online accuracy, for the 800-task
# stream (advisor review, item 5). Writes reports/ozet/6_donma_orani.png.  Usage: python3 scripts/ozet_donma.py
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "results/pmnist_fullstream"; OUT = "reports/ozet/6_donma_orani.png"
MODELS = [("sin", "Sinüs sınırı", "#eb6834"), ("tanh", "tanh sınırı", "#1baf7a"), ("tri", "Üçgen dalga (yansıma)", "#eda100")]
LAYER_STYLE = {"L0": ("-", 2.2, "giriş katmanı (784→100)"), "L1": ((0, (5, 2)), 1.8, "2. katman"), "L2": ((0, (2, 2)), 1.8, "3. katman"), "L3": ((0, (1, 1)), 1.8, "çıkış katmanı")}
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

def mean_over_seeds(model, key, smooth=1):
    series = []
    for p in sorted(glob.glob(f"{ROOT}/{model}/seed*/tasks.jsonl")):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        pts = [(r["task"], r[key]) for r in rows if key in r and r[key] is not None]
        series.append(dict(pts))
    tasks = sorted(set.intersection(*[set(s) for s in series]))
    vals = np.array([[s[t] for t in tasks] for s in series]).mean(0)
    if smooth > 1:  # centred moving average with shrinking window at the edges (no zero padding)
        vals = np.array([vals[max(0, i - smooth // 2):i + smooth // 2 + 1].mean() for i in range(len(vals))])
    return np.array(tasks), vals

fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
for j, (m, name, col) in enumerate(MODELS):
    ax = axs[0, j]; ax.grid(True, color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    for L, (ls, lw, lab) in LAYER_STYLE.items():
        t, v = mean_over_seeds(m, f"sat_frac/{L}")
        ax.plot(t, v, color=col, ls=ls, lw=lw, label=lab)
    ax.set_ylim(-0.02, 1.0); ax.set_title(name, loc="left", fontweight="bold")
    if j == 0:
        ax.set_ylabel("Donmuş parametre payı (|∂W/∂Θ|/A < 0.1)")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax2 = axs[1, j]; ax2.grid(True, color="#e6e5e1", lw=0.8); ax2.set_axisbelow(True)
    t, v = mean_over_seeds(m, "online_accuracy", smooth=10)
    tb, vb = mean_over_seeds("baseline", "online_accuracy", smooth=10)
    ax2.plot(tb, vb, color="#2a78d6", lw=1.6, label="Standart ağ")
    ax2.plot(t, v, color=col, lw=2.2, label=name)
    ax2.set_ylim(0.68, 0.96); ax2.set_xlabel("Görev"); ax2.set_xlim(0, 800)
    if j == 0:
        ax2.set_ylabel("Görevdeki doğruluk")
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("Donma oranı (üst, katman başına) ve doğruluk (alt): 800 görev, 3 tohum ortalaması", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.01, "Sinüste giriş katmanı 400 görevde %80'in üzerinde donar ve doğruluk eşzamanlı düşer; tanh daha yavaş; üçgen dalgada tanım gereği 0.", fontsize=9, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(OUT, dpi=150); print("wrote", OUT)
