# All seven models of the 800-task stream in one figure (advisor briefing, November 2026).
# Usage: python3 scripts/ozet_tum_modeller.py   (needs reports/pmnist_fullstream/curves.csv); writes reports/ozet/7_tum_modeller_800_gorev.png
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "reports/ozet/7_tum_modeller_800_gorev.png"
MODELS = [("tri", "Üçgen dalga (yansıma)", "#eda100", "-"), ("weight_clipping", "Weight Clipping (kırpma)", "#e87ba4", (0, (4, 2))),
          ("ln_wd", "Layer Norm + Weight Decay", "#7b61c9", "-"), ("sin_refl", "Yansımalı sinüs (3 tohum)", "#9c3d12", (0, (1, 1))),
          ("baseline", "Standart ağ", "#2a78d6", "-"), ("tanh", "tanh sınırı", "#1baf7a", "-"), ("sin", "Sinüs sınırı", "#eb6834", "-")]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
d = pd.read_csv("reports/pmnist_fullstream/curves.csv", usecols=["label", "task", "online_accuracy"])
fig, ax = plt.subplots(figsize=(10, 5.4)); ax.grid(True, color="#e6e5e1", linewidth=0.8); ax.set_axisbelow(True)
ends = []
for lab, name, col, ls in MODELS:
    s = d[d.label == lab].groupby("task")["online_accuracy"].mean().sort_index().rolling(10, min_periods=1, center=True).mean()
    ax.plot(s.index, s.values, color=col, lw=2, ls=ls); ends.append((s.iloc[-1], name))
ax.set_xlim(0, 800 * 1.33); ax.set_ylim(0.68, 0.96); ax.set_xticks(range(0, 801, 100))
ys = []; mind = (0.96 - 0.68) * 0.05  # spread the end labels so that they do not overlap
for v, name in sorted(ends):
    y = v if not ys or v - ys[-1] >= mind else ys[-1] + mind; ys.append(y)
    ax.annotate(name, (799, v), xytext=(812, y), fontsize=10, color="#0b0b0b", va="center",
                arrowprops=dict(arrowstyle="-", color="#b8b7b2", lw=0.8) if abs(y - v) > 1e-9 else None)
ax.set_xlabel("Görev sırası (her görev yeni bir piksel karışımı, 60.000 örnek)"); ax.set_ylabel("Görevdeki çevrimiçi doğruluk")
ax.set_title("Online Permuted MNIST, 800 görev × 60.000 örnek, 3×100 ağ, SGD", loc="left", fontweight="bold", pad=22)
ax.text(0, 1.015, "10 tohum ortalaması (yansımalı sinüs: 3 tohum), 10 görevlik kayan ortalama.", transform=ax.transAxes, fontsize=9.5, color="#52514e")
fig.tight_layout(); fig.savefig(OUT, dpi=160); plt.close(fig); print("wrote", OUT)
