# Konferans bildirisinin (kisa protokol: 200 gorev x 5.000 ornek) sekilleri ve istatistikleri.
# Kullanim: python3 scripts/bildiri_figures.py
# Okur: results/pmnist_main/<etiket>/seed*/{summary.json,tasks.jsonl}
# Yazar: bildiri/figures/*.pdf ve bildiri/istatistikler.md
import glob, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SRC, OUT = "results/pmnist_main", "bildiri"
os.makedirs(f"{OUT}/figures", exist_ok=True)
LABELS = ["baseline", "sin", "tanh", "tri", "weight_clipping"]
COL = {"baseline": "#2a78d6", "sin": "#eb6834", "tanh": "#1baf7a", "tri": "#eda100", "weight_clipping": "#d6498a"}
NAME = {"baseline": "Standart", "sin": "Sinüs", "tanh": "Tanh", "tri": "Üçgen dalga", "weight_clipping": "Ağırlık kırpma"}
LS = {"weight_clipping": (0, (4, 2))}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False, "font.family": "DejaVu Sans"})


def grid(ax):
    ax.grid(True, color="#e6e5e1", linewidth=0.6); ax.set_axisbelow(True)


# ---------------------------------------------------------------- veriyi oku
summ, rows = {}, []
for lab in LABELS:
    for d in sorted(glob.glob(f"{SRC}/{lab}/seed*")):
        s = json.load(open(f"{d}/summary.json")); summ.setdefault(lab, {})[s["seed"]] = s
        for line in open(f"{d}/tasks.jsonl"):
            r = json.loads(line)
            rows.append({"label": lab, "seed": s["seed"], "task": r["task"], "acc": r["online_accuracy"],
                         "jac_sq": r.get("jac_sq_mean/all"), "w_abs": r.get("w_abs_mean/all"),
                         "sat": r.get("sat_frac/all")})
df = pd.DataFrame(rows)


def band(lab, key, win=1):
    p = df[df.label == lab].dropna(subset=[key]).pivot(index="task", columns="seed", values=key).sort_index()
    if win > 1: p = p.rolling(win, min_periods=1, center=True).mean()
    n = p.shape[1]
    return p.index.values + 1, p.mean(axis=1).values, (stats.t.ppf(0.975, n - 1) * p.std(axis=1, ddof=1) / np.sqrt(n)).values


def ci(v):
    v = np.asarray(v, float); n = len(v)
    return v.mean(), stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)


# ---------------------------------------------------------------- Sekil 1: haritalar ve turevleri
x = np.linspace(-3.2, 3.2, 1301)
maps = [("sin", np.sin(x), np.abs(np.cos(x))), ("tanh", np.tanh(x), 1 - np.tanh(x) ** 2),
        ("tri", 2 / np.pi * np.arcsin(np.sin(np.pi * x / 2)), np.ones_like(x))]
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75))
for lab, f, dfx in maps:
    axs[0].plot(x, f, color=COL[lab], lw=1.4, label=NAME[lab]); axs[1].plot(x, dfx, color=COL[lab], lw=1.4)
for ax, yl in zip(axs, [r"$W/A=f(\Theta/A)$", r"$|f'(\Theta/A)|$"]):
    grid(ax); ax.set_xlabel(r"$\Theta/A$"); ax.set_ylabel(yl, labelpad=1)
axs[0].axhline(1, color="#b8b7b2", ls="--", lw=0.7); axs[0].axhline(-1, color="#b8b7b2", ls="--", lw=0.7)
axs[1].set_ylim(-0.05, 1.15)
fig.legend(loc="upper center", ncol=3, fontsize=7, handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.6); fig.savefig(f"{OUT}/figures/sekil_haritalar.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Sekil 2: cevrim ici dogruluk
fig, ax = plt.subplots(figsize=(3.45, 2.25))
for lab in LABELS:
    t, m, h = band(lab, "acc", win=5)
    ax.fill_between(t, m - h, m + h, color=COL[lab], alpha=0.18, lw=0)
    ax.plot(t, m, color=COL[lab], lw=1.4, ls=LS.get(lab, "-"), label=NAME[lab])
grid(ax); ax.set_xlabel("Görev"); ax.set_ylabel("Çevrim içi doğruluk"); ax.set_xlim(1, 200)
ax.legend(fontsize=7, ncol=1, loc="lower left", handlelength=2.0, labelspacing=0.25, borderaxespad=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/figures/sekil_dogruluk.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Sekil 3: mekanizma
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9))
for lab in ["sin", "tanh", "tri"]:
    t, m, h = band(lab, "jac_sq")
    axs[0].fill_between(t, m - h, m + h, color=COL[lab], alpha=0.18, lw=0); axs[0].plot(t, m, color=COL[lab], lw=1.4)
for lab in LABELS:
    t, m, h = band(lab, "w_abs")
    axs[1].fill_between(t, m - h, m + h, color=COL[lab], alpha=0.18, lw=0)
    axs[1].plot(t, m, color=COL[lab], lw=1.4, ls=LS.get(lab, "-"), label=NAME[lab])
for ax, yl in zip(axs, [r"Adım çarpanı $\langle f'^2\rangle$", r"Ortalama $|W|$"]):
    grid(ax); ax.set_xlabel("Görev"); ax.set_ylabel(yl, labelpad=1); ax.set_xlim(1, 200)
axs[0].set_ylim(0.4, 1.05)
fig.legend(loc="upper center", ncol=3, fontsize=6.5, handlelength=1.8, columnspacing=1.0, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=0.6); fig.savefig(f"{OUT}/figures/sekil_mekanizma.pdf"); plt.close(fig)

# ---------------------------------------------------------------- istatistikler
KEYS = [("auc_norm", "AUC"), ("early_window_mean", "ilk pencere"), ("final_window_mean", "son pencere"),
        ("retention_ratio", "koruma orani"), ("slope_per_100", "egim/100 gorev"), ("fresh_gap_final", "taze model farki"),
        ("final_jac_sq_mean/all", "adim carpani"), ("final_sat_frac/all", "doymus oran"), ("final_w_over_A_mean/all", "|W|/A"),
        ("early_w_abs_mean/all", "|W| ilk"), ("final_w_abs_mean/all", "|W| son"), ("final_dead_frac/all", "olu birim"),
        ("early_effective_rank/last", "etkin kerte ilk"), ("final_effective_rank/last", "etkin kerte son"),
        ("early_step_dw_norm", "adim normu ilk"), ("final_step_dw_norm", "adim normu son")]
out = ["# Bildiri istatistikleri (200 gorev x 5.000 ornek, results/pmnist_main)", "",
       "Ortalama ± %95 t-araligi (tohumlar uzerinden).", ""]
out.append("| Model | n | " + " | ".join(k for _, k in KEYS) + " |"); out.append("|---|---|" + "---|" * len(KEYS))
for lab in LABELS:
    cells = []
    for k, _ in KEYS:
        v = [summ[lab][s].get(k) for s in sorted(summ[lab])]
        cells.append("-" if any(a is None for a in v) else "%.4f ± %.4f" % ci(v))
    out.append(f"| {NAME[lab]} | {len(summ[lab])} | " + " | ".join(cells) + " |")
out += ["", "## Eslestirilmis farklar (ayni tohumlar; %95 t-araligi, eslestirilmis t-testi)", "",
        "| Karsilastirma | Olcut | n | Fark | %95 GA | p | Kazanilan tohum |", "|---|---|---|---|---|---|---|"]
for a, b in [("sin", "baseline"), ("tanh", "baseline"), ("sin", "tanh"), ("tri", "baseline"), ("tri", "sin"),
             ("tri", "tanh"), ("weight_clipping", "baseline"), ("tri", "weight_clipping")]:
    for k, kn in KEYS[:4]:
        seeds = sorted(set(summ[a]) & set(summ[b])); d = np.array([summ[a][s][k] - summ[b][s][k] for s in seeds])
        m, h = ci(d); p = stats.ttest_1samp(d, 0).pvalue
        out.append(f"| {NAME[a]} − {NAME[b]} | {kn} | {len(d)} | {m:+.4f} | [{m - h:+.4f}, {m + h:+.4f}] | {p:.2g} | {int((d > 0).sum())}/{len(d)} |")
out += ["", "## Ucgen dalga ile agirlik kirpma: esdegerlik (TOST, sinir 0.005, eslestirilmis)", "",
        "| Olcut | n | Fark | %90 GA | p_TOST |", "|---|---|---|---|---|"]
for k, kn in KEYS[:4]:
    seeds = sorted(set(summ["tri"]) & set(summ["weight_clipping"]))
    d = np.array([summ["tri"][s][k] - summ["weight_clipping"][s][k] for s in seeds]); n = len(d)
    se = d.std(ddof=1) / np.sqrt(n); m = d.mean(); h90 = stats.t.ppf(0.95, n - 1) * se
    p = max(1 - stats.t.cdf((m + 0.005) / se, n - 1), stats.t.cdf((m - 0.005) / se, n - 1))
    out.append(f"| {kn} | {n} | {m:+.4f} | [{m - h90:+.4f}, {m + h90:+.4f}] | {p:.3g} |")
open(f"{OUT}/istatistikler.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ---------------------------------------------------------------- LaTeX tablolari (ondalik virgul)
def tr(x, nd=3, sign=False):
    return (("%+." if sign else "%.") + f"{nd}f") % x


def cell(lab, k, nd=3, sign=False):
    v = [summ[lab][s].get(k) for s in sorted(summ[lab])]
    if any(a is None for a in v): return "--"
    m, h = ci(v)
    return ("$%s\\pm%s$" % (tr(m, nd, sign), tr(h, nd))).replace(".", "{,}").replace("-", "-")


def pair(lab, k1, k2, nd=3):
    a = np.mean([summ[lab][s][k1] for s in summ[lab]]); b = np.mean([summ[lab][s][k2] for s in summ[lab]])
    return ("$%s\\to%s$" % (tr(a, nd), tr(b, nd))).replace(".", "{,}")


rows = []
for lab in ["tri", "weight_clipping", "sin", "tanh", "baseline"]:
    rows.append(" & ".join([NAME[lab], str(len(summ[lab])), cell(lab, "auc_norm"), cell(lab, "final_window_mean"),
                            cell(lab, "retention_ratio"), cell(lab, "fresh_gap_final", sign=True)]) + r"\\")
open(f"{OUT}/tablo_sonuclar.tex", "w").write("\n".join(rows) + "\n")
rows = []
for lab in ["tri", "weight_clipping", "sin", "tanh", "baseline"]:
    ratio = np.mean([summ[lab][s]["final_step_dw_norm"] / summ[lab][s]["early_step_dw_norm"] for s in summ[lab]])
    rows.append(" & ".join([NAME[lab], cell(lab, "final_jac_sq_mean/all", 2).split("\\pm")[0] + "$" if cell(lab, "final_jac_sq_mean/all") != "--" else "--",
                            ("$%.1f$" % (100 * np.mean([summ[lab][s]["final_sat_frac/all"] for s in summ[lab]]))).replace(".", "{,}") if "final_sat_frac/all" in summ[lab][0] else "--",
                            pair(lab, "early_w_abs_mean/all", "final_w_abs_mean/all"),
                            ("$%.2f$" % ratio).replace(".", "{,}"),
                            ("$%.0f\\to%.0f$" % (np.mean([summ[lab][s]["early_effective_rank/last"] for s in summ[lab]]),
                                                np.mean([summ[lab][s]["final_effective_rank/last"] for s in summ[lab]])))]) + r"\\")
open(f"{OUT}/tablo_mekanizma.tex", "w").write("\n".join(rows) + "\n")
first = {lab: band(lab, "jac_sq")[1][0] for lab in ["sin", "tanh", "tri"]}
print("ilk olcumde adim carpani:", first)
print(open(f"{OUT}/tablo_sonuclar.tex").read()); print(open(f"{OUT}/tablo_mekanizma.tex").read())
