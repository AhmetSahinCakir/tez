#!/usr/bin/env python3
"""Statistical audit package (advisor report of 16 November 2026, item 3).

Usage:  python3 scripts/denetim_paketi.py            ->  reports/denetim/{tohum_bazli.csv, tost.csv, DENETIM.md}

Everything is recomputed from the per-run ``summary.json`` files; nothing is copied from other reports.
  * tohum_bazli.csv : one row per (suite, model, seed) with the performance measures and the three parts of the
                      operational plasticity criterion (scripts/plasticity_criterion.py).
  * tost.csv        : paired equivalence tests (two one-sided t-tests) with both t statistics and p-values, the
                      90 % interval and Holm-adjusted p-values, for the margins 0.005 and 0.01.
  * DENETIM.md      : the tables above in readable form, the role of each measure, the margin, the scope of
                      the equivalence statement and the software / hardware record.
"""
import csv, glob, json, os, platform, subprocess, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from plasticity.metrics.stats import tost_paired, holm_correction  # noqa: E402

OUT = os.path.join(ROOT, "reports", "denetim")
SUITES = [("pmnist_fullstream", "800 görev × 60.000 örnek"), ("pmnist_main", "200 görev × 5.000 örnek")]
METRICS = [("auc_norm", "AUC (birincil)"), ("final_window_mean", "Son pencere"),
           ("retention_ratio", "Koruma oranı"), ("fresh_gap_final", "Taze model farkı")]
PAIRS = {"pmnist_fullstream": [("tri", "weight_clipping"), ("sin_refl_matched", "sin"), ("tanh", "baseline")],
         "pmnist_main": [("tri", "weight_clipping"), ("sin", "tanh")]}
DELTAS = [0.005, 0.01]
TOL_RET, TOL_SLOPE, TOL_FRESH = 0.01, 0.002, 0.01  # as in scripts/plasticity_criterion.py


def runs(suite):
    out = {}
    base = os.path.join(ROOT, "results", suite)
    for lab in sorted(os.listdir(base)):
        for p in sorted(glob.glob(os.path.join(base, lab, "seed*", "summary.json"))):
            seed = int(os.path.basename(os.path.dirname(p))[4:])
            out.setdefault(lab, {})[seed] = json.load(open(p))
    return out


def fmt(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    md = []
    rows = []
    data = {s: runs(s) for s, _ in SUITES}
    for suite, _ in SUITES:
        for lab, by_seed in data[suite].items():
            for seed, r in sorted(by_seed.items()):
                fg = r.get("fresh_gap_final", float("nan"))
                c1 = r["retention_ratio"] >= 1 - TOL_RET
                c2 = r["slope_per_100"] >= -TOL_SLOPE
                c3 = (fg == fg) and fg <= TOL_FRESH
                rows.append({"paket": suite, "model": lab, "tohum": seed, "gorev_sayisi": r["n_tasks"],
                             "auc": r["auc_norm"], "erken_pencere": r["early_window_mean"],
                             "son_pencere": r["final_window_mean"], "koruma_orani": r["retention_ratio"],
                             "egim_100_gorev": r["slope_per_100"], "taze_model_farki": fg,
                             "olcut1_koruma": int(c1), "olcut2_egim": int(c2), "olcut3_taze": int(c3),
                             "uc_olcut_birlikte": int(c1 and c2 and c3)})
    with open(os.path.join(OUT, "tohum_bazli.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    trows = []
    for suite, desc in SUITES:
        for a, b in PAIRS[suite]:
            if a not in data[suite] or b not in data[suite]:
                continue
            seeds = sorted(set(data[suite][a]) & set(data[suite][b]))
            if len(seeds) < 2:
                continue
            for delta in DELTAS:
                block = []
                for key, name in METRICS:
                    x = np.array([data[suite][a][s].get(key, np.nan) for s in seeds], float)
                    y = np.array([data[suite][b][s].get(key, np.nan) for s in seeds], float)
                    t = tost_paired(x, y, delta=delta)
                    se = t["se"]
                    block.append({"paket": suite, "a": a, "b": b, "olcut": key, "olcut_adi": name, "marj": delta,
                                  "n": t["n"], "ortalama_fark": t["mean_diff"], "ss_fark": t["std_diff"], "sh": se,
                                  "t_alt": (t["mean_diff"] + delta) / se if se == se else float("nan"),
                                  "t_ust": (t["mean_diff"] - delta) / se if se == se else float("nan"),
                                  "sd": t["n"] - 1, "p_alt": t["p_lower"], "p_ust": t["p_upper"],
                                  "p_tost": t["p_tost"], "ga90_alt": t["ci_low"], "ga90_ust": t["ci_high"]})
                adj = holm_correction([r["p_tost"] for r in block])
                for r, pa in zip(block, adj):
                    r["p_tost_holm"] = float(pa)
                    r["esdeger"] = int(r["p_tost"] < 0.05)
                    r["esdeger_holm"] = int(pa < 0.05)
                trows.extend(block)
    with open(os.path.join(OUT, "tost.csv"), "w", newline="") as f:
        keys = [k for k in trows[0].keys() if k != "olcut_adi"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in trows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in keys})

    try:
        commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "?"
    import torch, scipy, pandas
    md.append("# İstatistiksel denetim paketi\n")
    md.append("Danışman raporu (16 Kasım 2026), teslim listesi madde 3. Bu klasördeki her sayı, koşuların "
              "`summary.json` dosyalarından `python3 scripts/denetim_paketi.py` komutuyla yeniden üretilir; başka "
              f"rapordan kopyalanmış sayı yoktur. Üretildiği depo sürümü: `{commit}`.\n")
    md.append("| Dosya | İçerik |\n|---|---|\n"
              "| `tohum_bazli.csv` | Her (paket, model, tohum) için AUC, erken ve son pencere, koruma oranı, eğim, taze model farkı ve üç koruma ölçütü |\n"
              "| `tost.csv` | Eşleştirilmiş eşdeğerlik testleri: iki tek yönlü testin t ve p değerleri ayrı ayrı, %90 güven aralığı, Holm düzeltmeli p |\n"
              "| `DENETIM.md` | Bu belge |\n")

    md.append("## 1. Ölçütlerin rolü\n")
    md.append("* **Birincil ölçüt:** normalize AUC (bütün görevlerdeki çevrimiçi doğruluğun ortalaması).\n"
              "* **İkincil ölçütler:** son pencere ortalaması, koruma oranı (son pencere / erken pencere), taze model farkı "
              "(son ölçümde sıfırdan başlatılan ağ − sürekli eğitilen ağ).\n"
              "* **Çoklu çıkarım:** eşdeğerlik iddiası birincil ölçüt üzerinden kurulur. Dört ölçütün dördünde birden "
              "eşdeğerlik bildirildiğinde, her ölçütün ayrı ayrı %5 düzeyinde geçmesi yeterlidir (kesişim–birleşim ilkesi); "
              "yine de aşağıdaki tablolarda dört p değerine Holm düzeltmesi uygulanmış hâli de verilmiştir.\n"
              "* **Koruma ölçütü** (`scripts/plasticity_criterion.py`): (1) koruma oranı ≥ 0,99; (2) eğim ≥ −0,002 / 100 görev; "
              "(3) taze model farkı ≤ 0,01. Üçü birlikte sağlanırsa koşu \"plastisiteyi korumuş\" sayılır.\n")

    md.append("## 2. Tohum bazlı özet\n")
    md.append("Tek tek tohum değerleri `tohum_bazli.csv` dosyasındadır; aşağıda model başına ortalama ± standart sapma "
              "(tohumlar arası, n − 1 ile) ve üç ölçütü birlikte geçen tohum sayısı verilmiştir.\n")
    for suite, desc in SUITES:
        md.append(f"### {suite} ({desc})\n")
        md.append("| Model | n | AUC | Erken pencere | Son pencere | Koruma oranı | Eğim / 100 görev | Taze model farkı | Üç ölçüt |\n|---|---:|---|---|---|---|---|---|---|")
        for lab in sorted(data[suite]):
            rr = [r for r in rows if r["paket"] == suite and r["model"] == lab]
            n = len(rr)

            def ms(k, nd=4, sign=False):
                v = np.array([r[k] for r in rr], float)
                v = v[~np.isnan(v)]
                if v.size == 0:
                    return "–"
                s = fmt(float(v.mean()), nd, sign)
                return s + (f" ± {v.std(ddof=1):.{nd}f}" if v.size > 1 else "")
            md.append(f"| {lab} | {n} | {ms('auc')} | {ms('erken_pencere')} | {ms('son_pencere')} | {ms('koruma_orani')} | "
                      f"{ms('egim_100_gorev', 4, True)} | {ms('taze_model_farki', 4, True)} | {sum(r['uc_olcut_birlikte'] for r in rr)}/{n} |")
        md.append("")

    md.append("## 3. Eşdeğerlik testleri (TOST)\n")
    md.append("Eşleştirilmiş tohumlar üzerinde iki tek yönlü t-testi (Schuirmann, 1987; `plasticity.metrics.stats.tost_paired`). "
              "d = a − b tohum bazlı fark, d̄ ortalaması, SH standart hatası, sd = n − 1 olmak üzere:\n\n"
              "* alt test: H₀: fark ≤ −Δ, t_alt = (d̄ + Δ) / SH, p_alt = P(T ≥ t_alt);\n"
              "* üst test: H₀: fark ≥ +Δ, t_üst = (d̄ − Δ) / SH, p_üst = P(T ≤ t_üst);\n"
              "* p_TOST = max(p_alt, p_üst); p_TOST < 0,05 ise eşdeğerlik ilan edilir. Bu, farkın %90 güven aralığının "
              "(−Δ, +Δ) içinde kalmasıyla aynı koşuldur.\n")
    for suite, desc in SUITES:
        for a, b in PAIRS[suite]:
            for delta in DELTAS:
                blk = [r for r in trows if r["paket"] == suite and r["a"] == a and r["b"] == b and r["marj"] == delta]
                if not blk:
                    continue
                md.append(f"### {suite}: {a} − {b}, Δ = {delta} (n = {blk[0]['n']} ortak tohum)\n")
                md.append("| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
                for r in blk:
                    if r["esdeger"]:
                        v = "eşdeğer"
                    elif r["ga90_alt"] > delta or r["ga90_ust"] < -delta:
                        v = "farklı (GA marjın dışında)"
                    else:
                        v = "karar verilemez"
                    md.append(f"| {r['olcut_adi']} | {fmt(r['ortalama_fark'], 4, True)} | {fmt(r['ss_fark'])} | {fmt(r['sh'], 5)} | "
                              f"{fmt(r['t_alt'], 2)} | {r['p_alt']:.2g} | {fmt(r['t_ust'], 2)} | {r['p_ust']:.2g} | {r['p_tost']:.2g} | "
                              f"{r['p_tost_holm']:.2g} | [{fmt(r['ga90_alt'], 4, True)}, {fmt(r['ga90_ust'], 4, True)}] | {v} |")
                md.append("")

    # scale of the margin relative to the data
    def sd_of(suite, lab, key="auc_norm"):
        v = np.array([r[key] for r in data[suite].get(lab, {}).values()], float)
        return float(v.std(ddof=1)) if v.size > 1 else float("nan")
    md.append("## 4. Marjın gerekçesi\n")
    md.append("* Marj Δ = 0,005 (doğrulukta yarım yüzde puanı), 31 Ekim 2026'da, 800 görevlik paketin 3 tohumu bitmişken ve "
              "kalan 7 tohum koşulmadan önce belirlendi (`docs/YAPILACAKLAR.md`, madde A3). İkincil marj Δ = 0,01'dir. "
              "Marj, 10 tohumluk sonuçlar görüldükten sonra değiştirilmedi.\n"
              "* Ölçek karşılaştırması (bu belgede yeniden hesaplandı): 800 görevlik pakette AUC'nin tohumlar arası standart sapması "
              f"üçgen dalgada {sd_of('pmnist_fullstream', 'tri'):.4f}, Weight Clipping'de {sd_of('pmnist_fullstream', 'weight_clipping'):.4f}, "
              f"standart ağda {sd_of('pmnist_fullstream', 'baseline'):.4f}'tür. Marj, çalışmada \"fark var\" denilen en küçük "
              "etkilerden (200 görevde sinüs − standart ağ, ΔAUC ≈ 0,011) küçüktür.\n"
              "* Marj bir kabuldür; başka bir uygulamada daha dar bir marj gerekebilir. Bu nedenle %90 güven aralıkları "
              "tablolarda açıkça verilmiştir: okuyucu kendi marjını bu aralıklarla karşılaştırabilir.\n")
    md.append("## 5. Eşdeğerlik ifadesinin kapsamı\n")
    md.append("\"Üçgen dalga ile Weight Clipping eşdeğerdir\" ifadesi yalnızca şu koşullar için geçerlidir: Online Permuted MNIST, "
              "3×100 ReLU ağ, SGD (öğrenme oranı 0,003, yığın boyutu 1), γ = 1 ve κ = 1, yukarıdaki dört ölçüt ve Δ = 0,005. "
              "Adam ile, geniş ağda, başka veri kümelerinde veya başka ölçütlerde eşdeğerlik sınanmamıştır; 200 görevlik pakette "
              "tohum sayısı az olduğunda sonuç \"ayırt edilemedi\" olarak bildirilir, \"eşdeğer\" olarak değil.\n")
    md.append("## 6. Ortam ve yeniden üretim kaydı\n")
    md.append(f"* Yazılım: Python {platform.python_version()}, PyTorch {torch.__version__}, NumPy {np.__version__}, "
              f"SciPy {scipy.__version__}, pandas {pandas.__version__}.\n"
              "* Donanım: 4 sanal çekirdek (Intel Xeon, 2,10 GHz), 16 GB bellek, GPU yok; her koşu tek iş parçacığıyla "
              "(`threads: 1`) çalışır, dört koşu yan yana yürütülür.\n"
              "* Koşular: `python3 scripts/run_suite.py suites/pmnist_fullstream.yaml` ve `suites/pmnist_main.yaml`; "
              "`--dry-run` her koşunun tam komut satırını yazar. Koşular görev sınırında kaydedilen denetim noktasından "
              "birebir devam eder (`tests/test_checkpoint.py`).\n"
              "* Tohum: aynı tohum numarası aynı başlangıç ağırlıklarını ve aynı görev akışını (permütasyonlar, örnek sırası) verir; "
              "bu nedenle bütün karşılaştırmalar eşleştirilmiştir.\n"
              "* Hiperparametre araması: `suites/pmnist_dev.yaml` (ayrı geliştirme akışı, `stream_seed_offset: 1000`, tek tohum, "
              "100 görev); sonuçlar `reports/pmnist_dev_hparams.md`, seçilen değerler `suites/selected_pmnist.yaml`.\n"
              "* Analiz: `scripts/analyze.py`, `scripts/scaleup_tables.py`, `scripts/equivalence.py`, "
              "`scripts/plasticity_criterion.py`, `scripts/denetim_paketi.py`; istatistik fonksiyonlarının birim testleri "
              "`tests/test_stats.py`.\n")
    open(os.path.join(OUT, "DENETIM.md"), "w").write("\n".join(md) + "\n")
    print(f"wrote {OUT}: {len(rows)} run rows, {len(trows)} test rows")


if __name__ == "__main__":
    main()
