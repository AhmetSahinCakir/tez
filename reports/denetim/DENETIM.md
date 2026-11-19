# İstatistiksel denetim paketi

Danışman raporu (16 Kasım 2026), teslim listesi madde 3. Bu klasördeki her sayı, koşuların `summary.json` dosyalarından `python3 scripts/denetim_paketi.py` komutuyla yeniden üretilir; başka rapordan kopyalanmış sayı yoktur. Üretildiği depo sürümü: `53b0af6`.

| Dosya | İçerik |
|---|---|
| `tohum_bazli.csv` | Her (paket, model, tohum) için AUC, erken ve son pencere, koruma oranı, eğim, taze model farkı ve üç koruma ölçütü |
| `tost.csv` | Eşleştirilmiş eşdeğerlik testleri: iki tek yönlü testin t ve p değerleri ayrı ayrı, %90 güven aralığı, Holm düzeltmeli p |
| `DENETIM.md` | Bu belge |

## 1. Ölçütlerin rolü

* **Birincil ölçüt:** normalize AUC (bütün görevlerdeki çevrimiçi doğruluğun ortalaması).
* **İkincil ölçütler:** son pencere ortalaması, koruma oranı (son pencere / erken pencere), taze model farkı (son ölçümde sıfırdan başlatılan ağ − sürekli eğitilen ağ).
* **Çoklu çıkarım:** eşdeğerlik iddiası birincil ölçüt üzerinden kurulur. Dört ölçütün dördünde birden eşdeğerlik bildirildiğinde, her ölçütün ayrı ayrı %5 düzeyinde geçmesi yeterlidir (kesişim–birleşim ilkesi); yine de aşağıdaki tablolarda dört p değerine Holm düzeltmesi uygulanmış hâli de verilmiştir.
* **Koruma ölçütü** (`scripts/plasticity_criterion.py`): (1) koruma oranı ≥ 0,99; (2) eğim ≥ −0,002 / 100 görev; (3) taze model farkı ≤ 0,01. Üçü birlikte sağlanırsa koşu "plastisiteyi korumuş" sayılır.

## 2. Tohum bazlı özet

Tek tek tohum değerleri `tohum_bazli.csv` dosyasındadır; aşağıda model başına ortalama ± standart sapma (tohumlar arası, n − 1 ile) ve üç ölçütü birlikte geçen tohum sayısı verilmiştir.

### pmnist_fullstream (800 görev × 60.000 örnek)

| Model | n | AUC | Erken pencere | Son pencere | Koruma oranı | Eğim / 100 görev | Taze model farkı | Üç ölçüt |
|---|---:|---|---|---|---|---|---|---|
| baseline | 10 | 0.8564 ± 0.0011 | 0.9044 ± 0.0007 | 0.8355 ± 0.0016 | 0.9238 ± 0.0019 | -0.0083 ± 0.0003 | +0.0845 ± 0.0080 | 0/10 |
| ln_wd | 10 | 0.9249 ± 0.0006 | 0.9303 ± 0.0006 | 0.9227 ± 0.0011 | 0.9918 ± 0.0012 | -0.0008 ± 0.0001 | +0.0080 ± 0.0037 | 8/10 |
| sin | 10 | 0.8178 ± 0.0029 | 0.9092 ± 0.0005 | 0.7204 ± 0.0172 | 0.7923 ± 0.0188 | -0.0237 ± 0.0019 | +0.2118 ± 0.0213 | 0/10 |
| sin_refl | 3 | 0.9239 ± 0.0002 | 0.9256 ± 0.0000 | 0.9202 ± 0.0005 | 0.9942 ± 0.0005 | -0.0010 ± 0.0003 | +0.0045 ± 0.0055 | 3/3 |
| sin_refl_matched | 8 | 0.9223 ± 0.0005 | 0.9243 ± 0.0012 | 0.9214 ± 0.0011 | 0.9969 ± 0.0018 | -0.0002 ± 0.0002 | -0.0005 ± 0.0035 | 8/8 |
| tanh | 10 | 0.8590 ± 0.0012 | 0.9111 ± 0.0003 | 0.8305 ± 0.0034 | 0.9115 ± 0.0038 | -0.0101 ± 0.0004 | +0.0853 ± 0.0039 | 0/10 |
| tri | 10 | 0.9389 ± 0.0005 | 0.9345 ± 0.0008 | 0.9396 ± 0.0009 | 1.0054 ± 0.0012 | +0.0004 ± 0.0002 | -0.0234 ± 0.0021 | 10/10 |
| weight_clipping | 10 | 0.9386 ± 0.0003 | 0.9343 ± 0.0011 | 0.9396 ± 0.0007 | 1.0058 ± 0.0012 | +0.0005 ± 0.0001 | -0.0222 ± 0.0035 | 10/10 |

### pmnist_main (200 görev × 5.000 örnek)

| Model | n | AUC | Erken pencere | Son pencere | Koruma oranı | Eğim / 100 görev | Taze model farkı | Üç ölçüt |
|---|---:|---|---|---|---|---|---|---|
| baseline | 10 | 0.7743 ± 0.0012 | 0.8207 ± 0.0020 | 0.7432 ± 0.0044 | 0.9057 ± 0.0054 | -0.0407 ± 0.0030 | +0.0212 ± 0.0114 | 0/10 |
| continual_backprop | 5 | 0.8209 ± 0.0009 | 0.8185 ± 0.0022 | 0.8212 ± 0.0038 | 1.0032 ± 0.0065 | +0.0011 ± 0.0019 | -0.0549 ± 0.0056 | 4/5 |
| l2_init | 5 | 0.8304 ± 0.0006 | 0.8254 ± 0.0019 | 0.8312 ± 0.0017 | 1.0071 ± 0.0033 | +0.0020 ± 0.0009 | -0.0670 ± 0.0051 | 5/5 |
| ln_wd | 5 | 0.8163 ± 0.0013 | 0.8284 ± 0.0020 | 0.8088 ± 0.0028 | 0.9762 ± 0.0043 | -0.0101 ± 0.0018 | +0.0029 ± 0.0062 | 0/5 |
| nap | 5 | 0.8329 ± 0.0004 | 0.8283 ± 0.0027 | 0.8338 ± 0.0011 | 1.0067 ± 0.0035 | +0.0013 ± 0.0008 | -0.0141 ± 0.0028 | 5/5 |
| parseval | 3 | 0.8438 ± 0.0006 | 0.8300 ± 0.0014 | 0.8437 ± 0.0017 | 1.0164 ± 0.0038 | +0.0046 ± 0.0010 | -0.0951 ± 0.0094 | 3/3 |
| shrink_perturb | 3 | 0.8095 ± 0.0002 | 0.8098 ± 0.0017 | 0.8100 ± 0.0005 | 1.0002 ± 0.0027 | +0.0004 ± 0.0009 | -0.0184 ± 0.0066 | 3/3 |
| sin | 10 | 0.7853 ± 0.0009 | 0.8208 ± 0.0017 | 0.7603 ± 0.0052 | 0.9263 ± 0.0053 | -0.0338 ± 0.0013 | -0.0077 ± 0.0127 | 0/10 |
| sin_act | 3 | 0.8092 ± 0.0006 | 0.8323 ± 0.0019 | 0.7990 ± 0.0009 | 0.9601 ± 0.0026 | -0.0173 ± 0.0015 | +0.0144 ± 0.0033 | 0/3 |
| sin_floor | 5 | 0.7859 ± 0.0016 | 0.8199 ± 0.0025 | 0.7645 ± 0.0033 | 0.9324 ± 0.0034 | -0.0313 ± 0.0023 | +0.0106 ± 0.0119 | 0/5 |
| sin_refl | 10 | 0.8258 ± 0.0015 | 0.8289 ± 0.0015 | 0.8250 ± 0.0043 | 0.9953 ± 0.0060 | -0.0012 ± 0.0025 | -0.0257 ± 0.0088 | 6/10 |
| smooth_leaky | 3 | 0.7992 ± 0.0002 | 0.8121 ± 0.0022 | 0.7841 ± 0.0012 | 0.9655 ± 0.0038 | -0.0186 ± 0.0007 | -0.0561 ± 0.0126 | 0/3 |
| tanh | 10 | 0.7852 ± 0.0012 | 0.8178 ± 0.0026 | 0.7611 ± 0.0033 | 0.9306 ± 0.0050 | -0.0326 ± 0.0020 | -0.0192 ± 0.0168 | 0/10 |
| tanh_refl | 5 | 0.7994 ± 0.0017 | 0.8305 ± 0.0019 | 0.7798 ± 0.0051 | 0.9390 ± 0.0052 | -0.0264 ± 0.0017 | +0.0144 ± 0.0127 | 0/5 |
| tri | 10 | 0.8358 ± 0.0026 | 0.8256 ± 0.0020 | 0.8395 ± 0.0033 | 1.0168 ± 0.0043 | +0.0051 ± 0.0022 | -0.0800 ± 0.0103 | 10/10 |
| upgd | 3 | 0.8000 ± 0.0012 | 0.8265 ± 0.0003 | 0.7859 ± 0.0024 | 0.9509 ± 0.0026 | -0.0213 ± 0.0009 | +0.0108 ± 0.0039 | 0/3 |
| weight_clipping | 8 | 0.8365 ± 0.0009 | 0.8259 ± 0.0021 | 0.8394 ± 0.0029 | 1.0163 ± 0.0039 | +0.0058 ± 0.0015 | -0.0799 ± 0.0093 | 8/8 |

## 3. Eşdeğerlik testleri (TOST)

Eşleştirilmiş tohumlar üzerinde iki tek yönlü t-testi (Schuirmann, 1987; `plasticity.metrics.stats.tost_paired`). d = a − b tohum bazlı fark, d̄ ortalaması, SH standart hatası, sd = n − 1 olmak üzere:

* alt test: H₀: fark ≤ −Δ, t_alt = (d̄ + Δ) / SH, p_alt = P(T ≥ t_alt);
* üst test: H₀: fark ≥ +Δ, t_üst = (d̄ − Δ) / SH, p_üst = P(T ≤ t_üst);
* p_TOST = max(p_alt, p_üst); p_TOST < 0,05 ise eşdeğerlik ilan edilir. Bu, farkın %90 güven aralığının (−Δ, +Δ) içinde kalmasıyla aynı koşuldur.

### pmnist_fullstream: tri − weight_clipping, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0003 | 0.0005 | 0.00016 | 32.47 | 6.1e-11 | -28.73 | 1.8e-10 | 1.8e-10 | 7.3e-10 | [+0.0000, +0.0006] | eşdeğer |
| Son pencere | -0.0001 | 0.0009 | 0.00029 | 17.13 | 1.8e-08 | -17.58 | 1.4e-08 | 1.8e-08 | 5.3e-08 | [-0.0006, +0.0005] | eşdeğer |
| Koruma oranı | -0.0003 | 0.0020 | 0.00063 | 7.40 | 2e-05 | -8.41 | 7.4e-06 | 2e-05 | 4.1e-05 | [-0.0015, +0.0008] | eşdeğer |
| Taze model farkı | -0.0012 | 0.0028 | 0.00090 | 4.24 | 0.0011 | -6.87 | 3.7e-05 | 0.0011 | 0.0011 | [-0.0028, +0.0005] | eşdeğer |

### pmnist_fullstream: tri − weight_clipping, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0003 | 0.0005 | 0.00016 | 63.08 | 1.6e-13 | -59.33 | 2.8e-13 | 2.8e-13 | 1.1e-12 | [+0.0000, +0.0006] | eşdeğer |
| Son pencere | -0.0001 | 0.0009 | 0.00029 | 34.49 | 3.6e-11 | -34.94 | 3.2e-11 | 3.6e-11 | 1.1e-10 | [-0.0006, +0.0005] | eşdeğer |
| Koruma oranı | -0.0003 | 0.0020 | 0.00063 | 15.31 | 4.7e-08 | -16.31 | 2.7e-08 | 4.7e-08 | 9.5e-08 | [-0.0015, +0.0008] | eşdeğer |
| Taze model farkı | -0.0012 | 0.0028 | 0.00090 | 9.79 | 2.1e-06 | -12.42 | 2.9e-07 | 2.1e-06 | 2.1e-06 | [-0.0028, +0.0005] | eşdeğer |

### pmnist_fullstream: sin_refl_matched − sin, Δ = 0.005 (n = 8 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.1044 | 0.0020 | 0.00072 | 152.36 | 6.9e-14 | 138.44 | 1 | 1 | 1 | [+0.1031, +0.1058] | farklı (GA marjın dışında) |
| Son pencere | +0.1990 | 0.0131 | 0.00464 | 43.98 | 4.1e-10 | 41.82 | 1 | 1 | 1 | [+0.1902, +0.2078] | farklı (GA marjın dışında) |
| Koruma oranı | +0.2024 | 0.0143 | 0.00506 | 40.99 | 6.7e-10 | 39.02 | 1 | 1 | 1 | [+0.1928, +0.2120] | farklı (GA marjın dışında) |
| Taze model farkı | -0.2110 | 0.0223 | 0.00788 | -26.16 | 1 | -27.43 | 1.1e-08 | 1 | 1 | [-0.2259, -0.1961] | farklı (GA marjın dışında) |

### pmnist_fullstream: sin_refl_matched − sin, Δ = 0.01 (n = 8 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.1044 | 0.0020 | 0.00072 | 159.33 | 5.1e-14 | 131.48 | 1 | 1 | 1 | [+0.1031, +0.1058] | farklı (GA marjın dışında) |
| Son pencere | +0.1990 | 0.0131 | 0.00464 | 45.06 | 3.5e-10 | 40.75 | 1 | 1 | 1 | [+0.1902, +0.2078] | farklı (GA marjın dışında) |
| Koruma oranı | +0.2024 | 0.0143 | 0.00506 | 41.98 | 5.7e-10 | 38.03 | 1 | 1 | 1 | [+0.1928, +0.2120] | farklı (GA marjın dışında) |
| Taze model farkı | -0.2110 | 0.0223 | 0.00788 | -25.52 | 1 | -28.06 | 9.4e-09 | 1 | 1 | [-0.2259, -0.1961] | farklı (GA marjın dışında) |

### pmnist_fullstream: tanh − baseline, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0026 | 0.0020 | 0.00062 | 12.37 | 3e-07 | -3.83 | 0.002 | 0.002 | 0.008 | [+0.0015, +0.0038] | eşdeğer |
| Son pencere | -0.0050 | 0.0042 | 0.00132 | 0.03 | 0.49 | -7.56 | 1.7e-05 | 0.49 | 0.97 | [-0.0074, -0.0025] | karar verilemez |
| Koruma oranı | -0.0123 | 0.0042 | 0.00133 | -5.48 | 1 | -13.03 | 1.9e-07 | 1 | 1 | [-0.0147, -0.0098] | farklı (GA marjın dışında) |
| Taze model farkı | +0.0008 | 0.0074 | 0.00235 | 2.47 | 0.018 | -1.79 | 0.053 | 0.053 | 0.16 | [-0.0035, +0.0051] | karar verilemez |

### pmnist_fullstream: tanh − baseline, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0026 | 0.0020 | 0.00062 | 20.47 | 3.7e-09 | -11.93 | 4e-07 | 4e-07 | 1.6e-06 | [+0.0015, +0.0038] | eşdeğer |
| Son pencere | -0.0050 | 0.0042 | 0.00132 | 3.83 | 0.002 | -11.36 | 6.1e-07 | 0.002 | 0.0052 | [-0.0074, -0.0025] | eşdeğer |
| Koruma oranı | -0.0123 | 0.0042 | 0.00133 | -1.71 | 0.94 | -16.80 | 2.1e-08 | 0.94 | 0.94 | [-0.0147, -0.0098] | karar verilemez |
| Taze model farkı | +0.0008 | 0.0074 | 0.00235 | 4.60 | 0.00064 | -3.92 | 0.0017 | 0.0017 | 0.0052 | [-0.0035, +0.0051] | eşdeğer |

### pmnist_main: tri − weight_clipping, Δ = 0.005 (n = 8 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | -0.0004 | 0.0024 | 0.00085 | 5.41 | 0.0005 | -6.34 | 0.00019 | 0.0005 | 0.0017 | [-0.0020, +0.0012] | eşdeğer |
| Son pencere | +0.0002 | 0.0024 | 0.00086 | 6.11 | 0.00024 | -5.55 | 0.00043 | 0.00043 | 0.0017 | [-0.0014, +0.0019] | eşdeğer |
| Koruma oranı | +0.0008 | 0.0033 | 0.00118 | 4.90 | 0.00088 | -3.57 | 0.0045 | 0.0045 | 0.0091 | [-0.0015, +0.0030] | eşdeğer |
| Taze model farkı | +0.0026 | 0.0061 | 0.00215 | 3.53 | 0.0048 | -1.13 | 0.15 | 0.15 | 0.15 | [-0.0015, +0.0066] | karar verilemez |

### pmnist_main: tri − weight_clipping, Δ = 0.01 (n = 8 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | -0.0004 | 0.0024 | 0.00085 | 11.29 | 4.8e-06 | -12.22 | 2.8e-06 | 4.8e-06 | 1.8e-05 | [-0.0020, +0.0012] | eşdeğer |
| Son pencere | +0.0002 | 0.0024 | 0.00086 | 11.94 | 3.3e-06 | -11.38 | 4.5e-06 | 4.5e-06 | 1.8e-05 | [-0.0014, +0.0019] | eşdeğer |
| Koruma oranı | +0.0008 | 0.0033 | 0.00118 | 9.13 | 1.9e-05 | -7.80 | 5.3e-05 | 5.3e-05 | 0.00011 | [-0.0015, +0.0030] | eşdeğer |
| Taze model farkı | +0.0026 | 0.0061 | 0.00215 | 5.85 | 0.00031 | -3.46 | 0.0053 | 0.0053 | 0.0053 | [-0.0015, +0.0066] | eşdeğer |

### pmnist_main: sin − tanh, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0001 | 0.0017 | 0.00053 | 9.61 | 2.5e-06 | -9.28 | 3.3e-06 | 3.3e-06 | 1.3e-05 | [-0.0009, +0.0011] | eşdeğer |
| Son pencere | -0.0008 | 0.0060 | 0.00190 | 2.24 | 0.026 | -3.03 | 0.0071 | 0.026 | 0.078 | [-0.0042, +0.0027] | eşdeğer |
| Koruma oranı | -0.0043 | 0.0083 | 0.00264 | 0.27 | 0.4 | -3.52 | 0.0033 | 0.4 | 0.79 | [-0.0091, +0.0005] | karar verilemez |
| Taze model farkı | +0.0115 | 0.0119 | 0.00376 | 4.40 | 0.00087 | 1.74 | 0.94 | 0.94 | 0.94 | [+0.0046, +0.0184] | karar verilemez |

### pmnist_main: sin − tanh, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0001 | 0.0017 | 0.00053 | 19.05 | 7e-09 | -18.72 | 8.1e-09 | 8.1e-09 | 3.3e-08 | [-0.0009, +0.0011] | eşdeğer |
| Son pencere | -0.0008 | 0.0060 | 0.00190 | 4.87 | 0.00044 | -5.67 | 0.00015 | 0.00044 | 0.0013 | [-0.0042, +0.0027] | eşdeğer |
| Koruma oranı | -0.0043 | 0.0083 | 0.00264 | 2.16 | 0.029 | -5.41 | 0.00021 | 0.029 | 0.059 | [-0.0091, +0.0005] | eşdeğer |
| Taze model farkı | +0.0115 | 0.0119 | 0.00376 | 5.72 | 0.00014 | 0.41 | 0.65 | 0.65 | 0.65 | [+0.0046, +0.0184] | karar verilemez |

## 4. Marjın gerekçesi

* Marj Δ = 0,005 (doğrulukta yarım yüzde puanı), 31 Ekim 2026'da, 800 görevlik paketin 3 tohumu bitmişken ve kalan 7 tohum koşulmadan önce belirlendi (`docs/YAPILACAKLAR.md`, madde A3). İkincil marj Δ = 0,01'dir. Marj, 10 tohumluk sonuçlar görüldükten sonra değiştirilmedi.
* Ölçek karşılaştırması (bu belgede yeniden hesaplandı): 800 görevlik pakette AUC'nin tohumlar arası standart sapması üçgen dalgada 0.0005, Weight Clipping'de 0.0003, standart ağda 0.0011'tür. Marj, çalışmada "fark var" denilen en küçük etkilerden (200 görevde sinüs − standart ağ, ΔAUC ≈ 0,011) küçüktür.
* Marj bir kabuldür; başka bir uygulamada daha dar bir marj gerekebilir. Bu nedenle %90 güven aralıkları tablolarda açıkça verilmiştir: okuyucu kendi marjını bu aralıklarla karşılaştırabilir.

## 5. Eşdeğerlik ifadesinin kapsamı

"Üçgen dalga ile Weight Clipping eşdeğerdir" ifadesi yalnızca şu koşullar için geçerlidir: Online Permuted MNIST, 3×100 ReLU ağ, SGD (öğrenme oranı 0,003, yığın boyutu 1), γ = 1 ve κ = 1, yukarıdaki dört ölçüt ve Δ = 0,005. Adam ile, geniş ağda, başka veri kümelerinde veya başka ölçütlerde eşdeğerlik sınanmamıştır; 200 görevlik pakette tohum sayısı az olduğunda sonuç "ayırt edilemedi" olarak bildirilir, "eşdeğer" olarak değil.

## 6. Ortam ve yeniden üretim kaydı

* Yazılım: Python 3.13.18, PyTorch 2.15.0+cu132, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5.
* Donanım: 4 sanal çekirdek (Intel Xeon, 2,10 GHz), 16 GB bellek, GPU yok; her koşu tek iş parçacığıyla (`threads: 1`) çalışır, dört koşu yan yana yürütülür.
* Koşular: `python3 scripts/run_suite.py suites/pmnist_fullstream.yaml` ve `suites/pmnist_main.yaml`; `--dry-run` her koşunun tam komut satırını yazar. Koşular görev sınırında kaydedilen denetim noktasından birebir devam eder (`tests/test_checkpoint.py`).
* Tohum: aynı tohum numarası aynı başlangıç ağırlıklarını ve aynı görev akışını (permütasyonlar, örnek sırası) verir; bu nedenle bütün karşılaştırmalar eşleştirilmiştir.
* Hiperparametre araması: `suites/pmnist_dev.yaml` (ayrı geliştirme akışı, `stream_seed_offset: 1000`, tek tohum, 100 görev); sonuçlar `reports/pmnist_dev_hparams.md`, seçilen değerler `suites/selected_pmnist.yaml`.
* Analiz: `scripts/analyze.py`, `scripts/scaleup_tables.py`, `scripts/equivalence.py`, `scripts/plasticity_criterion.py`, `scripts/denetim_paketi.py`; istatistik fonksiyonlarının birim testleri `tests/test_stats.py`.

