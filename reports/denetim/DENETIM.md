# İstatistiksel denetim paketi

Danışman raporu (16 Kasım 2026), teslim listesi madde 3. Bu klasördeki her sayı, koşuların `summary.json` dosyalarından `python3 scripts/denetim_paketi.py` komutuyla yeniden üretilir; başka rapordan kopyalanmış sayı yoktur. Üretildiği depo sürümü: `881bb9b`.

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
| sin_refl_matched | 10 | 0.9223 ± 0.0004 | 0.9242 ± 0.0011 | 0.9213 ± 0.0011 | 0.9969 ± 0.0017 | -0.0002 ± 0.0002 | -0.0021 ± 0.0047 | 10/10 |
| tanh | 10 | 0.8590 ± 0.0012 | 0.9111 ± 0.0003 | 0.8305 ± 0.0034 | 0.9115 ± 0.0038 | -0.0101 ± 0.0004 | +0.0853 ± 0.0039 | 0/10 |
| tri | 10 | 0.9389 ± 0.0005 | 0.9345 ± 0.0008 | 0.9396 ± 0.0009 | 1.0054 ± 0.0012 | +0.0004 ± 0.0002 | -0.0234 ± 0.0021 | 10/10 |
| weight_clipping | 10 | 0.9386 ± 0.0003 | 0.9343 ± 0.0011 | 0.9396 ± 0.0007 | 1.0058 ± 0.0012 | +0.0005 ± 0.0001 | -0.0222 ± 0.0035 | 10/10 |

### pmnist_main (200 görev × 5.000 örnek)

| Model | n | AUC | Erken pencere | Son pencere | Koruma oranı | Eğim / 100 görev | Taze model farkı | Üç ölçüt |
|---|---:|---|---|---|---|---|---|---|
| baseline | 10 | 0.7743 ± 0.0012 | 0.8207 ± 0.0020 | 0.7432 ± 0.0044 | 0.9057 ± 0.0054 | -0.0407 ± 0.0030 | +0.0212 ± 0.0114 | 0/10 |
| continual_backprop | 10 | 0.8210 ± 0.0009 | 0.8189 ± 0.0016 | 0.8218 ± 0.0028 | 1.0035 ± 0.0045 | +0.0009 ± 0.0014 | -0.0553 ± 0.0094 | 9/10 |
| l2_init | 10 | 0.8305 ± 0.0005 | 0.8257 ± 0.0015 | 0.8313 ± 0.0021 | 1.0068 ± 0.0033 | +0.0017 ± 0.0011 | -0.0704 ± 0.0061 | 10/10 |
| ln_wd | 10 | 0.8164 ± 0.0011 | 0.8291 ± 0.0016 | 0.8094 ± 0.0033 | 0.9763 ± 0.0046 | -0.0101 ± 0.0017 | +0.0045 ± 0.0069 | 0/10 |
| nap | 10 | 0.8325 ± 0.0008 | 0.8289 ± 0.0022 | 0.8329 ± 0.0020 | 1.0048 ± 0.0041 | +0.0008 ± 0.0010 | -0.0156 ± 0.0083 | 10/10 |
| parseval | 10 | 0.8434 ± 0.0005 | 0.8296 ± 0.0014 | 0.8442 ± 0.0018 | 1.0176 ± 0.0034 | +0.0049 ± 0.0011 | -0.1028 ± 0.0084 | 10/10 |
| shrink_perturb | 10 | 0.8093 ± 0.0004 | 0.8090 ± 0.0014 | 0.8093 ± 0.0013 | 1.0004 ± 0.0023 | +0.0000 ± 0.0008 | -0.0236 ± 0.0085 | 10/10 |
| sin | 10 | 0.7853 ± 0.0009 | 0.8208 ± 0.0017 | 0.7603 ± 0.0052 | 0.9263 ± 0.0053 | -0.0338 ± 0.0013 | -0.0077 ± 0.0127 | 0/10 |
| sin_act | 10 | 0.8093 ± 0.0009 | 0.8310 ± 0.0015 | 0.7987 ± 0.0032 | 0.9612 ± 0.0042 | -0.0174 ± 0.0015 | +0.0141 ± 0.0082 | 0/10 |
| sin_floor | 5 | 0.7859 ± 0.0016 | 0.8199 ± 0.0025 | 0.7645 ± 0.0033 | 0.9324 ± 0.0034 | -0.0313 ± 0.0023 | +0.0106 ± 0.0119 | 0/5 |
| sin_refl | 10 | 0.8258 ± 0.0015 | 0.8289 ± 0.0015 | 0.8250 ± 0.0043 | 0.9953 ± 0.0060 | -0.0012 ± 0.0025 | -0.0257 ± 0.0088 | 6/10 |
| smooth_leaky | 10 | 0.7991 ± 0.0004 | 0.8129 ± 0.0020 | 0.7829 ± 0.0022 | 0.9631 ± 0.0034 | -0.0192 ± 0.0009 | -0.0586 ± 0.0108 | 0/10 |
| tanh | 10 | 0.7852 ± 0.0012 | 0.8178 ± 0.0026 | 0.7611 ± 0.0033 | 0.9306 ± 0.0050 | -0.0326 ± 0.0020 | -0.0192 ± 0.0168 | 0/10 |
| tanh_refl | 5 | 0.7994 ± 0.0017 | 0.8305 ± 0.0019 | 0.7798 ± 0.0051 | 0.9390 ± 0.0052 | -0.0264 ± 0.0017 | +0.0144 ± 0.0127 | 0/5 |
| tri | 10 | 0.8358 ± 0.0026 | 0.8256 ± 0.0020 | 0.8395 ± 0.0033 | 1.0168 ± 0.0043 | +0.0051 ± 0.0022 | -0.0800 ± 0.0103 | 10/10 |
| upgd | 10 | 0.8001 ± 0.0015 | 0.8255 ± 0.0015 | 0.7857 ± 0.0033 | 0.9518 ± 0.0040 | -0.0205 ± 0.0018 | +0.0063 ± 0.0109 | 0/10 |
| weight_clipping | 10 | 0.8365 ± 0.0009 | 0.8258 ± 0.0019 | 0.8397 ± 0.0027 | 1.0167 ± 0.0036 | +0.0059 ± 0.0014 | -0.0813 ± 0.0090 | 10/10 |

### pmnist_adam10 (200 görev × 5.000 örnek, Adam (lr 10⁻³; ayarlar SGD'den aktarıldı))

| Model | n | AUC | Erken pencere | Son pencere | Koruma oranı | Eğim / 100 görev | Taze model farkı | Üç ölçüt |
|---|---:|---|---|---|---|---|---|---|
| adam_sin | 10 | 0.8256 ± 0.0018 | 0.8285 ± 0.0028 | 0.8264 ± 0.0034 | 0.9975 ± 0.0049 | -0.0003 ± 0.0021 | -0.0023 ± 0.0101 | 6/10 |
| adam_std | 10 | 0.7413 ± 0.0071 | 0.8063 ± 0.0028 | 0.7059 ± 0.0071 | 0.8754 ± 0.0105 | -0.0532 ± 0.0063 | +0.1215 ± 0.0158 | 0/10 |
| adam_tanh | 10 | 0.8110 ± 0.0020 | 0.8286 ± 0.0021 | 0.8049 ± 0.0051 | 0.9714 ± 0.0047 | -0.0109 ± 0.0027 | +0.0274 ± 0.0112 | 0/10 |
| adam_tri | 10 | 0.7916 ± 0.0050 | 0.8060 ± 0.0063 | 0.7860 ± 0.0128 | 0.9753 ± 0.0176 | -0.0080 ± 0.0047 | +0.0281 ± 0.0374 | 0/10 |
| adam_weight_clipping | 10 | 0.7914 ± 0.0064 | 0.8077 ± 0.0051 | 0.7841 ± 0.0232 | 0.9707 ± 0.0289 | -0.0109 ± 0.0101 | +0.0383 ± 0.0537 | 1/10 |

### pmnist_inputnorm (200 görev × 5.000 örnek, standartlaştırılmış girdi (ayarlar aktarıldı))

| Model | n | AUC | Erken pencere | Son pencere | Koruma oranı | Eğim / 100 görev | Taze model farkı | Üç ölçüt |
|---|---:|---|---|---|---|---|---|---|
| adam_sin | 5 | 0.8224 ± 0.0007 | 0.8157 ± 0.0011 | 0.8230 ± 0.0014 | 1.0090 ± 0.0023 | +0.0027 ± 0.0015 | -0.0162 ± 0.0078 | 5/5 |
| adam_std | 5 | 0.7284 ± 0.0120 | 0.7719 ± 0.0091 | 0.6958 ± 0.0211 | 0.9016 ± 0.0304 | -0.0312 ± 0.0125 | +0.1246 ± 0.0310 | 0/5 |
| adam_tanh | 5 | 0.7846 ± 0.0035 | 0.8105 ± 0.0015 | 0.7751 ± 0.0131 | 0.9563 ± 0.0150 | -0.0191 ± 0.0060 | +0.0426 ± 0.0281 | 0/5 |
| adam_tri | 5 | 0.8136 ± 0.0007 | 0.8062 ± 0.0013 | 0.8163 ± 0.0022 | 1.0126 ± 0.0036 | +0.0043 ± 0.0010 | -0.0214 ± 0.0034 | 5/5 |
| adam_weight_clipping | 5 | 0.8149 ± 0.0015 | 0.8065 ± 0.0011 | 0.8171 ± 0.0028 | 1.0131 ± 0.0033 | +0.0043 ± 0.0009 | -0.0188 ± 0.0093 | 5/5 |
| sgd_std | 5 | 0.7814 ± 0.0013 | 0.8283 ± 0.0010 | 0.7525 ± 0.0019 | 0.9085 ± 0.0016 | -0.0401 ± 0.0015 | +0.0809 ± 0.0184 | 0/5 |
| sgd_tri | 5 | 0.8359 ± 0.0007 | 0.8359 ± 0.0017 | 0.8352 ± 0.0012 | 0.9992 ± 0.0019 | -0.0005 ± 0.0008 | -0.0161 ± 0.0067 | 5/5 |
| sgd_weight_clipping | 5 | 0.8357 ± 0.0004 | 0.8369 ± 0.0007 | 0.8342 ± 0.0012 | 0.9968 ± 0.0017 | -0.0010 ± 0.0008 | -0.0072 ± 0.0036 | 4/5 |

### rlmnist_main (Random Label MNIST, 50 görev × 400 epok (L2 Init: results/rlmnist_ctrl_main))

| Model | n | AUC | Erken pencere | Son pencere | Koruma oranı | Eğim / 100 görev | Taze model farkı | Üç ölçüt |
|---|---:|---|---|---|---|---|---|---|
| baseline | 10 | 0.2535 ± 0.0524 | 0.6175 ± 0.0526 | 0.1514 ± 0.0562 | 0.2440 ± 0.0814 | -0.9714 ± 0.1141 | +0.2464 ± 0.0616 | 0/10 |
| l2_init | 10 | 0.9508 ± 0.0007 | 0.9475 ± 0.0018 | 0.9520 ± 0.0012 | 1.0047 ± 0.0016 | +0.0111 ± 0.0040 | -0.0545 ± 0.0055 | 10/10 |
| sin | 10 | 0.2864 ± 0.0260 | 0.6104 ± 0.0258 | 0.1613 ± 0.0232 | 0.2644 ± 0.0377 | -1.0064 ± 0.0787 | +0.2167 ± 0.0256 | 0/10 |
| tanh | 10 | 0.2924 ± 0.0185 | 0.5094 ± 0.0184 | 0.1677 ± 0.0082 | 0.3297 ± 0.0203 | -0.8557 ± 0.0599 | +0.1442 ± 0.0185 | 0/10 |
| tri | 10 | 0.3665 ± 0.0274 | 0.7197 ± 0.0107 | 0.1223 ± 0.0234 | 0.1698 ± 0.0315 | -1.4666 ± 0.0699 | +0.8188 ± 0.0242 | 0/10 |
| weight_clipping | 10 | 0.3070 ± 0.0311 | 0.6153 ± 0.0310 | 0.1738 ± 0.0391 | 0.2819 ± 0.0592 | -1.0243 ± 0.1482 | +0.2236 ± 0.0407 | 0/10 |

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

### pmnist_fullstream: sin_refl_matched − sin, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.1045 | 0.0030 | 0.00095 | 115.12 | 7.1e-16 | 104.61 | 1 | 1 | 1 | [+0.1028, +0.1062] | farklı (GA marjın dışında) |
| Son pencere | +0.2010 | 0.0165 | 0.00520 | 39.58 | 1e-11 | 37.66 | 1 | 1 | 1 | [+0.1914, +0.2105] | farklı (GA marjın dışında) |
| Koruma oranı | +0.2046 | 0.0179 | 0.00567 | 36.97 | 1.9e-11 | 35.21 | 1 | 1 | 1 | [+0.1942, +0.2150] | farklı (GA marjın dışında) |
| Taze model farkı | -0.2139 | 0.0220 | 0.00697 | -29.97 | 1 | -31.40 | 8.3e-11 | 1 | 1 | [-0.2267, -0.2011] | farklı (GA marjın dışında) |

### pmnist_fullstream: sin_refl_matched − sin, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.1045 | 0.0030 | 0.00095 | 120.38 | 4.8e-16 | 99.35 | 1 | 1 | 1 | [+0.1028, +0.1062] | farklı (GA marjın dışında) |
| Son pencere | +0.2010 | 0.0165 | 0.00520 | 40.54 | 8.4e-12 | 36.70 | 1 | 1 | 1 | [+0.1914, +0.2105] | farklı (GA marjın dışında) |
| Koruma oranı | +0.2046 | 0.0179 | 0.00567 | 37.85 | 1.6e-11 | 34.32 | 1 | 1 | 1 | [+0.1942, +0.2150] | farklı (GA marjın dışında) |
| Taze model farkı | -0.2139 | 0.0220 | 0.00697 | -29.25 | 1 | -32.12 | 6.8e-11 | 1 | 1 | [-0.2267, -0.2011] | farklı (GA marjın dışında) |

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

### pmnist_main: tri − weight_clipping, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | -0.0007 | 0.0022 | 0.00070 | 6.12 | 8.8e-05 | -8.12 | 9.8e-06 | 8.8e-05 | 0.00035 | [-0.0020, +0.0006] | eşdeğer |
| Son pencere | -0.0002 | 0.0025 | 0.00080 | 6.03 | 9.8e-05 | -6.53 | 5.4e-05 | 9.8e-05 | 0.00035 | [-0.0017, +0.0013] | eşdeğer |
| Koruma oranı | +0.0001 | 0.0034 | 0.00107 | 4.73 | 0.00054 | -4.59 | 0.00066 | 0.00066 | 0.0013 | [-0.0019, +0.0020] | eşdeğer |
| Taze model farkı | +0.0013 | 0.0063 | 0.00199 | 3.17 | 0.0057 | -1.86 | 0.048 | 0.048 | 0.048 | [-0.0023, +0.0049] | eşdeğer |

### pmnist_main: tri − weight_clipping, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | -0.0007 | 0.0022 | 0.00070 | 13.24 | 1.7e-07 | -15.25 | 4.9e-08 | 1.7e-07 | 6.7e-07 | [-0.0020, +0.0006] | eşdeğer |
| Son pencere | -0.0002 | 0.0025 | 0.00080 | 12.30 | 3.1e-07 | -12.80 | 2.2e-07 | 3.1e-07 | 9.3e-07 | [-0.0017, +0.0013] | eşdeğer |
| Koruma oranı | +0.0001 | 0.0034 | 0.00107 | 9.39 | 3e-06 | -9.25 | 3.4e-06 | 3.4e-06 | 6.8e-06 | [-0.0019, +0.0020] | eşdeğer |
| Taze model farkı | +0.0013 | 0.0063 | 0.00199 | 5.68 | 0.00015 | -4.38 | 0.00089 | 0.00089 | 0.00089 | [-0.0023, +0.0049] | eşdeğer |

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

### pmnist_adam10: adam_tri − adam_weight_clipping, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0001 | 0.0089 | 0.00282 | 1.82 | 0.051 | -1.73 | 0.059 | 0.059 | 0.24 | [-0.0050, +0.0053] | karar verilemez |
| Son pencere | +0.0019 | 0.0235 | 0.00742 | 0.94 | 0.19 | -0.41 | 0.35 | 0.35 | 1 | [-0.0117, +0.0156] | karar verilemez |
| Koruma oranı | +0.0046 | 0.0319 | 0.01009 | 0.95 | 0.18 | -0.04 | 0.48 | 0.48 | 1 | [-0.0139, +0.0231] | karar verilemez |
| Taze model farkı | -0.0101 | 0.0444 | 0.01404 | -0.37 | 0.64 | -1.08 | 0.15 | 0.64 | 1 | [-0.0359, +0.0156] | karar verilemez |

### pmnist_adam10: adam_tri − adam_weight_clipping, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0001 | 0.0089 | 0.00282 | 3.59 | 0.0029 | -3.50 | 0.0034 | 0.0034 | 0.014 | [-0.0050, +0.0053] | eşdeğer |
| Son pencere | +0.0019 | 0.0235 | 0.00742 | 1.61 | 0.071 | -1.08 | 0.15 | 0.15 | 0.46 | [-0.0117, +0.0156] | karar verilemez |
| Koruma oranı | +0.0046 | 0.0319 | 0.01009 | 1.44 | 0.092 | -0.54 | 0.3 | 0.3 | 0.6 | [-0.0139, +0.0231] | karar verilemez |
| Taze model farkı | -0.0101 | 0.0444 | 0.01404 | -0.01 | 0.5 | -1.43 | 0.093 | 0.5 | 0.6 | [-0.0359, +0.0156] | karar verilemez |

### pmnist_adam10: adam_sin − adam_weight_clipping, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0342 | 0.0065 | 0.00206 | 19.05 | 7e-09 | 14.18 | 1 | 1 | 1 | [+0.0304, +0.0379] | farklı (GA marjın dışında) |
| Son pencere | +0.0423 | 0.0225 | 0.00713 | 6.64 | 4.7e-05 | 5.24 | 1 | 1 | 1 | [+0.0293, +0.0554] | farklı (GA marjın dışında) |
| Koruma oranı | +0.0268 | 0.0295 | 0.00932 | 3.41 | 0.0039 | 2.34 | 0.98 | 0.98 | 1 | [+0.0097, +0.0438] | farklı (GA marjın dışında) |
| Taze model farkı | -0.0406 | 0.0517 | 0.01634 | -2.18 | 0.97 | -2.79 | 0.011 | 0.97 | 1 | [-0.0705, -0.0106] | farklı (GA marjın dışında) |

### pmnist_adam10: adam_sin − adam_weight_clipping, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0342 | 0.0065 | 0.00206 | 21.48 | 2.4e-09 | 11.75 | 1 | 1 | 1 | [+0.0304, +0.0379] | farklı (GA marjın dışında) |
| Son pencere | +0.0423 | 0.0225 | 0.00713 | 7.34 | 2.2e-05 | 4.53 | 1 | 1 | 1 | [+0.0293, +0.0554] | farklı (GA marjın dışında) |
| Koruma oranı | +0.0268 | 0.0295 | 0.00932 | 3.95 | 0.0017 | 1.80 | 0.95 | 0.95 | 1 | [+0.0097, +0.0438] | karar verilemez |
| Taze model farkı | -0.0406 | 0.0517 | 0.01634 | -1.87 | 0.95 | -3.09 | 0.0064 | 0.95 | 1 | [-0.0705, -0.0106] | farklı (GA marjın dışında) |

### pmnist_adam10: adam_sin − adam_tri, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0340 | 0.0045 | 0.00143 | 27.22 | 3e-10 | 20.24 | 1 | 1 | 1 | [+0.0314, +0.0366] | farklı (GA marjın dışında) |
| Son pencere | +0.0404 | 0.0134 | 0.00425 | 10.68 | 1e-06 | 8.33 | 1 | 1 | 1 | [+0.0326, +0.0482] | farklı (GA marjın dışında) |
| Koruma oranı | +0.0222 | 0.0191 | 0.00605 | 4.49 | 0.00075 | 2.84 | 0.99 | 0.99 | 1 | [+0.0111, +0.0333] | farklı (GA marjın dışında) |
| Taze model farkı | -0.0304 | 0.0338 | 0.01069 | -2.38 | 0.98 | -3.31 | 0.0045 | 0.98 | 1 | [-0.0500, -0.0108] | farklı (GA marjın dışında) |

### pmnist_adam10: adam_sin − adam_tri, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0340 | 0.0045 | 0.00143 | 30.70 | 1e-10 | 16.75 | 1 | 1 | 1 | [+0.0314, +0.0366] | farklı (GA marjın dışında) |
| Son pencere | +0.0404 | 0.0134 | 0.00425 | 11.86 | 4.2e-07 | 7.15 | 1 | 1 | 1 | [+0.0326, +0.0482] | farklı (GA marjın dışında) |
| Koruma oranı | +0.0222 | 0.0191 | 0.00605 | 5.32 | 0.00024 | 2.02 | 0.96 | 0.96 | 1 | [+0.0111, +0.0333] | farklı (GA marjın dışında) |
| Taze model farkı | -0.0304 | 0.0338 | 0.01069 | -1.91 | 0.96 | -3.78 | 0.0022 | 0.96 | 1 | [-0.0500, -0.0108] | farklı (GA marjın dışında) |

### pmnist_inputnorm: adam_tri − adam_weight_clipping, Δ = 0.005 (n = 5 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | -0.0013 | 0.0012 | 0.00054 | 6.77 | 0.0012 | -11.67 | 0.00015 | 0.0012 | 0.005 | [-0.0025, -0.0002] | eşdeğer |
| Son pencere | -0.0008 | 0.0020 | 0.00089 | 4.77 | 0.0044 | -6.48 | 0.0015 | 0.0044 | 0.013 | [-0.0027, +0.0011] | eşdeğer |
| Koruma oranı | -0.0005 | 0.0026 | 0.00115 | 3.93 | 0.0086 | -4.80 | 0.0043 | 0.0086 | 0.017 | [-0.0029, +0.0019] | eşdeğer |
| Taze model farkı | -0.0026 | 0.0060 | 0.00267 | 0.88 | 0.21 | -2.86 | 0.023 | 0.21 | 0.21 | [-0.0083, +0.0031] | karar verilemez |

### pmnist_inputnorm: adam_tri − adam_weight_clipping, Δ = 0.01 (n = 5 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | -0.0013 | 0.0012 | 0.00054 | 15.99 | 4.5e-05 | -20.88 | 1.6e-05 | 4.5e-05 | 0.00018 | [-0.0025, -0.0002] | eşdeğer |
| Son pencere | -0.0008 | 0.0020 | 0.00089 | 10.40 | 0.00024 | -12.11 | 0.00013 | 0.00024 | 0.00073 | [-0.0027, +0.0011] | eşdeğer |
| Koruma oranı | -0.0005 | 0.0026 | 0.00115 | 8.29 | 0.00058 | -9.16 | 0.00039 | 0.00058 | 0.0012 | [-0.0029, +0.0019] | eşdeğer |
| Taze model farkı | -0.0026 | 0.0060 | 0.00267 | 2.75 | 0.026 | -4.73 | 0.0046 | 0.026 | 0.026 | [-0.0083, +0.0031] | eşdeğer |

### pmnist_inputnorm: adam_sin − adam_weight_clipping, Δ = 0.005 (n = 5 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0075 | 0.0019 | 0.00087 | 14.41 | 6.7e-05 | 2.88 | 0.98 | 0.98 | 1 | [+0.0056, +0.0093] | farklı (GA marjın dışında) |
| Son pencere | +0.0060 | 0.0036 | 0.00159 | 6.89 | 0.0012 | 0.60 | 0.71 | 0.71 | 1 | [+0.0026, +0.0094] | karar verilemez |
| Koruma oranı | -0.0041 | 0.0043 | 0.00192 | 0.48 | 0.33 | -4.73 | 0.0046 | 0.33 | 1 | [-0.0082, +0.0000] | karar verilemez |
| Taze model farkı | +0.0026 | 0.0083 | 0.00371 | 2.05 | 0.055 | -0.65 | 0.28 | 0.28 | 1 | [-0.0053, +0.0105] | karar verilemez |

### pmnist_inputnorm: adam_sin − adam_weight_clipping, Δ = 0.01 (n = 5 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0075 | 0.0019 | 0.00087 | 20.18 | 1.8e-05 | -2.89 | 0.022 | 0.022 | 0.073 | [+0.0056, +0.0093] | eşdeğer |
| Son pencere | +0.0060 | 0.0036 | 0.00159 | 10.03 | 0.00028 | -2.54 | 0.032 | 0.032 | 0.073 | [+0.0026, +0.0094] | eşdeğer |
| Koruma oranı | -0.0041 | 0.0043 | 0.00192 | 3.09 | 0.018 | -7.33 | 0.00092 | 0.018 | 0.073 | [-0.0082, +0.0000] | eşdeğer |
| Taze model farkı | +0.0026 | 0.0083 | 0.00371 | 3.39 | 0.014 | -1.99 | 0.058 | 0.058 | 0.073 | [-0.0053, +0.0105] | karar verilemez |

### pmnist_inputnorm: adam_sin − adam_tri, Δ = 0.005 (n = 5 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0088 | 0.0010 | 0.00044 | 31.09 | 3.2e-06 | 8.60 | 1 | 1 | 1 | [+0.0079, +0.0098] | farklı (GA marjın dışında) |
| Son pencere | +0.0067 | 0.0022 | 0.00100 | 11.71 | 0.00015 | 1.72 | 0.92 | 0.92 | 1 | [+0.0046, +0.0089] | karar verilemez |
| Koruma oranı | -0.0036 | 0.0031 | 0.00141 | 1.01 | 0.18 | -6.09 | 0.0018 | 0.18 | 0.74 | [-0.0066, -0.0006] | karar verilemez |
| Taze model farkı | +0.0052 | 0.0068 | 0.00306 | 3.35 | 0.014 | 0.08 | 0.53 | 0.53 | 1 | [-0.0013, +0.0118] | karar verilemez |

### pmnist_inputnorm: adam_sin − adam_tri, Δ = 0.01 (n = 5 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0088 | 0.0010 | 0.00044 | 42.34 | 9.3e-07 | -2.64 | 0.029 | 0.029 | 0.057 | [+0.0079, +0.0098] | eşdeğer |
| Son pencere | +0.0067 | 0.0022 | 0.00100 | 16.70 | 3.8e-05 | -3.28 | 0.015 | 0.015 | 0.046 | [+0.0046, +0.0089] | eşdeğer |
| Koruma oranı | -0.0036 | 0.0031 | 0.00141 | 4.57 | 0.0051 | -9.64 | 0.00032 | 0.0051 | 0.021 | [-0.0066, -0.0006] | eşdeğer |
| Taze model farkı | +0.0052 | 0.0068 | 0.00306 | 4.98 | 0.0038 | -1.56 | 0.097 | 0.097 | 0.097 | [-0.0013, +0.0118] | karar verilemez |

### pmnist_inputnorm: sgd_tri − sgd_weight_clipping, Δ = 0.005 (n = 5 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0003 | 0.0006 | 0.00028 | 18.52 | 2.5e-05 | -16.70 | 3.8e-05 | 3.8e-05 | 0.00015 | [-0.0003, +0.0009] | eşdeğer |
| Son pencere | +0.0010 | 0.0018 | 0.00079 | 7.62 | 0.0008 | -5.06 | 0.0036 | 0.0036 | 0.011 | [-0.0007, +0.0027] | eşdeğer |
| Koruma oranı | +0.0024 | 0.0024 | 0.00107 | 6.95 | 0.0011 | -2.42 | 0.036 | 0.036 | 0.073 | [+0.0001, +0.0047] | eşdeğer |
| Taze model farkı | -0.0088 | 0.0077 | 0.00344 | -1.11 | 0.84 | -4.02 | 0.0079 | 0.84 | 0.84 | [-0.0162, -0.0015] | karar verilemez |

### pmnist_inputnorm: sgd_tri − sgd_weight_clipping, Δ = 0.01 (n = 5 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0003 | 0.0006 | 0.00028 | 36.13 | 1.8e-06 | -34.31 | 2.2e-06 | 2.2e-06 | 8.6e-06 | [-0.0003, +0.0009] | eşdeğer |
| Son pencere | +0.0010 | 0.0018 | 0.00079 | 13.96 | 7.6e-05 | -11.40 | 0.00017 | 0.00017 | 0.00051 | [-0.0007, +0.0027] | eşdeğer |
| Koruma oranı | +0.0024 | 0.0024 | 0.00107 | 11.64 | 0.00016 | -7.11 | 0.001 | 0.001 | 0.0021 | [+0.0001, +0.0047] | eşdeğer |
| Taze model farkı | -0.0088 | 0.0077 | 0.00344 | 0.34 | 0.38 | -5.47 | 0.0027 | 0.38 | 0.38 | [-0.0162, -0.0015] | karar verilemez |

### rlmnist_main: l2_init − tri, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.5843 | 0.0278 | 0.00878 | 67.14 | 9.1e-14 | 66.00 | 1 | 1 | 1 | [+0.5682, +0.6003] | farklı (GA marjın dışında) |
| Son pencere | +0.8297 | 0.0236 | 0.00747 | 111.75 | 9.3e-16 | 110.41 | 1 | 1 | 1 | [+0.8160, +0.8434] | farklı (GA marjın dışında) |
| Koruma oranı | +0.8349 | 0.0321 | 0.01016 | 82.63 | 1.4e-14 | 81.65 | 1 | 1 | 1 | [+0.8163, +0.8536] | farklı (GA marjın dışında) |
| Taze model farkı | -0.8732 | 0.0231 | 0.00729 | -119.06 | 1 | -120.43 | 4.8e-16 | 1 | 1 | [-0.8866, -0.8599] | farklı (GA marjın dışında) |

### rlmnist_main: l2_init − tri, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.5843 | 0.0278 | 0.00878 | 67.71 | 8.4e-14 | 65.43 | 1 | 1 | 1 | [+0.5682, +0.6003] | farklı (GA marjın dışında) |
| Son pencere | +0.8297 | 0.0236 | 0.00747 | 112.42 | 8.9e-16 | 109.74 | 1 | 1 | 1 | [+0.8160, +0.8434] | farklı (GA marjın dışında) |
| Koruma oranı | +0.8349 | 0.0321 | 0.01016 | 83.12 | 1.3e-14 | 81.16 | 1 | 1 | 1 | [+0.8163, +0.8536] | farklı (GA marjın dışında) |
| Taze model farkı | -0.8732 | 0.0231 | 0.00729 | -118.37 | 1 | -121.12 | 4.5e-16 | 1 | 1 | [-0.8866, -0.8599] | farklı (GA marjın dışında) |

### rlmnist_main: l2_init − baseline, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.6973 | 0.0520 | 0.01645 | 42.69 | 5.3e-12 | 42.08 | 1 | 1 | 1 | [+0.6671, +0.7274] | farklı (GA marjın dışında) |
| Son pencere | +0.8005 | 0.0558 | 0.01766 | 45.62 | 2.9e-12 | 45.05 | 1 | 1 | 1 | [+0.7682, +0.8329] | farklı (GA marjın dışında) |
| Koruma oranı | +0.7607 | 0.0817 | 0.02584 | 29.63 | 1.4e-10 | 29.25 | 1 | 1 | 1 | [+0.7134, +0.8081] | farklı (GA marjın dışında) |
| Taze model farkı | -0.3008 | 0.0599 | 0.01894 | -15.62 | 1 | -16.15 | 3e-08 | 1 | 1 | [-0.3355, -0.2661] | farklı (GA marjın dışında) |

### rlmnist_main: l2_init − baseline, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.6973 | 0.0520 | 0.01645 | 42.99 | 5e-12 | 41.78 | 1 | 1 | 1 | [+0.6671, +0.7274] | farklı (GA marjın dışında) |
| Son pencere | +0.8005 | 0.0558 | 0.01766 | 45.90 | 2.8e-12 | 44.77 | 1 | 1 | 1 | [+0.7682, +0.8329] | farklı (GA marjın dışında) |
| Koruma oranı | +0.7607 | 0.0817 | 0.02584 | 29.83 | 1.3e-10 | 29.05 | 1 | 1 | 1 | [+0.7134, +0.8081] | farklı (GA marjın dışında) |
| Taze model farkı | -0.3008 | 0.0599 | 0.01894 | -15.36 | 1 | -16.42 | 2.6e-08 | 1 | 1 | [-0.3355, -0.2661] | farklı (GA marjın dışında) |

### rlmnist_main: tri − baseline, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.1130 | 0.0585 | 0.01850 | 6.38 | 6.4e-05 | 5.84 | 1 | 1 | 1 | [+0.0791, +0.1470] | farklı (GA marjın dışında) |
| Son pencere | -0.0291 | 0.0405 | 0.01279 | -1.89 | 0.95 | -2.67 | 0.013 | 0.95 | 1 | [-0.0526, -0.0057] | farklı (GA marjın dışında) |
| Koruma oranı | -0.0742 | 0.0600 | 0.01899 | -3.64 | 1 | -4.17 | 0.0012 | 1 | 1 | [-0.1090, -0.0394] | farklı (GA marjın dışında) |
| Taze model farkı | +0.5724 | 0.0462 | 0.01460 | 39.56 | 1e-11 | 38.87 | 1 | 1 | 1 | [+0.5456, +0.5991] | farklı (GA marjın dışında) |

### rlmnist_main: tri − baseline, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.1130 | 0.0585 | 0.01850 | 6.65 | 4.7e-05 | 5.57 | 1 | 1 | 1 | [+0.0791, +0.1470] | farklı (GA marjın dışında) |
| Son pencere | -0.0291 | 0.0405 | 0.01279 | -1.50 | 0.92 | -3.06 | 0.0068 | 0.92 | 1 | [-0.0526, -0.0057] | karar verilemez |
| Koruma oranı | -0.0742 | 0.0600 | 0.01899 | -3.38 | 1 | -4.43 | 0.00082 | 1 | 1 | [-0.1090, -0.0394] | farklı (GA marjın dışında) |
| Taze model farkı | +0.5724 | 0.0462 | 0.01460 | 39.90 | 9.7e-12 | 38.53 | 1 | 1 | 1 | [+0.5456, +0.5991] | farklı (GA marjın dışında) |

### rlmnist_main: tri − weight_clipping, Δ = 0.005 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0595 | 0.0240 | 0.00760 | 8.49 | 6.9e-06 | 7.17 | 1 | 1 | 1 | [+0.0456, +0.0735] | farklı (GA marjın dışında) |
| Son pencere | -0.0515 | 0.0306 | 0.00966 | -4.82 | 1 | -5.85 | 0.00012 | 1 | 1 | [-0.0692, -0.0338] | farklı (GA marjın dışında) |
| Koruma oranı | -0.1121 | 0.0476 | 0.01504 | -7.12 | 1 | -7.79 | 1.4e-05 | 1 | 1 | [-0.1397, -0.0845] | farklı (GA marjın dışında) |
| Taze model farkı | +0.5951 | 0.0289 | 0.00915 | 65.58 | 1.1e-13 | 64.49 | 1 | 1 | 1 | [+0.5784, +0.6119] | farklı (GA marjın dışında) |

### rlmnist_main: tri − weight_clipping, Δ = 0.01 (n = 10 ortak tohum)

| Ölçüt | Ortalama fark | SS | SH | t_alt | p_alt | t_üst | p_üst | p_TOST | Holm p | %90 GA | Karar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC (birincil) | +0.0595 | 0.0240 | 0.00760 | 9.15 | 3.7e-06 | 6.52 | 1 | 1 | 1 | [+0.0456, +0.0735] | farklı (GA marjın dışında) |
| Son pencere | -0.0515 | 0.0306 | 0.00966 | -4.30 | 1 | -6.37 | 6.5e-05 | 1 | 1 | [-0.0692, -0.0338] | farklı (GA marjın dışında) |
| Koruma oranı | -0.1121 | 0.0476 | 0.01504 | -6.79 | 1 | -8.12 | 9.8e-06 | 1 | 1 | [-0.1397, -0.0845] | farklı (GA marjın dışında) |
| Taze model farkı | +0.5951 | 0.0289 | 0.00915 | 66.13 | 1e-13 | 63.94 | 1 | 1 | 1 | [+0.5784, +0.6119] | farklı (GA marjın dışında) |

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

