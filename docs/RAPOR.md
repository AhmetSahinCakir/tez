# Proje Aşaması Raporu — Sınırlı-Periyodik Ağırlık Yeniden Parametrizasyonu ile Plastisite Kaybı

> Bu rapor tez öneri formundaki İş Paketleri 1–5'in proje aşamasını kapsar: deney altyapısının kurulması,
> plastisite kaybı bulgularının pilot replikasyonu, karşılaştırma yöntemleri kütüphanesi, önerilen
> A·sin(Θ) / A·tanh(Θ) yeniden parametrizasyonları, bileşen deneyleri, mekanizma analizleri ve istatistiksel
> karşılaştırma. Bütün sayılar bu depodaki `results/` çalışmalarından `scripts/analyze.py` ile üretilmiştir.

<!-- İÇİNDEKİLER: 1 Özet · 2 Altyapı · 3 Hesaplama bütçesi ve protokol ölçekleri · 4 Pilot replikasyon ·
     5 Hiperparametre seçimi · 6 Ana karşılaştırma (PMNIST) · 7 CIFAR-100 · 8 Bileşen analizi ·
     9 Mekanizma analizi (H3) · 10 Durağan kontrol · 11 İkincil deneyler · 12 Sınırlılıklar ve sonraki adımlar -->

## 1. Özet
* **Altyapı** (İP-1/2): PyTorch tabanlı, yapılandırma dosyalarıyla sürülen, tohum/görev akışı deterministik,
  305 birim testli bir deney altyapısı kurulmuş; 10 karşılaştırma yöntemi, mekanizma ölçütleri, eşleştirilmiş
  istatistik, paket çalıştırıcı ve rapor üretimi tamamlanmıştır. Proje aşamasında 16 pilot, 142 geliştirme (hiperparametre), 70 + 69 PMNIST, 40 + 24 CIFAR-100 ve 90
  durağan kontrol olmak üzere 451 çalışma yürütülmüştür (≈ 95 CPU-saati, 4 çekirdek, ≈ 24 saat duvar süresi);
  hiçbir çalışma başarısız olmamıştır.
* **Pilot replikasyon** (İP-1): Dohare vd. (2024)'ün plastisite kaybı imzaları (çevrimiçi doğruluğun azalması,
  ölü birim ve ağırlık büyümesi, kerte çöküşü, taze modelle açılan fark) küçültülmüş Online Permuted MNIST
  protokolünde yeniden üretilmiştir.
* **H1 (sınırlı yeniden parametrizasyon)**: A·sin(Θ) modeli standart ağa göre PMNIST'te anlamlı fakat küçük bir
  plastisite koruması sağlar (normalize AUC +0.011 [+0.010, +0.012], koruma oranı 0.906 → 0.926, taze model
  farkı kapanır; 10 eşleştirilmiş tohum). CIFAR-100 çevrimiçi protokolünde — ağırlıklar büyümediğinden — etki
  yoktur.
* **H2 (sınırlılık vs periyodiklik)**: A·sin(Θ) ile A·tanh(Θ) ayırt edilemez; etki sınırlılıktan kaynaklanır,
  periyodikliğin ek katkısı yoktur (aksine daha fazla doygunluk).
* **H3 (optimizasyon geometrisi)**: Sınıra yaklaşan parametrelerde cos² kaynaklı adım küçülmesi ölçülmüş
  (etkin adım %30 küçülür; Fisher/gradyan ölçümleri koordinat bağımlı), bu küçülmeyi gideren ölçek-düzeltmeli
  güncelleme plastisiteyi *bozmuştur*: Jacobian sönümü yöntemin koruyucu bileşenidir. Sert projeksiyonlu
  Weight Clipping, aynı sınırla, çok daha iyi sonuç verir (koruma 1.017).
* **Bileşenler**: etkinin büyük bölümü çıkış katmanının sınırlanmasından gelir; yanlılıkların sınırlanması
  önemsizdir; sınır genişliği γ ∈ [1.5, 2.5] en iyidir; sinüs + Continual Backprop plastisiteyi tamamen korur.
* **Yayımlanmış yöntemler** (eşit arama bütçesiyle): Weight Clipping, NaP, L2 Init, Continual Backprop,
  Shrink & Perturb ve Parseval PMNIST'te plastisiteyi tamamen korurken önerilen yöntemin önündedir; CIFAR-100'de
  yalnızca ölü birimleri hedefleyen yöntemler (CBP, L2 Init, NaP) iyileşme sağlar.
* Sonuç olarak tez, öneri formunda öngörülen "mekanizma ayrıştırması" senaryosunu gerçekleştirmiştir: önerilen
  parametrizasyon yeni bir en-iyi yöntem değildir, ancak sınırlılık–periyodiklik–eğitilebilirlik ödünleşimini
  nicel ve yeniden üretilebilir biçimde ortaya koymaktadır (ayrıntılar §6–§11).

## 2. Deney altyapısı (İP-1, İP-2)
Altyapı Python 3.11 / PyTorch ile yapılandırma dosyası tabanlı olarak geliştirilmiştir (bkz. `README.md`,
`docs/PROTOKOL.md`). Bileşenler:

* **Görev akışları** (`plasticity/data`): Online Permuted MNIST ve ardışık CIFAR-100 ikili akışları
  `(tohum, görev)` çiftinin deterministik fonksiyonudur; aynı tohumla çalışan yöntemler birebir aynı
  permütasyon/sınıf çifti/örnek sırasını görür (eşleştirilmiş tasarım). Geliştirme akışları tohum ofsetiyle
  final akışlarından ayrılır.
* **Yeniden parametrizasyon** (`plasticity/models/reparam.py`): `ReparamLinear` katmanı standart / A·sin(Θ) /
  A·tanh(Θ) efektif ağırlıkları, eşleştirilmiş başlangıcı (Θ₀ = arcsin(W₀/A) vb.), tam-kompakt yanlılığı,
  öğrenilebilir genlik seçeneğini, Jacobian ve normalize Jacobian (cos Θ) yardımcılarını ve CBP için birim
  yeniden başlatma işlemlerini içerir; ileri/geri geçiş tek bir kaynaşık `autograd.Function` ile hesaplanır
  (gradyan denetimi testlerle doğrulanmıştır).
* **Yöntem eklentileri** (`plasticity/methods`): ortak arayüz (`regularizer`, `before_step`, `after_step`,
  görev başı/sonu kancaları) üzerinde 10 yöntem. Her uygulama, kaynak makalenin formülasyonuna karşı bağımsız
  bir inceleme turundan geçirilmiş ve birim testlerle doğrulanmıştır (toplam 305 test, `tests/`).
* **Eğitici** (`plasticity/training`): çevrimiçi protokol (güncellemeden önce tahmin → çevrimiçi doğruluk),
  mini-batch/dönemli protokol (CIFAR), durağan kontrol ve seçilmiş görevlerde eşleştirilmiş başlangıçlı taze
  model referansı.
* **Ölçütler ve istatistik** (`plasticity/metrics`): mekanizma ölçütleri (ağırlık büyüklüğü/normu, ölü ve
  etkisiz birim oranı, kararlı/etkin kerte, aktivasyon ve gradyan büyüklükleri, Fisher izi ve etkin kertesi —
  hem Θ hem W koordinatlarında, F_Θ = Jᵀ F_W J uyarısıyla), özet ölçütler (normalize AUC, erken/son pencere,
  plastisite koruma oranı, taze model farkı) ve eşleştirilmiş istatistik (bootstrap GA, işaret-çevirme
  permütasyon testi, Wilcoxon, Cohen d_z, Holm düzeltmesi).
* **Çalıştırma ve analiz** (`scripts/`): yeniden başlatılabilir paralel paket çalıştırıcı, geliştirme
  akışlarında hiperparametre seçimi, tablo/şekil üretimi.

Yöntemlerin adım maliyeti (784-100-100-100-10, batch 1, tek iş parçacığı; standart = 1.0×): weight clipping
0.89×, L2 Init 0.77×, NaP 0.71×, LN+WD 0.65×, Continual Backprop 0.48×, tanh modeli 0.47×, Shrink&Perturb
0.43×, sinüs modeli 0.39×, ölçek-düzeltmeli sinüs 0.31×, Parseval 0.30×, UPGD 0.27×.

## 3. Hesaplama bütçesi ve protokol ölçekleri
Proje aşamasındaki bütün deneyler GPU'suz, 4 çekirdekli bir CPU ortamında yürütülmüştür. Ölçülen verimlilik
(tek iş parçacığı, batch 1, PyTorch eager):

| Model | Örnek/s (saf eğitim döngüsü) | Not |
|---|---|---|
| Standart MLP 784-100-100-100-10 | ≈2200 | otomatik türev + SGD |
| Sinüs MLP (aynı boyut) | ≈1050 | sin/cos ve çarpım işlemleri adım maliyetini ~2× artırır |
| Standart MLP 784-2000-2000-2000-10 (kanonik) | ≈86 | |

Kanonik Online Permuted MNIST protokolü (800 görev × 60.000 örnek = 48 M çevrimiçi güncelleme, 3×2000 birim)
tek bir tekrar için ≈155 saat CPU süresi gerektirir; 10 tohum × ~10 yöntem ile GPU'suz uygulanabilir değildir.
Bu nedenle öneri formunda öngörüldüğü gibi ("hesaplama gereksinimi pilot deneylerle önceden ölçülecektir";
"hiperparametre aramaları kısaltılmış geliştirme akışlarında") protokolün *yapısı* korunarak ölçeği küçültülmüştür:

| Protokol | Kanonik | Bu raporda | Değişmeyen özellikler |
|---|---|---|---|
| Online Permuted MNIST | 800 görev × 60.000 örnek, 3×2000 birim, lr 0.003 | 200 görev × 5.000 rastgele örnek/görev, 3×100 birim, lr yönteme göre seçilir | batch 1, tek geçiş, görev kimliği yok, çevrimiçi doğruluk, görev başına yeni permütasyon |
| Geliştirme akışı (hiperparametre) | — | 100 görev × 5.000 örnek, tohum ofseti 1000 | aynı |
| Ardışık CIFAR-100 ikili | 3000 görev | 3000 görev (kalibrasyon sonucuna göre ağ genişliği/eniyileyici, bkz. §7) | gri 1024-boyutlu girdi, 2 rastgele sınıf, test doğruluğu |

Küçültülmüş PMNIST protokolünde bir tekrar standart model için ≈12 dk, sinüs modeli için ≈23 dk sürmektedir
(4 paralel süreç). Kanonik ölçekli yapılandırmalar (`configs/pmnist_canonical.yaml`, `configs/cifar_canonical.yaml`)
aynı kodla, yalnızca yapılandırma değiştirilerek çalıştırılabilir; bu, İP-5'in GPU altyapısında yürütülmesi için
hazırdır.

## 4. Pilot replikasyon: standart ağda plastisite kaybı (İP-1)
Dohare vd. (2024) bulguları küçültülmüş protokolde tek tohumla yeniden üretilmiştir (`results/pilot/`,
standart ReLU MLP 784-100-100-100-10, SGD, batch 1, tek geçiş, 200 görev × 5.000 örnek):

| Ölçüt | lr = 0.01 | lr = 0.03 |
|---|---|---|
| İlk 25 görev ortalama çevrimiçi doğruluk | 0.796 | 0.680 |
| Son 25 görev ortalama çevrimiçi doğruluk | 0.672 | 0.221 |
| Plastisite koruma oranı (son/erken pencere, %10) | 0.844 | 0.33 |
| Ölü (hiç aktive olmayan) birim oranı, görev 0 → 199 | 0.00 → 0.32 | 0.02 → 0.88 |
| Ortalama ağırlık büyüklüğü \|W\|, görev 0 → 199 | 0.063 → 0.183 | 0.070 → 0.265 |
| Son gizli temsilin etkin kertesi (merkezlenmiş), görev 0 → 199 | 49.9 → 21.1 | 40.1 → 1.6 |
| Taze (sıfırdan) model ile fark, görev 199 | 0.803 − 0.658 = 0.145 | 0.77 − 0.34 |

Üç imza birlikte gözlenmektedir: çevrimiçi doğruluğun görev sayısıyla tekdüze azalması, ölü birim oranı ve
ağırlık büyüklüğünün artması, temsil kertesinin azalması; aynı görevi sıfırdan öğrenen eşleştirilmiş taze model
ise sabit ≈0.80 doğruluk verir, yani düşüş görevlerin zorlaşmasından değil ağın öğrenme kapasitesinin
azalmasından kaynaklanır. Daha yüksek öğrenme oranı plastisite kaybını belirgin biçimde hızlandırır (Dohare vd.
ile uyumlu). Daha kısa görevlerle (400 görev × 2.500 örnek) aynı eğilim (0.725 → 0.538) gözlenmiştir.

Aynı pilotta önerilen sinüs modeli (γ = 1.5, tam-kompakt, aynı lr = 0.01, aynı akış) 0.818 → 0.713
(koruma oranı 0.867, normalize AUC 0.758'e karşı 0.720) elde etmiş; yani plastisite kaybı azalmış fakat
ortadan kalkmamıştır. Mekanizma ölçütleri H3'ü doğrudan görünür kılmaktadır: ortalama \|W\|/A 0.35'ten
0.83'e yükselmiş, cos²Θ ortalaması (etkin adım çarpanı) 0.84'ten 0.24'e düşmüş ve parametrelerin %50'si
\|cos Θ\| < 0.1 doygunluk bölgesine girmiştir (`sat_frac`). Sınıra yığılan ağırlıkların gradyanı Jacobian
nedeniyle sönmekte ve bu birimler ReLU ölümüyle birleşince (ölü birim oranı 0.44) temsil kertesi standarttan
daha hızlı düşmektedir (47 → 13). Bu gözlem, genlik taraması ve ölçek-düzeltmeli güncelleme deneylerinin
(§11) gerekçesini oluşturur.

## 5. Hiperparametre seçimi (İP-2)
Öneri formunun gereği olarak yöntemlere ortak bir öğrenme oranı dayatılmamış; her yöntem için önceden
belirlenmiş, eşit büyüklükte bir arama bütçesi **geliştirme akışlarında** (tohum ofseti 1000; final
akışlarıyla permütasyon/sınıf çifti paylaşmaz) tek tohumla uygulanmıştır. Seçim ölçütü normalize AUC'dir.

| Yöntem | Arama ızgarası (PMNIST ve CIFAR-100 için aynı) | Yapılandırma sayısı |
|---|---|---|
| Standart, Sin-MLP, Smooth-Leaky, NaP | lr ∈ {0.001, 0.003, 0.01, 0.03, 0.1} (NaP: {0.003 … 0.3}) | 5 |
| Sinüs, tanh | lr ∈ {0.001 … 0.1} (γ = 1.5) ∪ lr ∈ {0.003, 0.01} × γ ∈ {2.5, 5} | 9 |
| Weight Clipping | lr ∈ {0.003, 0.01, 0.03, 0.1} × κ ∈ {1, 2} | 8 |
| L2 Init | lr × λ ∈ {1e-3, 1e-2} | 8 |
| Continual Backprop | lr × ρ ∈ {1e-4, 1e-3} (m = 100, η = 0.99) | 8 |
| LayerNorm + WD | lr × λ ∈ {1e-4, 1e-3} | 8 |
| Shrink & Perturb, UPGD, Parseval (ikincil küme) | lr ∈ {0.003, 0.01} × 2 yönteme özgü değer | 4 |

Geliştirme akışları: PMNIST 100 görev × 5.000 örnek; CIFAR-100 300 görev. Seçilen değerler
`suites/selected_pmnist.yaml` ve `suites/selected_cifar.yaml` dosyalarında dondurulmuş, bütün varyantların
sonuçları `reports/pmnist_dev_hparams.md` ve `reports/cifar_dev_hparams.md` tablolarında verilmiştir.
İkincil kümenin bütçesi hesaplama maliyeti (adım başına 3–4 kat yavaş) nedeniyle yarıya indirilmiştir;
bu yöntemler ana hipotez testlerinde yer almaz.

_(seçilen değerler tablosu aşağıda, geliştirme akışı sonuçlarından doldurulacak)_

## 6. Ana karşılaştırma — Online Permuted MNIST (İP-5)
Final akışlar (tohum ofseti 0), 200 görev × 5.000 örnek, batch 1, tek geçiş; standart/sinüs/tanh 10 tohum,
yayımlanmış yöntemler 5 tohum, ikincil küme 3 tohum. Aynı tohum aynı görev akışını tanımlar (eşleştirilmiş
tasarım). Bütün sayılar `reports/pmnist_summary_main.md`, `reports/pmnist_vs_baseline.md` ve
`reports/pmnist_vs_sin.md` dosyalarından alınmıştır (ortalama ± %95 GA, tohumlar üzerinden).

| Yöntem | n | Normalize AUC | Son pencere (son 20 görev) | Koruma oranı | Taze model farkı | Ölü birim | Ort. \|w\| | Etkin kerte |
|---|---:|---|---|---|---|---|---|---|
| Standart (baseline) | 10 | 0.774 ± 0.001 | 0.743 ± 0.003 | 0.906 ± 0.004 | +0.021 ± 0.008 | 0.177 | 0.109 | 25.5 |
| **Sinüs A·sin(Θ)** | 10 | 0.785 ± 0.001 | 0.760 ± 0.004 | 0.926 ± 0.004 | −0.008 ± 0.009 | 0.141 | 0.098 | 28.1 |
| Tanh A·tanh(Θ) | 10 | 0.785 ± 0.001 | 0.761 ± 0.002 | 0.931 ± 0.004 | −0.019 ± 0.012 | 0.119 | 0.093 | 31.7 |
| Weight Clipping (κ=1) | 5 | 0.836 ± 0.001 | 0.839 ± 0.002 | 1.017 ± 0.004 | −0.077 ± 0.012 | 0.145 | 0.064 | 42.9 |
| L2 Init (λ=0.01) | 5 | 0.830 ± 0.001 | 0.831 ± 0.002 | 1.007 ± 0.004 | −0.067 ± 0.006 | 0.013 | 0.061 | 45.8 |
| Continual Backprop (ρ=1e-4) | 5 | 0.821 ± 0.001 | 0.821 ± 0.005 | 1.003 ± 0.008 | −0.055 ± 0.007 | 0.000 | 0.051 | 41.6 |
| NaP | 5 | 0.833 ± 0.001 | 0.834 ± 0.001 | 1.007 ± 0.004 | −0.014 ± 0.003 | 0.155 | 0.047 | 36.3 |
| LayerNorm + WD (λ=1e-3) | 5 | 0.816 ± 0.002 | 0.809 ± 0.003 | 0.976 ± 0.005 | +0.003 ± 0.008 | 0.102 | 0.043 | 33.2 |
| Shrink & Perturb | 3 | 0.810 ± 0.000 | 0.810 ± 0.001 | 1.000 ± 0.007 | −0.018 ± 0.016 | 0.001 | 0.060 | 35.3 |
| UPGD | 3 | 0.800 ± 0.003 | 0.786 ± 0.006 | 0.951 ± 0.006 | +0.011 ± 0.010 | 0.149 | 0.095 | 39.3 |
| Parseval | 3 | 0.844 ± 0.001 | 0.844 ± 0.004 | 1.016 ± 0.009 | −0.095 ± 0.023 | 0.010 | 0.044 | 52.9 |
| Smooth-Leaky aktivasyon | 3 | 0.799 ± 0.001 | 0.784 ± 0.003 | 0.965 ± 0.010 | −0.056 ± 0.031 | 0.012 | 0.263 | 49.9 |
| Sin-MLP (sinüs aktivasyon) | 3 | 0.809 ± 0.002 | 0.799 ± 0.002 | 0.960 ± 0.006 | +0.014 ± 0.008 | 0.000 | 0.097 | 79.1 |

Şekiller: `reports/pmnist_performance_core.png` (standart/sinüs/tanh eğrileri), `reports/pmnist_performance_methods.png`
(yayımlanmış yöntemler), `reports/pmnist_mechanism_core.png` (mekanizma panelleri), `reports/pmnist_fresh_gap.png`,
`reports/pmnist_dots_*.png` (özet ölçütler, tohum noktalarıyla).

### 6.1 H1 — sınırlı yeniden parametrizasyon vs. standart ağ
Sinüs modeli standart ağa göre bütün birincil ölçütlerde tutarlı ve istatistiksel olarak anlamlı biçimde
daha iyidir (n = 10 eşleştirilmiş tohum): normalize AUC farkı **+0.011 [%95 GA +0.010, +0.012]**,
son pencere doğruluğu **+0.017 [+0.013, +0.022]**, koruma oranı **+0.021 [+0.015, +0.026]**, taze model
farkı **−0.029 [−0.035, −0.023]** (işaret-çevirme permütasyon testi p = 0.002, Holm düzeltmeli p = 0.023;
eşleştirilmiş t-testi p < 1e-6; Cohen d_z 2.2–7.8). Görev 200'de standart ağ aynı görevi sıfırdan öğrenen
taze modelin 0.021 altına düşerken sinüs modeli taze modelle eşit düzeydedir. Mekanizma düzeyinde sinüs modeli
daha az ölü birim (0.141 vs 0.177; Holm p = 0.065 permütasyon, 0.020 t-testi) ve daha yüksek temsil kertesi
(28.1 vs 25.5; Holm p = 0.023) ile birlikte daha küçük ağırlık büyüklüğü (0.098 vs 0.109) üretir.
**H1 desteklenmiştir; etki büyüklüğü ise küçüktür**: koruma oranındaki iyileşme (0.906 → 0.926) standart ağın
kaybının yaklaşık dörtte birini telafi eder; plastisite kaybı sinüs modelinde de sürmektedir
(şekil: eğri eğimi standart ağa yakındır).

### 6.2 H2 — sınırlılık mı, periyodiklik mi?
Sinüs ve tanh modelleri birbirinden ayırt edilemez: normalize AUC farkı −0.000 [−0.001, +0.001] (p = 0.87),
son pencere +0.001 [−0.003, +0.004] (p = 0.69), koruma oranı +0.004 [−0.001, +0.009] (p = 0.14). Tanh modeli
yalnızca taze model farkında (−0.012 [−0.018, −0.004]; Holm p = 0.12 permütasyon, 0.04 t-testi) ve mekanizma
ölçütlerinde (ölü birim −0.022, etkin kerte +3.6, doygunluk 0.003 vs 0.043; Holm p ≤ 0.05) sinüsten hafifçe
daha iyidir. Öneri formundaki karar kuralına göre bu sonuç **sınırlılığın baskın rolünü** gösterir: periyodik
geometri ölçülebilir bir ek katkı sağlamamakta, aksine sinüs parametrizasyonu daha fazla doygun parametre
(cos Θ ≈ 0) ve biraz daha fazla ölü birim üretmektedir. Sin-MLP kontrolü (sinüs *aktivasyon*, sınırsız ağırlık;
koruma 0.960, 3 tohum) ise sınırlı-ağırlık modellerinden daha iyi plastisite korumuş, fakat Chen ve Zhang'ın
(2026) bildirdiği gibi kayıp devam etmiştir; aktivasyon periyodikliği ile ağırlık sınırlılığı birbirinden
bağımsız etkenlerdir.

### 6.3 Yayımlanmış yöntemlerle karşılaştırma
Hiperparametreleri aynı bütçeyle seçilen yayımlanmış yöntemlerin tamamı bu ölçekte sinüs/tanh modellerinden
belirgin biçimde daha iyidir ve çoğu plastisiteyi tamamen korur (koruma oranı ≈ 1.0): Weight Clipping 0.839,
NaP 0.834, L2 Init 0.831, Continual Backprop 0.821, Shrink & Perturb 0.810, LayerNorm+WD 0.809 son pencere
doğruluğu (standart 0.743, sinüs 0.760). Parseval düzenlileştirmesi (ikincil küme) en yüksek değeri vermiştir
(0.844). Bu yöntemlerin standart ağa göre farkları eşleştirilmiş t-testinde p < 1e-6'dır; 5 ve 3 tohumlu
karşılaştırmalarda işaret-çevirme testinin ulaşabileceği en küçük p değerleri (0.0625 ve 0.25) nedeniyle
permütasyon p değerleri anlamlılık eşiğine ulaşamamaktadır (bkz. tablo notu).

En öğretici karşılaştırma **Weight Clipping (κ = 1)** ile sinüs modelidir: ikisi de efektif ağırlıkları
başlangıç sınırının katı bir aralığında tutar (κ = 1 ⇔ |W| ≤ b_l; sinüs modelinde |W| ≤ 1.5·b_l), fakat
kırpma gradyanı değiştirmeden *projeksiyon* uygularken yeniden parametrizasyon gradyanı
cos²(Θ) ile ölçekler. Kırpma 0.839, sinüs 0.760 son pencere doğruluğu vermiştir. Bu fark, sınırlılığın kendisi
değil, sınıra yaklaşan parametrelerdeki Jacobian kaynaklı adım küçülmesinin (H3) yöntemin başarımını
sınırladığına işaret eder; §8–§9'daki bileşen ve ölçek-düzeltme deneyleri bu yorumu doğrudan sınar.

## 7. Ardışık CIFAR-100 ikili sınıflandırma (İP-5)

### 7.1 Protokol kalibrasyonu
Chen ve Zhang (2026) protokolü (3000 ardışık ikili görev, gri 1024-boyutlu girdi, görev başına test doğruluğu)
standart MLP ile tek tohumlu kalibrasyon çalışmalarında yeniden üretilmeye çalışılmıştır (`results/pilot/cifar*`):

| Yapılandırma (standart MLP) | Görev | Test doğruluğu: ilk → son dilim | Ölü birim | Etkin kerte (merkezl.) | Taze model farkı (son) |
|---|---|---|---|---|---|
| 2×256, SGD lr 0.01, batch 32, 5 dönem | 3000 | 0.759 → 0.756 | 0.00 → 0.29 | 128 → 37 | ≈0 |
| 2×64, SGD lr 0.01, batch 32, 5 dönem | 3000 | 0.752 → 0.747 | 0.00 → 0.09 | 46 → 21 | ≈0 |
| 2×32, SGD lr 0.01, batch 32, 5 dönem | 3000 | 0.744 → 0.743 | 0.00 → 0.09 | 23 → 14 | ≈0 |
| 2×64, Adam lr 1e-3, batch 32, 5 dönem | 3000 | 0.740 → 0.736 | 0.00 → 0.37 | 42 → 7 | ≈0 |
| 2×64, Adam lr 1e-3, batch 32, 20 dönem | 1000 | 0.732 → 0.733 | 0.00 → 0.44 | 43 → 5 | 0.03 |
| 2×64, Adam lr 1e-2, batch 32, 5 dönem | 1000 | 0.52 → 0.50 (şans) | 0.09 → 0.50 | 28 → 0 | 0.2–0.4 |
| **2×64, SGD lr 0.01, batch 1, 1 geçiş (çevrimiçi)** | 1000 | 0.722 → 0.710 | 0.01 → 0.34 | 41 → 6 | 0.035 |

Bulgu: mini-batch ve çok dönemli eğitimde iki sınıflı görevler, iç yapısı belirgin biçimde bozulmuş (ölü
birimler, kerte çöküşü, Adam'da 7–9 kat ağırlık büyümesi) bir ağ tarafından bile 160 güncellemede öğrenilebildiği
için **performans düzeyinde** plastisite kaybı bu ölçekte görünmemektedir; aynı ağın mekanizma ölçütleri ise
Dohare vd. (2024) imzalarını taşır. Yüksek öğrenme oranlı Adam'da ağ tümüyle ölür (bütün birimler sıfır,
şans düzeyi) ve taze model öğrenmeye devam eder; bu, plastisite kaybının aşırı ucu olmakla birlikte yöntem
karşılaştırması için ayırt edici değildir. Performans düzeyinde ölçülebilir ve PMNIST ile aynı çevrimiçi yapıya
sahip olan tek yapılandırma batch 1 / tek geçiş varyantıdır; CIFAR-100 ana karşılaştırması bu **çevrimiçi
varyant** (2×64, 1000 görev, `configs/cifar_pilot.yaml`) ile yürütülmüş, kanonik 3000 görevlik mini-batch
yapılandırması (`configs/cifar_canonical.yaml`) korunmuştur. Bu, öneri formuna göre bir sapmadır ve tez
metninde raporlanacaktır: Chen ve Zhang'ın klasik MLP'de bildirdiği performans düşüşü, bu depodaki
ölçek/eniyileyici seçimleriyle mini-batch rejiminde yeniden üretilememiştir.

### 7.2 Ana karşılaştırma (çevrimiçi varyant, 1000 görev, 5 tohum)
`reports/cifar_summary_main.md`, `reports/cifar_vs_baseline.md`, `reports/cifar_performance_*.png`,
`reports/cifar_mechanism_*.png`:

| Yöntem | Normalize AUC | Son pencere | Koruma oranı | Taze model farkı | Ölü birim | Ort. \|w\| | Etkin kerte |
|---|---|---|---|---|---|---|---|
| Standart | 0.749 ± 0.003 | 0.749 ± 0.012 | 0.983 ± 0.035 | +0.025 ± 0.017 | 0.184 | 0.047 | 13.3 |
| Sinüs (γ = 5) | 0.747 ± 0.003 | 0.744 ± 0.010 | 0.977 ± 0.031 | +0.016 ± 0.011 | 0.212 | 0.047 | 12.2 |
| Tanh (γ = 5) | 0.748 ± 0.002 | 0.747 ± 0.010 | 0.983 ± 0.028 | +0.006 ± 0.027 | 0.198 | 0.047 | 13.6 |
| Weight Clipping (κ = 1) | 0.750 ± 0.003 | 0.752 ± 0.006 | 0.988 ± 0.025 | +0.014 ± 0.016 | 0.188 | 0.040 | 13.8 |
| L2 Init | 0.759 ± 0.002 | 0.763 ± 0.012 | 1.001 ± 0.034 | −0.012 ± 0.019 | 0.001 | 0.045 | 42.0 |
| Continual Backprop | 0.760 ± 0.002 | 0.766 ± 0.009 | 1.007 ± 0.030 | +0.004 ± 0.023 | 0.000 | 0.047 | 44.3 |
| NaP | 0.760 ± 0.001 | 0.766 ± 0.011 | 1.010 ± 0.031 | −0.018 ± 0.030 | 0.009 | 0.039 | 32.3 |
| LayerNorm + WD | 0.742 ± 0.003 | 0.741 ± 0.011 | 0.981 ± 0.029 | +0.024 ± 0.029 | 0.270 | 0.012 | 8.8 |

* Standart ağ 1000 görevde performans düzeyinde yalnızca hafif bir plastisite kaybı gösterir (koruma 0.983;
  taze model farkı +0.025 ± 0.017), buna karşın iç yapı güçlü biçimde bozulur (ölü birim 0.18, etkin kerte
  45 → 13). Görev başına test doğruluğunun tohumlar arası değişkenliği (sınıf çiftlerinin zorluğu) PMNIST'e
  göre çok yüksektir; bu nedenle güven aralıkları geniştir.
* **H1/H2 bu protokolde desteklenmemiştir**: sinüs ve tanh modelleri standart ağdan ayırt edilemez (AUC farkı
  −0.002 [−0.003, −0.001] ve −0.001 [−0.002, +0.001]; Holm p > 0.17). Mekanizma ölçütleri nedenini gösterir:
  bu protokolde ağırlık büyüklüğü hiç büyümemektedir (ortalama |w| 0.046 → 0.047), dolayısıyla sınırın
  (γ = 5 ile |W|/A ≈ 0.1, doygunluk 0) devreye gireceği bir rejim oluşmaz; Weight Clipping de aynı nedenle
  etkisizdir (+0.001). Plastisite kaybının bu protokoldeki taşıyıcısı ağırlık büyümesi değil, ölü birimler ve
  kerte çöküşüdür.
* Buna uygun olarak, ölü birimleri doğrudan hedefleyen yöntemler — Continual Backprop (ölü birim 0.000,
  kerte 44), L2 Init (0.001, 42) ve NaP (0.009, 32) — anlamlı iyileşme sağlar (AUC +0.010 … +0.011
  [+0.008, +0.013]; eşleştirilmiş t-testi Holm p ≤ 0.004; permütasyon testi n = 5 ile en küçük p = 0.0625).
  LayerNorm + WD ise seçilen λ = 10⁻³ ile ağırlıkları aşırı küçültmüş (|w| 0.012) ve en kötü sonucu vermiştir
  (−0.007 [−0.009, −0.005]).
* Bu bulgu PMNIST sonuçlarını tamamlar: sınırlı yeniden parametrizasyonun etkisi, plastisite kaybının ağırlık
  büyümesi kanalıyla gerçekleştiği rejimlerle sınırlıdır (Lyle vd., 2025'in "birden fazla bağımsız mekanizma"
  bulgusuyla uyumlu).

### 7.3 CIFAR-100 bileşen deneyleri (3 tohum, `reports/cifar_ablation_summary.md`, `reports/cifar_vs_sin.md`)
Sinüs modelinin bileşen varyantları (yanlılık standart, yalnızca gizli / yalnızca çıkış katmanı, γ = 2.5,
ölçek-düzeltmeli c_max = 10) bu protokolde referansla aynıdır (ΔAUC ≤ 0.001; ağırlıklar büyümediğinden
hiçbir varyantta sınır devreye girmez). Tek istisna sıkı sınırdır: γ = 1.2 ile AUC 0.712 (−0.035 [−0.039,
−0.032]; Holm p = 0.027), koruma 0.912 — ölü birim oranı %3'e düşüp kerte 35'e çıkmasına rağmen ağırlıkların
başlangıç ölçeğinin 1.2 katını aşamaması görev başına öğrenmeyi sınırlar (kapasite kısıtı). Aktivasyon
kontrolleri (CIFAR için geliştirme araması yapılmadan lr = 0.003 ile): Smooth-Leaky bu protokolde en iyi
sonucu vermiştir (AUC 0.755 ± 0.003, koruma 1.026, ölü birim 0, kerte 32 — ölü birim mekanizmasını doğrudan
ortadan kaldırdığı için); Sin-MLP (sinüs aktivasyon) ise bu öğrenme oranında çökmüştür (AUC 0.532, son pencere
şans düzeyine yakın; PMNIST'te aynı kontrol 0.809 vermişti), yani sinüs aktivasyonu veri kümesine ve öğrenme
oranına güçlü biçimde duyarlıdır.

## 8. Bileşen analizi (İP-4): sınırlılık, periyodiklik, yanlılık, katman seçimi
Final PMNIST akışlarında, ana paketle aynı tohumlarda (eşleştirilmiş) yürütülen bileşen deneyleri
(`reports/pmnist_ablation_summary.md`, `reports/pmnist_vs_sin.md`, `reports/pmnist_performance_ablation.png`,
`reports/pmnist_gamma*.png`, `reports/pmnist_scale_corrected*.png`, `reports/pmnist_sin_unit.png`). Referans:
tam-kompakt sinüs modeli (γ = 1.5, lr = 0.003; AUC 0.785, son pencere 0.760, koruma 0.926). Farklar sinüs
modeline göre eşleştirilmiş farklardır [%95 GA]; p değerleri eşleştirilmiş t-testi (Holm) — 3–5 tohumlu
karşılaştırmalarda permütasyon testi anlamlılık eşiğine ulaşamaz (bkz. §6.3).

| Varyant | n | AUC | Son pencere | Koruma | Doygunluk | Ölü birim | Kerte | ΔAUC vs sinüs |
|---|---:|---|---|---|---|---|---|---|
| Sinüs, tam-kompakt (referans) | 10 | 0.785 | 0.760 | 0.926 | 0.043 | 0.141 | 28.1 | — |
| Yanlılık standart (`compact_bias: false`) | 5 | 0.784 | 0.761 | 0.927 | 0.043 | 0.150 | 29.0 | −0.000 [−0.001, +0.001] |
| Yalnızca gizli katmanlar sınırlı | 5 | 0.778 | 0.752 | 0.918 | 0.046 | 0.151 | 26.5 | −0.007 [−0.009, −0.006] (Holm p = 0.016) |
| Yalnızca çıkış katmanı sınırlı | 5 | 0.784 | 0.759 | 0.923 | 0.193 | 0.170 | 26.6 | −0.001 [−0.002, +0.000] |
| Tanh, yanlılık standart | 3 | 0.783 | 0.758 | 0.927 | 0.003 | 0.148 | 30.6 | −0.002 [−0.003, −0.001] |
| γ = 1.2 (daha sıkı sınır) | 3 | 0.782 | 0.751 | 0.917 | 0.138 | 0.117 | 30.8 | −0.003 [−0.004, −0.002] |
| γ = 2.5 | 3 | 0.784 | 0.758 | 0.922 | 0.001 | 0.136 | 26.3 | −0.001 [−0.003, +0.001] |
| γ = 5.0 (gevşek sınır) | 3 | 0.778 | 0.756 | 0.921 | 0.000 | 0.165 | 25.8 | −0.007 [−0.009, −0.004] |
| Ölçek-düzeltmeli, c_max = 3 | 3 | 0.782 | 0.749 | 0.909 | 0.281 | 0.155 | 27.2 | −0.003 [−0.004, −0.002] (Holm p = 0.21) |
| Ölçek-düzeltmeli, c_max = 10 | 3 | 0.764 | 0.699 | 0.849 | 0.545 | 0.179 | 24.9 | −0.021 [−0.024, −0.019] (Holm p = 0.049) |
| Ölçek-düzeltmeli, c_max = 30 | 3 | 0.744 | 0.624 | 0.757 | 0.671 | 0.218 | 23.1 | −0.042 [−0.045, −0.040] (Holm p = 0.018) |
| Öğrenilebilir A | 3 | 0.773 | 0.749 | 0.913 | — | 0.143 | 29.6 | −0.012 [−0.012, −0.012] (Holm p = 0.002) |
| Sinüs + Continual Backprop | 3 | 0.825 | 0.827 | 1.009 | — | — | — | +0.039 [+0.038, +0.041] (Holm p = 0.004) |
| Standart (karşılaştırma için) | 10 | 0.774 | 0.743 | 0.906 | — | 0.177 | 25.5 | −0.011 |

**Yanlılık terimleri.** Öneri formundaki ön gözlemin aksine, yanlılıkların serbest bırakılması bu ölçekte
sonucu değiştirmemektedir (fark −0.000 [−0.001, +0.001]; tanh için de aynı). Tam-kompakt yapılandırma gereksiz
değildir fakat etkinin kaynağı değildir.

**Katman seçimi.** Etkinin büyük bölümü **çıkış katmanının** sınırlanmasından gelmektedir: yalnızca çıkış
katmanı sınırlı model referansla hemen hemen aynıdır (−0.001), yalnızca gizli katmanları sınırlı model ise
etkinin yaklaşık üçte ikisini kaybeder (−0.007, koruma 0.918). Standart ağda büyüyen ağırlıkların plastisiteyi
en çok zedelediği yer çıkış katmanıdır (çıkış ağırlıklarının büyümesi lojit ölçeğini ve dolayısıyla
softmax'ın doygunluğunu artırır); bu, Dohare vd. (2024)'ün ağırlık büyümesi bulgusunun katman düzeyinde bir
ayrıştırmasıdır. Yalnızca çıkış katmanı sınırlandığında çıkış ağırlıklarının %19'u doygunluğa ulaşır.

**Genlik (sınır genişliği).** Etki γ'ya karşı tek tepeli ve geniştir: γ = 1.2'de doygunluk %14'e çıkar ve
koruma düşer (0.917); γ = 2.5 referansla eşdeğerdir; γ = 5'te sınır o kadar gevşektir ki (|W|/A ≈ 0.20,
doygunluk 0) model standart ağa yaklaşır (0.921 → standart 0.906). Yani yöntem, "sınıra yaklaşan ama doymayan"
bir rejimde çalışmakta; etkisi ağırlıkların büyümesine izin verilen alanla orantılı olarak azalmaktadır.

**Jacobian maliyetinin giderilmesi (H3 testi).** Ölçek-düzeltmeli güncelleme — gradyanın min(1/cos², c_max) ile
ölçeklenerek sınıra yakın adımların standart büyüklüğe getirilmesi — plastisiteyi *iyileştirmemiş*, tersine
c_max arttıkça tekdüze biçimde bozmuştur (koruma 0.909 → 0.849 → 0.757; doygunluk %28 → %55 → %67). Adım
küçülmesi giderilince parametreler sınıra yığılıp donmakta ve ölü birimler artmaktadır. Sonuç, öneri
formundaki üçüncü senaryoya karşılık gelir: cos² kaynaklı sönüm bir "optimizasyon maliyeti" olmaktan çok,
sınırlı parametrizasyonun örtük bir koruyucu (yumuşak projeksiyon) mekanizmasıdır; onu kaldırmak sınırlılığın
yararını da ortadan kaldırmaktadır. Weight Clipping'in sert projeksiyonu (§6.3) bu ikilemi yaşamadığı için
üstündür: ağırlık sınırda kalır, gradyan ne ölçeklenir ne de parametre donar.

**Öğrenilebilir genlik.** A öğrenilebilir yapıldığında (ikincil duyarlılık analizi) performans düşmüştür
(−0.012; koruma 0.913): genlik eğitim boyunca sürüklenmiş (bazı katmanlarda işaret değiştirmiş; fonksiyon
A → −A dönüşümüne göre simetrik olduğundan bu bir hata değildir, ancak |A|'nın büyümesi sınırı gevşetir). Bu
yapılandırma için |W|/A tabanlı doygunluk ölçütleri işaretli A ile hesaplandığından tabloda verilmemiştir.

**Birim yeniden başlatma ile birleşim.** Sinüs modeline Continual Backprop eklendiğinde plastisite tamamen
korunmuş (koruma 1.009, son pencere 0.827) ve tek başına CBP'den (0.821, 1.003) bir miktar daha iyi sonuç
alınmıştır; sınırlılık ve ölü birim yeniden başlatma birbirini tamamlayan, kısmen bağımsız mekanizmalara
etki etmektedir (Lyle vd., 2025).

**Serbest parametre ölçeği (`theta_scale: unit`).** Literal W = A·sin(Θ) biçiminde (katman başına η·A²
etkin öğrenme oranı; ana deneylerde kullanılan genlik-ölçekli Φ = A·Θ biçimi yerine) en iyi öğrenme oranıyla
(lr = 0.1) koruma oranı 0.938'e çıkmış fakat normalize AUC 0.782'ye düşmüştür: ilk katmanın etkin öğrenme
oranı ≈ 60 kat küçük olduğundan ağ daha yavaş öğrenir, ancak daha az plastisite kaybeder (ölü birim oranı
%42'ye rağmen). Bu, "yavaş öğrenen ağ daha az plastisite kaybeder" ödünleşiminin bir örneğidir ve iki
ölçeklendirmenin aynı mekanizmayı farklı öğrenme oranı dağılımlarıyla sergilediğini gösterir
(`reports/pmnist_sin_unit.png`).

## 9. Mekanizma analizi (İP-3): Jacobian, etkin adım, Fisher koordinatları
PMNIST ana paketindeki standart/sinüs/tanh çalışmalarından (10 tohum) ilk 20 ve son 20 görevin ortalamaları
(`reports/pmnist_mechanism_core.png`; ölçütler görev sonunda 1.000 örneklik sonda üzerinde, adım ölçütleri her
100 adımda bir örneklenerek hesaplanmıştır):

| Ölçüt | Standart (erken → son) | Sinüs (erken → son) | Tanh (erken → son) |
|---|---|---|---|
| Ortalama \|W\|/A | — | 0.36 → 0.59 | 0.36 → 0.55 |
| Ortalama cos²(·) (normalize Jacobian²) | 1 | 0.82 → 0.56 | 0.71 → 0.47 |
| Doygun parametre oranı (\|cos\| < 0.1) | — | 0.000 → 0.043 | 0.000 → 0.003 |
| ‖∇_Θ L‖ / ‖∇_W L‖ | 1 | 0.87 → 0.61 | 0.80 → 0.58 |
| ‖∇_W L‖ (örnek başına) | 5.5 → 6.1 | 5.9 → 8.2 | 6.4 → 8.3 |
| Tr F_W (örnek Fisher) | 161 → 134 | 184 → 258 | 187 → 246 |
| Tr F_Θ | 161 → 134 | 141 → 97 | 121 → 83 |
| Adım ‖ΔW‖ | 0.028 → 0.028 | 0.024 → 0.020 | 0.022 → 0.018 |
| Ölü birim oranı | 0.01 → 0.18 | 0.01 → 0.14 | 0.02 → 0.12 |
| Etkin kerte (merkezlenmiş, son gizli katman) | 45 → 26 | 43 → 28 | 41 → 32 |

Üç bulgu H3'ü doğrudan destekler:

1. **Jacobian kaynaklı adım küçülmesi gerçekleşmektedir.** Ağırlıklar sınıra yaklaştıkça (|W|/A 0.36 → 0.59)
   cos² çarpanı 0.82'den 0.56'ya düşmekte, aynı gradyan için efektif adım ‖ΔW‖ standart ağın 0.028'ine karşı
   0.020'ye gerilemektedir. Yani sınırlı modellerin "yavaşlaması" yalnızca ağırlıkların sınırda sıkışmasından
   değil, sınıra yaklaşan her parametrede sürekli azalan bir etkin öğrenme oranından kaynaklanır.
   Tam doygunluk (|cos Θ| < 0.1) ana ayarda (γ = 1.5, lr = 0.003) parametrelerin yalnızca %4'ünde görülür;
   yüksek öğrenme oranlarında ise (geliştirme akışları, lr ≥ 0.03) %80'in üzerine çıkar ve sinüs modeli
   tamamen donar (§5, `reports/pmnist_dev_hparams.md`).
2. **Fonksiyon uzayında duyarlılık korunmaktadır.** Efektif ağırlık koordinatlarında ölçülen gradyan normu ve
   Fisher izi sınırlı modellerde standart ağdan *yüksektir* (8.2 vs 6.1; 258 vs 134) ve eğitim boyunca artar;
   standart ağda ise Fisher izi azalır. Sınırlı modeller yeni görev sinyaline duyarlılığını (kayıp yüzeyinin
   eğriliğini) kaybetmemekte, fakat bu sinyali Jacobian nedeniyle daha küçük adımlara dönüştürmektedir. Bu,
   kırpma ile projeksiyonun (Weight Clipping) aynı sınır altında neden çok daha iyi sonuç verdiğini açıklar:
   projeksiyon gradyanı ölçeklemez.
3. **Koordinat bağımlılığı uyarısı doğrulanmıştır.** Aynı modelde Tr F_Θ (97) ile Tr F_W (258) birbirinden
   2.7 kat farklıdır ve zıt yönde değişir (Θ koordinatında azalır, W koordinatında artar); ‖∇_Θ L‖/‖∇_W L‖
   oranı 0.87'den 0.61'e düşer. Yeniden parametrize edilmiş ağlarda Chen ve Zhang (2026) tarzı Fisher/gradyan
   ölçümlerinin ancak ortak efektif-ağırlık koordinatlarında karşılaştırılabileceği (Martens, 2020) deneysel
   olarak gösterilmiştir; Θ koordinatında ölçüm yapan bir analiz "plastisite kaybı" ile "Jacobian sönümünü"
   birbirine karıştırır.

Ölü birim ve kerte ölçütleri de sınırlı modellerde daha iyidir (ölü birim 0.12–0.14 vs 0.18; kerte 28–32 vs 26),
fakat yayımlanmış yöntemlere göre (ölü birim ≈ 0, kerte 42–53) iyileşme küçüktür: ağırlık büyüklüğünü
sınırlamak tek başına ReLU birimlerinin ölmesini engellememektedir (Lyle vd., 2025 ile uyumlu: birden fazla
kısmen bağımsız mekanizma).

## 10. Durağan öğrenme kontrolü
Sürekli öğrenme protokollerindeki farkların genel kapasite farkından kaynaklanmadığını göstermek için
standart, sinüs ve tanh modelleri i.i.d. koşullarda karşılaştırılmıştır (`suites/stationary.yaml`,
`reports/stationary_summary.md`; 5 tohum, lr ∈ {0.01, 0.03, 0.1} içinden model başına en iyi değer):

| Veri kümesi / model | lr | Test doğruluğu | Eğitim doğruluğu |
|---|---|---|---|
| MNIST (784-100-100-100-10, 10 dönem), standart | 0.1 | 0.977 ± 0.001 | 0.993 |
| MNIST, sinüs | 0.1 | 0.978 ± 0.002 | 0.993 |
| MNIST, tanh | 0.1 | 0.978 ± 0.002 | 0.993 |
| CIFAR-100 gri, 100 sınıf (1024-256-256-100, 20 dönem), standart | 0.01 | 0.162 ± 0.003 | 0.318 |
| CIFAR-100 gri, sinüs | 0.01 | 0.163 ± 0.001 | 0.290 |
| CIFAR-100 gri, tanh | 0.01 | 0.162 ± 0.002 | 0.273 |

Üç model durağan koşullarda aynı test doğruluğuna ulaşmaktadır (MNIST 0.977–0.978; CIFAR-100 0.162–0.163;
farklar güven aralıklarının içinde). CIFAR-100'de sınırlı modellerin eğitim doğruluğu biraz daha düşüktür
(0.29/0.27 vs 0.32): sınırlı ağırlık uzayı aşırı uyumu bir miktar kısıtlar fakat genellemeyi değiştirmez.
Dolayısıyla §6'daki plastisite farkı ve §7'deki fark yokluğu genel öğrenme kapasitesindeki bir eksiklikle
açıklanamaz; fark, eğitim ilerledikçe ortaya çıkan dinamiklerle ilgilidir. Sınırlı modeller yüksek öğrenme
oranlarına karşı daha dayanıklıdır (CIFAR-100, lr = 0.1: standart 0.126, sinüs 0.142, tanh 0.145), bu da
§5'teki geliştirme akışı gözlemiyle (sınırlı modellerin büyük lr'de çökmemesi) tutarlıdır.

## 11. İkincil deneyler: Adam duyarlılığı, serbest parametre ölçeği, ikincil karşılaştırma kümesi
Genlik taraması ve ölçek-düzeltmeli güncelleme §8'de; ikincil karşılaştırma kümesi (Shrink & Perturb, UPGD,
Parseval, Smooth-Leaky, Sin-MLP) §6'daki ana tabloda verilmiştir. Bu bölüm eniyileyici duyarlılığını ele alır
(`reports/pmnist_adam_unit.md`, `reports/pmnist_adam.png`; 3 eşleştirilmiş tohum, PMNIST final akışları,
Adam için ayrı bir geliştirme araması yapılmamış, iki öğrenme oranı doğrudan raporlanmıştır).

| Eniyileyici / model | AUC | Son pencere | Koruma | Taze model farkı | Ölü birim | Ort. \|w\| |
|---|---|---|---|---|---|---|
| Adam 3e-4, standart | 0.789 ± 0.006 | 0.754 ± 0.002 | 0.898 ± 0.006 | +0.083 ± 0.023 | 0.457 | 0.179 |
| Adam 3e-4, **sinüs** | 0.819 ± 0.004 | 0.811 ± 0.005 | 0.966 ± 0.008 | +0.015 ± 0.017 | 0.163 | 0.102 |
| Adam 3e-4, tanh | 0.810 ± 0.001 | 0.791 ± 0.000 | 0.944 ± 0.004 | +0.041 ± 0.033 | 0.175 | 0.106 |
| Adam 1e-3, standart | 0.737 ± 0.021 | 0.707 ± 0.020 | 0.877 ± 0.032 | +0.127 ± 0.052 | 0.678 | 0.356 |
| Adam 1e-3, **sinüs** | 0.824 ± 0.004 | 0.826 ± 0.010 | 0.997 ± 0.022 | −0.002 ± 0.035 | 0.354 | 0.120 |
| Adam 1e-3, tanh | 0.812 ± 0.002 | 0.808 ± 0.008 | 0.974 ± 0.008 | +0.028 ± 0.022 | 0.260 | 0.140 |
| (SGD 3e-3, standart / sinüs / tanh; §6) | 0.774 / 0.785 / 0.785 | 0.743 / 0.760 / 0.761 | 0.906 / 0.926 / 0.931 | | | |

**Adam altında etki çok daha büyüktür.** Adam, standart ağda ağırlık büyümesini (|w| 0.18–0.36; SGD'de 0.11)
ve ölü birim oranını (%46–%68) şiddetlendirir ve plastisite kaybını derinleştirir (koruma 0.898 / 0.877, taze
model farkı +0.08 / +0.13). Sınırlı modeller bu rejimde çok daha iyi korunur: sinüs modelinin standart ağa göre
eşleştirilmiş farkı lr = 3e-4'te ΔAUC **+0.030 [+0.028, +0.034]**, koruma oranı **+0.068 [+0.067, +0.069]**,
taze model farkı −0.068; lr = 1e-3'te ΔAUC **+0.087 [+0.080, +0.094]**, koruma +0.120 [+0.102, +0.141],
ölü birim −0.32 (eşleştirilmiş t-testi p ≤ 0.02; n = 3 ile permütasyon testi p = 0.25). Sinüs modeli Adam 1e-3
ile plastisiteyi tamamen korur (koruma 0.997) ve SGD'li bütün yapılandırmalardan daha yüksek AUC verir (0.824).
H1, ağırlık büyümesinin güçlü olduğu Adam rejiminde çok daha büyük bir etki büyüklüğüyle desteklenmektedir.

**Adam altında periyodiklik lehine bir fark ortaya çıkar (H2'nin eniyileyiciye bağlı kısmı).** SGD'de ayırt
edilemeyen sinüs ve tanh modelleri Adam'da ayrışır: sinüs tanh'tan tutarlı biçimde daha iyidir (ΔAUC
+0.009 [+0.008, +0.011] ve +0.012 [+0.010, +0.015]; koruma +0.023; t-testi p ≤ 0.04). Mekanistik açıklama
§9'daki Jacobian analizinden çıkar: Adam güncellemeyi parametre başına gradyan büyüklüğüne böldüğünden
cos²(Θ) kaynaklı adım küçülmesi büyük ölçüde iptal olur; bu durumda sinüsün tanh'tan tek farkı, doyma
bölgesinden (Θ ≈ ±π/2) ötesine geçerek ağırlığın tekrar küçülebilmesidir (periyodik geometri), tanh'ta ise
doyan parametre kalıcı olarak sınırda kalır. Sinüs modelinde Adam ile ölü birim oranı (%35) tanh'tan yüksek
olmasına rağmen performansın daha iyi olması bu yorumla uyumludur. SGD'de ise cos² sönümü doyma bölgesine
girişi engellediğinden periyodikliğin devreye gireceği bir durum oluşmaz.

**Serbest parametre ölçeği.** Literal W = A·sin(Θ) biçimi (`theta_scale: unit`) lr = 0.1'de koruma 0.938
(genlik-ölçekli ana biçim 0.926) fakat AUC 0.781 (0.785) vermiş; lr = 0.3 ve 1.0'da sırasıyla kısmi ve tam
çöküş gözlenmiştir (koruma 0.894 ve 0.341; doygunluk ve ölü birim %50). Bu biçim katman başına A² ile ölçeklenen
etkin öğrenme oranı nedeniyle ilk katmanı çok yavaş eğitir; ana deneylerde kullanılan Φ = A·Θ ölçeklemesi
başlangıçta standart ağla eşit etkin adım sağladığından karşılaştırmalar için uygun biçimdir (§2, `docs/PROTOKOL.md`).

**İkincil karşılaştırma kümesi** (§6 tablosu; 3 tohum, yarı bütçeli hiperparametre araması): Parseval
düzenlileştirmesi bütün yöntemler içinde en yüksek son pencere doğruluğunu (0.844, koruma 1.016) ve en yüksek
temsil kertesini (52.9) vermiştir; Shrink & Perturb plastisiteyi tam korumuş (1.000), UPGD (0.951) ve Smooth-Leaky
aktivasyon (0.965) kısmen korumuştur. Smooth-Leaky ve Sin-MLP'nin ölü birim oranı sıfıra yakındır fakat
plastisite kaybı sürmektedir — ölü birimlerin ortadan kalkması tek başına yeterli değildir (Lyle vd., 2023;
2025 ile uyumlu).

## 12. Sınırlılıklar ve sonraki adımlar
**Ölçek.** Bütün sonuçlar CPU bütçesine uyarlanmış küçültülmüş protokollerden gelmektedir (PMNIST: 200 görev ×
5.000 örnek, 3×100 birim; CIFAR-100: 1000 görev, 2×64 birim, çevrimiçi varyant). Kanonik protokoller (800 görev ×
60.000 örnek, 3×2000 birim; 3000 görev) aynı kodla ve yapılandırma dosyalarıyla çalıştırılmaya hazırdır
(`configs/*_canonical.yaml`); İP-5'in GPU altyapısında yürütülmesiyle (i) sinüs/tanh modelinin uzun ufukta
yavaşlayan kaybının standart ağa göre farkının büyüyüp büyümediği, (ii) geniş ağlarda (2000 birim) ölü birim
ve sınır doygunluğu dinamiklerinin değişip değişmediği sınanmalıdır. Küçültülmüş protokolde 10 tohumla güven
aralıkları çok dardır; etki büyüklükleri küçük (ΔAUC ≈ 0.01) olsa da yönleri kesindir.

**CIFAR-100 protokolü.** Chen ve Zhang (2026)'ın klasik MLP'de bildirdiği performans düzeyindeki plastisite
kaybı, mini-batch/çok dönemli rejimde bu altyapıda yeniden üretilememiş; çevrimiçi varyantta zayıf biçimde
gözlenmiştir (§7). Kaynak çalışmanın kesin eğitim bütçesi (dönem/adım sayısı, eniyileyici, genişlik) ve kod
çıktılarıyla birebir karşılaştırma tez aşamasında yapılmalıdır.

**İstatistik.** 5 ve 3 tohumlu karşılaştırmalarda ön-kayıtlı işaret-çevirme testinin çözünürlüğü yetersiz
kalmış (en küçük p = 0.0625 / 0.25), sonuçlar eşleştirilmiş t-testiyle birlikte raporlanmıştır. Tez
aşamasında yayımlanmış yöntemlerin de 10 tohumla çalıştırılması (ek ≈ 6 saat CPU) permütasyon testini
yeterli hale getirir.

**Yöntem uygulamaları.** Karşılaştırma yöntemleri kaynak makalelerin tanımlarına göre yazılmış ve bağımsız
inceleme + birim testlerinden geçirilmiştir; ancak kaynak kodlarla birebir doğrulama (Dohare vd., 2024 ve
Elsayed vd., 2024'ün açık uygulamaları) yapılmamıştır. Smooth-Leaky aktivasyonu kavramsal bir uygulamadır
(a·x + (1−a)·softplus(x)); Lillo ve Cheney (2026)'ın kesin tanımıyla eşleştirilmelidir. Hyperspherical
normalization (Lee vd., 2025) kapsam dışı bırakılmıştır. Öğrenilebilir genlik varyantında genlik
işaret/ölçek sürüklenmesine karşı kısıtlanmamıştır (|A| parametrizasyonu veya A için ayrı öğrenme oranı
denenebilir).

**Yöntemin kendisi hakkında.** Bulgular, sınırlı yeniden parametrizasyonun (i) yalnızca ağırlık büyümesinin
plastisite kaybını taşıdığı rejimlerde, (ii) küçük ama tutarlı bir koruma sağladığını ve (iii) yararının sert
projeksiyon (Weight Clipping) karşısında Jacobian sönümünün çift yönlü etkisi nedeniyle sınırlı kaldığını
göstermektedir. Tez için en verimli sonraki adımlar: (a) kırpma ile reparametrizasyonu aynı sınırda doğrudan
eşleyen bir "yumuşak/sert projeksiyon" deney dizisi (A·tanh(Θ) ile kırpma arasında enterpolasyon), (b)
sınırlılığı ölü birim yeniden başlatma ile birleştiren varyant (sinüs + CBP bu aşamada en iyi sonuçlardan
birini vermiştir), (c) çıkış katmanı sınırlamasının tek başına incelenmesi (etkinin ana taşıyıcısı), (d)
Fisher/gradyan ölçümlerinin tez metninde yalnızca efektif ağırlık koordinatlarında raporlanması.
