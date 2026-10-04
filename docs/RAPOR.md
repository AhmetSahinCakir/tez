# Proje Aşaması Raporu — Sınırlı-Periyodik Ağırlık Yeniden Parametrizasyonu ile Plastisite Kaybı

> Bu rapor tez öneri formundaki İş Paketleri 1–5'in proje aşamasını kapsar: deney altyapısının kurulması,
> plastisite kaybı bulgularının pilot replikasyonu, karşılaştırma yöntemleri kütüphanesi, önerilen
> A·sin(Θ) / A·tanh(Θ) yeniden parametrizasyonları, bileşen deneyleri, mekanizma analizleri ve istatistiksel
> karşılaştırma. Bütün sayılar bu depodaki `results/` çalışmalarından `scripts/analyze.py` ile üretilmiştir.

<!-- İÇİNDEKİLER: 1 Özet · 2 Altyapı · 3 Hesaplama bütçesi ve protokol ölçekleri · 4 Pilot replikasyon ·
     5 Hiperparametre seçimi · 6 Ana karşılaştırma (PMNIST) · 7 CIFAR-100 · 8 Bileşen analizi ·
     9 Mekanizma analizi (H3) · 10 Durağan kontrol · 11 İkincil deneyler · 12 Sınırlılıklar ve sonraki adımlar -->

## 1. Özet
_(sonuçlar tamamlandığında doldurulacak)_

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
_(doldurulacak)_

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

## 8. Bileşen analizi (İP-4): sınırlılık, periyodiklik, yanlılık, katman seçimi
_(doldurulacak)_

## 9. Mekanizma analizi (İP-3): Jacobian, etkin adım, Fisher koordinatları
_(doldurulacak)_

## 10. Durağan öğrenme kontrolü
_(doldurulacak)_

## 11. İkincil deneyler: genlik taraması, Adam, ölçek düzeltmesi, ikincil karşılaştırma kümesi
_(doldurulacak)_

## 12. Sınırlılıklar ve sonraki adımlar
_(doldurulacak)_
