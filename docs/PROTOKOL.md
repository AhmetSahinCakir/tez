# Deney Protokolü

Bu belge, tez öneri formundaki "Yöntem" bölümünün bu depodaki somut karşılığını tanımlar. Her madde için
ilgili kod ve yapılandırma anahtarı verilmiştir.

## 1. Modeller ve yeniden parametrizasyon

| Model | Efektif ağırlık | Serbest parametre | Kod |
|---|---|---|---|
| Standart | W = Θ | Θ | `model.reparam: {hidden: standard, output: standard}` |
| Sinüs (önerilen) | W = A·sin(Θ) | Θ | `model.reparam: {hidden: sin, output: sin}` |
| Tanh (kontrol, H2) | W = A·tanh(Θ) | Θ | `model.reparam: {hidden: tanh, output: tanh}` |
| Üçgen dalga (sönümsüz, yansımalı) | W = A·tri(Θ), tri(x) = (2/π)·arcsin(sin(πx/2)) | Θ | `model.reparam: {hidden: tri, output: tri}` |
| Jacobian tabanlı sinüs/tanh | ileri geçiş aynı; geri geçişte f′ yerine sign(f′)·max(\|f′\|, ε) | Θ | `model.jacobian_floor: ε` (ε = 1: yansıma) |

* **Genlik**: A_l = γ·b_l; b_l katmanın Kaiming-uniform başlatma sınırı (He vd., 2015; `init: kaiming_uniform`,
  ReLU kazancı ile b_l = √(6/fan_in)). Varsayılan γ = 1.5 (`model.gamma`); genlik taraması ikincil deneydir.
* **Eşleştirilmiş başlangıç**: ortak W₀ ~ U(−b_l, b_l) çekilir; Θ₀ = arcsin(W₀/A) (sinüs), Θ₀ = atanh(W₀/A) (tanh).
  Üç model aynı efektif W₀ ile başlar (`tests/test_reparam.py` bunu doğrular).
* **Tam-kompakt yapılandırma** (`model.compact_bias: true`): yanlılıklar da b = A_b·sin(θ_b) biçimindedir
  (A_b = γ/√fan_in). `compact_bias: false` yalnızca ağırlık matrislerini sınırlar (bileşen deneyi).
* **Sönümsüz haritalar** (proje aşamasındaki mekanizma analizinden sonra eklenmiştir): üçgen dalga haritası
  sınırın içinde standart katmanla birebir aynıdır (tri(x) = x, |x| ≤ 1), sınıra ulaşan parametre donmak yerine
  yansır (|dW/dΘ| her yerde sabit); `jacobian_floor` ise sinüs/tanh için ileri geçişi değiştirmeden geri geçişte
  türevin büyüklüğünü ε tabanına sabitler (ε = 1 ile sinüs pürüzsüz bir yansıma haritası olur). γ = 1 bu
  haritalarla kullanılabilir (κ = 1 kırpmayla aynı sınır). Raporlanan Jacobian ölçütleri gerçek türevi kullanır.
* **Serbest parametre ölçeği** (`model.theta_scale`): `amplitude` (ana deneyler) serbest parametreyi Φ = A·Θ olarak
  tanımlar (W = A·sin(Φ/A)); bu, Θ üzerinde katman başına η/A_l² öğrenme oranına denktir ve başlangıçta her
  katmanın etkin adımını standart modelle eşitler. `unit` ise W = A·sin(Θ) biçimini olduğu gibi kullanır
  (ΔW ≈ −η·A²·cos²Θ·∂L/∂W; A² katman başına sabit bir öğrenme-oranı çarpanıdır ve ilk katmanda ≈0.017'dir).
  Her iki durumda da sınıra yaklaşan parametrelerde cos² kaynaklı adım küçülmesi (H3) aynen geçerlidir; `unit`
  duyarlılık analizinde raporlanır.
* **Mimari**: `hidden_sizes` gizli katman genişlikleri; aktivasyon `relu` (ana), `sin` (Sin-MLP kontrolü),
  `smooth_leaky` (yapısal kontrol); isteğe bağlı `layer_norm` (NaP ve LN+WD için).

## 2. Görev akışları

### Online Permuted MNIST (Dohare vd., 2024) — `stream.name: pmnist`
* Görev t: tohuma ve t'ye bağlı bağımsız bir piksel permütasyonu; MNIST eğitim görüntüleri rastgele sırada,
  tek geçişle, `train.batch_size: 1` ile sunulur. Görev kimliği modele verilmez.
* Ölçüt: görev boyunca güncellemeden önce yapılan tahminlerin doğruluğu (çevrimiçi doğruluk).
* Kanonik: 800 görev × 60.000 örnek, 3×2000 birim (`configs/pmnist_canonical.yaml`).
* Pilot/CPU: `configs/pmnist_pilot.yaml` ile `stream.n_tasks`, `stream.samples_per_task` (görev başına rastgele
  alt küme) ve `hidden_sizes` küçültülür; protokolün diğer bütün özellikleri aynıdır.

### Ardışık CIFAR-100 ikili sınıflandırma (Chen ve Zhang, 2026) — `stream.name: cifar100_binary`
* Görev t: 100 sınıftan rastgele 2 farklı sınıf; eğitim görüntüleri (sınıf başına 500) gri seviyeye
  (0.299R+0.587G+0.114B) dönüştürülüp 1024-boyutlu vektöre açılır ve eğitim kümesi istatistikleriyle
  standardize edilir; mini-batch (`train.batch_size: 32`), görev başına `train.epochs_per_task` geçiş.
* Ölçüt: görevin iki sınıfına ait test görüntülerinde (sınıf başına 100) doğruluk.
* Kanonik: 3000 görev (`configs/cifar_canonical.yaml`).

### Durağan kontrol — `stream.name: stationary`
i.i.d. MNIST (10 sınıf) veya gri CIFAR-100 (100 sınıf); dönem başına test doğruluğu. Sürekli öğrenmedeki
farkların genel kapasite kaybından kaynaklanmadığını göstermek için standart/sinüs/tanh karşılaştırılır.

## 3. Karşılaştırma yöntemleri (`method.name`)

| Yöntem | Kaynak | Müdahale noktası | Hiperparametreler |
|---|---|---|---|
| `baseline` | — | yok | lr |
| `weight_clipping` | Elsayed vd., 2024 | adım sonrası kırpma [−κb_l, κb_l] | lr, κ |
| `l2_init` | Kumar vd., 2025 | λ‖θ−θ₀‖² gradyanı | lr, λ |
| `continual_backprop` | Dohare vd., 2024 | düşük yararlılıklı birimlerin seçici yeniden başlatılması | lr, ρ, m, η |
| `nap` | Lyle vd., 2024 | LayerNorm + ağırlık normunun başlangıç normuna projeksiyonu | lr |
| `ln_wd` | Lyle vd., 2025 | LayerNorm + ağırlık sönümü | lr, λ |
| `shrink_perturb` | Ash ve Adams, 2020 | θ ← (1−s)θ + σε | lr, s, σ |
| `upgd` | Elsayed ve Mahmood, 2024 | yararlılık ölçekli pertürbe gradyan | lr, β, σ |
| `parseval` | Chung vd., 2024 | β‖WWᵀ−I‖² düzenlileştirmesi | lr, β |
| `scale_corrected` | bu tez (ikincil) | sin/tanh gradyanının min(1/cos², c_max) ile ölçeklenmesi | lr, c_max |

Smooth-Leaky (Lillo ve Cheney, 2026) ve Sin-MLP (Chen ve Zhang, 2026) kontrolleri `model.activation` ile
tanımlanır. Hyperspherical normalization (Lee vd., 2025) hesaplama bütçesi nedeniyle bu aşamada kapsam dışıdır.

## 4. Hiperparametre seçimi
Her yönteme eşit arama bütçesi: 5 öğrenme oranı (veya 4 öğrenme oranı × 2 yönteme özgü değer) tek tohumla,
**geliştirme akışlarında** (`stream_seed_offset: 1000`; permütasyonlar/sınıf çiftleri final akışlarından
farklı). Seçim ölçütü normalize AUC. Seçilen değerler `suites/selected_*.yaml` dosyalarında dondurulur ve
final akışlarında kullanılır; final akışları seçim için kullanılmaz.

## 5. Ölçütler
* **Performans**: görev başına çevrimiçi doğruluk (PMNIST) / test doğruluğu (CIFAR-100);
  normalize AUC (görev-performans eğrisinin ortalaması), erken/son pencere ortalaması (ilk/son %10 görev),
  plastisite koruma oranı = son/erken, taze-model farkı (aynı görevi sıfırdan öğrenen eşleştirilmiş modelle fark;
  `fresh_reference.every_n_tasks`).
* **Mekanizma** (`metrics`): ortalama |W| ve ‖W‖_F, ölü/etkisiz birim oranı, temsilin kararlı/etkin kertesi
  (ham ve merkezlenmiş), aktivasyon büyüklükleri, gradyan normları, Fisher izi ve etkin kertesi — hem Θ hem W
  koordinatlarında (F_Θ = Jᵀ F_W J; `jac_fro_ratio` = ‖∇_Θ‖/‖∇_W‖). Sinüs modeli için ayrıca |W|/A,
  |cos Θ|, cos²Θ (η_eff çarpanı) ve doygun (|cos Θ|<0.1) parametre oranı; adım başına ‖ΔW‖, ‖ΔΘ‖, ‖∇_Θ‖.

## 6. İstatistik
Deney birimi bağımsız tekrar (tohum = görev akışı). Yöntemler aynı tohumlarda çalıştığından karşılaştırmalar
eşleştirilmiştir: ortalama fark, bootstrap %95 güven aralığı, eşleştirilmiş permütasyon (işaret çevirme) testi,
Wilcoxon, Cohen d_z; çoklu karşılaştırmalarda Holm düzeltmesi (`plasticity/metrics/stats.py`).

## 7. Yeniden üretilebilirlik
Tohum ve görev akışı tanımları yapılandırma dosyalarındadır; akışlar `(seed, task)` çiftinin deterministik
fonksiyonudur; her çalışma kendi `config.json` dosyasını yazar; paket çalıştırıcı tamamlanmış çalışmaları atlar.
