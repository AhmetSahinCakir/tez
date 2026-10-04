# Sınırlı-Periyodik Ağırlık Yeniden Parametrizasyonu ile Plastisite Kaybının Azaltılması

Bu depo, *derin sürekli öğrenmede plastisite kaybı* (loss of plasticity) üzerine hazırlanan yüksek lisans tezinin
deney altyapısını ve proje aşamasının sonuçlarını içerir. Önerilen yöntem, her katmanın efektif ağırlıklarını
sınırlı ve periyodik bir dönüşüm üzerinden tanımlar:

```
W_l = A_l · sin(Θ_l)          (önerilen; |W_l| ≤ A_l, Θ'da periyodik)
W_l = A_l · tanh(Θ_l)         (kontrol; sınırlı fakat periyodik olmayan)
W_l = Θ_l                     (standart)
```

Üç model **eşleştirilmiş başlangıç** ile (ortak Kaiming-uniform W₀, A_l = γ·b_l, Θ₀ = arcsin(W₀/A_l) / atanh(W₀/A_l))
tam olarak aynı efektif ağırlıklardan başlar. Yanlılık terimleri de aynı biçimde sınırlandırılır ("tam-kompakt").

Deney protokolleri:

* **Online Permuted MNIST** (Dohare vd., 2024): her görev MNIST'e yeni bir rastgele piksel permütasyonu uygular;
  örnekler tek geçişle, tek tek (batch 1) sunulur; ölçüt görev boyunca çevrimiçi doğruluktur.
* **Ardışık CIFAR-100 ikili sınıflandırma** (Chen ve Zhang, 2026): her görev iki rastgele sınıf; gri seviye
  1024-boyutlu girdi; ölçüt görevin test doğruluğu.
* **Durağan kontrol**: i.i.d. MNIST / CIFAR-100 ile genel kapasite karşılaştırması.

Karşılaştırma yöntemleri: standart SGD, Weight Clipping (Elsayed vd., 2024), L2 Init (Kumar vd., 2025),
Continual Backprop (Dohare vd., 2024), Normalize-and-Project (Lyle vd., 2024), LayerNorm + Weight Decay
(Lyle vd., 2025); ikincil: Shrink-and-Perturb, UPGD, Parseval, Smooth-Leaky aktivasyon, Sin-MLP kontrolü,
ölçek-düzeltmeli sinüs güncellemesi.

Sonuçlar ve proje raporu: [`docs/RAPOR.md`](docs/RAPOR.md). Deney protokolü ayrıntıları: [`docs/PROTOKOL.md`](docs/PROTOKOL.md).
Üretilen tablolar ve şekiller: [`reports/`](reports/INDEX.md). Ham çalışma kayıtları (görev başına ölçütler): `results/`.

## Proje aşamasının başlıca bulguları (küçültülmüş CPU protokolleri)

| Online Permuted MNIST, 200 görev, SGD | Normalize AUC | Son 20 görev doğruluğu | Plastisite koruma oranı |
|---|---|---|---|
| Standart MLP (10 tohum) | 0.774 ± 0.001 | 0.743 ± 0.003 | 0.906 ± 0.004 |
| **A·sin(Θ)** (10 tohum) | 0.785 ± 0.001 | 0.760 ± 0.004 | 0.926 ± 0.004 |
| A·tanh(Θ) (10 tohum) | 0.785 ± 0.001 | 0.761 ± 0.002 | 0.931 ± 0.004 |
| Weight Clipping κ=1 (5 tohum) | 0.836 ± 0.001 | 0.839 ± 0.002 | 1.017 ± 0.004 |
| Continual Backprop (5 tohum) | 0.821 ± 0.001 | 0.821 ± 0.005 | 1.003 ± 0.008 |

* **H1**: sınırlı yeniden parametrizasyon plastisite kaybını anlamlı fakat küçük ölçüde azaltır (ΔAUC +0.011
  [+0.010, +0.012], eşleştirilmiş permütasyon p = 0.002); Adam ile etki çok büyür (ΔAUC +0.03 … +0.09).
* **H2**: sinüs ve tanh SGD altında ayırt edilemez → etki sınırlılıktan kaynaklanır; Adam altında sinüs tanh'ı geçer.
* **H3**: sınıra yaklaşan parametrelerde cos²Θ kaynaklı adım küçülmesi ölçülmüştür; bunu gideren ölçek-düzeltmeli
  güncelleme plastisiteyi *bozar* (parametreler sınıra yığılıp donar) — sönüm koruyucu bir bileşendir. Sert
  projeksiyonlu Weight Clipping aynı sınırla çok daha iyidir.
* Etkinin ana taşıyıcısı çıkış katmanının sınırlanmasıdır; yanlılıkların sınırlanması önemsizdir.
* CIFAR-100 ikili akışında ağırlıklar büyümediğinden sınırlı modellerin etkisi yoktur; yalnızca ölü birimleri
  hedefleyen yöntemler (CBP, L2 Init, NaP) iyileşme sağlar.
* Durağan kontrol: üç model i.i.d. koşullarda aynı test doğruluğuna ulaşır.
* Toplam 451 çalışma (≈ 95 CPU-saati); bütün ham kayıtlar `results/`, tablolar/şekiller `reports/` altındadır.

## Kurulum

```bash
pip install -r requirements.txt          # CPU PyTorch yeterlidir
bash scripts/download_data.sh            # MNIST + CIFAR-100 (S3 aynaları) -> data/processed/*.npz
python -m pytest tests -q                # birim testleri
```

## Tek bir deney çalıştırma

```bash
python -m plasticity.run --config configs/pmnist_pilot.yaml --out results/dev/sin_seed0 \
    --set model.reparam.hidden=sin --set model.reparam.output=sin --set seed=0
python -m plasticity.run --config configs/cifar_pilot.yaml --out results/dev/cbp \
    --set method.name=continual_backprop --set method.replacement_rate=1e-4
```

Her çalışma `config.json`, görev başına bir satır içeren `tasks.jsonl` (performans + mekanizma ölçütleri) ve
`summary.json` (normalize AUC, erken/son pencere ortalaması, plastisite koruma oranı, taze-model farkı, ...) üretir.

## Deney paketleri (yöntem × tohum ızgaraları)

```bash
python scripts/run_suite.py suites/pmnist_dev.yaml            # hiperparametre araması (geliştirme akışları)
python scripts/select_hparams.py --results results/pmnist_dev --out suites/selected_pmnist.yaml
python scripts/run_suite.py suites/pmnist_main.yaml           # final karşılaştırma (≥10 tohum)
python scripts/analyze.py --results results/pmnist_main --out reports/pmnist_main --reference baseline
```

Kanonik ölçekli protokoller (`configs/pmnist_canonical.yaml`: 800 görev × 60.000 örnek, 3×2000 birim;
`configs/cifar_canonical.yaml`: 3000 görev) aynı komutlarla çalıştırılır; bu depodaki raporlanan sonuçlar CPU
bütçesiyle uyumlu küçültülmüş protokollerle elde edilmiştir (bkz. `docs/RAPOR.md`, "Hesaplama bütçesi").

## Depo yapısı

```
plasticity/
  data/        MNIST / CIFAR-100 yükleyicileri, görev akışları (pmnist, cifar100_binary, stationary)
  models/      ReparamLinear (standard | sin | tanh), MLP, aktivasyonlar
  methods/     baseline, weight_clipping, l2_init, continual_backprop, nap, ln_wd,
               shrink_perturb, upgd, parseval, scale_corrected
  metrics/     mekanizma ölçütleri (ağırlık, ölü birim, kerte, gradyan/Fisher Θ ve W koordinatlarında),
               özet ölçütler, eşleştirilmiş istatistik (bootstrap GA, permütasyon testi, Holm)
  training/    çevrimiçi eğitici, durağan kontrol, taze-model referansı
  analysis/    toplulaştırma, tablolar, şekiller
configs/       YAML yapılandırmalar (pilot ve kanonik protokoller)
suites/        yöntem × tohum ızgaraları
scripts/       run_suite.py, select_hparams.py, analyze.py, download_data.sh
tests/         pytest birim testleri
docs/          RAPOR.md (proje raporu), PROTOKOL.md
```
