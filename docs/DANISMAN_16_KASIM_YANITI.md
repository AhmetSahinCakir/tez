# 16 Kasım 2026 danışman raporuna yanıt: durum çizelgesi

Kaynak: "Danışman Değerlendirme ve Yönlendirme Raporu" (Prof. Dr. Ecir Uğur Küçüksille, 16.11.2026).
Bu dosya, raporun 7. bölümündeki teslim listesinin durumunu ve deney sürüm kaydını tutar. Eski tablolar
silinmez; yeni sonuçlar sürüm notuyla eklenir.

## Teslim listesi

| # | İstenen çıktı | Durum | Nerede |
|---|---|---|---|
| 1 | Tek etkenli yansımalı sinüs deneyi (800 görev, 10 eşleştirilmiş tohum, aynı γ, lr, başlangıç ve akış) | ✔ 20 Kasım'da bitti (10/10 tohum, hata yok); sonuç aşağıda "Madde 1" | `suites/pmnist_fullstream.yaml` → `sin_refl_matched`; `results/pmnist_fullstream/sin_refl_matched/`; `reports/pmnist_fullstream/causal_tables.md`; RAPOR §14.2 |
| 2 | Matematiksel yöntem ve kod eki | ✔ 19 Kasım'da ayrı ek olarak yazıldı | `docs/EK_MATEMATIK_VE_KOD.md`; özet aşağıda "Madde 2" |
| 3 | İstatistiksel denetim paketi (tohum bazlı CSV, TOST kodu ve gerekçesi) | ✔ 19 Kasım'da tek paket hâline getirildi; süren koşular bitince yeniden üretilecek | `reports/denetim/` (`DENETIM.md`, `tohum_bazli.csv`, `tost.csv`), `scripts/denetim_paketi.py` |
| 4 | Genelleme ve sınırlılıklar eki (CIFAR-100, Adam, geniş ağ, γ) | ✔ raporda var; tezde ayrı bölüm olacak | RAPOR §7.4, §13.4, §14.1, §14.3; makale ekleri C–F |
| 5 | Tez taslağı | ✗ sırada | — |
| 6 | Makale iskeleti | ✔ ilk tam taslak | `paper/main.tex`, `paper/main.pdf` |

## Madde 1: tek etkenli deney tasarımı

* Karşılaştırılan iki model: `sin` (mevcut, 10 tohum) ve `sin_refl_matched` (yeni, 10 tohum).
* Aynı olanlar: γ = 1.5, öğrenme oranı 0.003, SGD, 3×100 ağ, başlangıç ağırlıkları (aynı tohum → aynı W₀),
  görev akışı (aynı tohum → aynı permütasyonlar ve örnek sırası), 800 görev × 60.000 örnek.
* Tek fark: geri geçişte kullanılan türev (`model.jacobian_floor`: 0 → 1).
* İki koşunun komut satırları `python3 scripts/run_suite.py suites/pmnist_fullstream.yaml --dry-run` ile
  yan yana görülebilir; yalnızca `--set model.jacobian_floor=1.0` farklıdır.
* Önceki 3 tohumluk `sin_refl` koşuları (γ = 1.0, lr = 0.01) silinmedi; "her varyant kendi en iyi ayarıyla"
  ek analizi olarak kalacak.

**Sonuç (20 Kasım 2026, 10 eşleştirilmiş tohum).**

| Ölçüt | Sinüs | Yansımalı sinüs (aynı ayar) | Fark (eşleştirilmiş, %95 GA) |
|---|---|---|---|
| Normalize AUC | 0.818 | 0.922 | +0.1045 [+0.1027, +0.1063] |
| Son pencere | 0.720 | 0.921 | +0.201 [+0.192, +0.211] |
| Koruma oranı | 0.792 | 0.997 | +0.205 [+0.195, +0.216] |
| Taze model farkı | +0.212 | −0.002 | −0.214 [−0.227, −0.201] |
| Doyma payı (son) | 0.78 | 0.08 | — |
| Koruma ölçütünü sağlayan tohum | 0/10 | 10/10 | — |

* Fark on tohumun onunda da pozitiftir (+0.0997 ile +0.1099 arası); eşleştirilmiş t-testi p < 0.001, işaret
  değiştirme permütasyon testi p = 0.002 (10 çiftle en küçük değer).
* Yansımalı sinüs üçgen dalganın 0.0166 [0.0164, 0.0169] altında kalır; bu fark tek etkenli değildir.
* Sınırlar: kural vekil gradyandır; γ ve öğrenme oranı sinüs için seçilmiş değerlerdir; tek veri akışı, küçük
  ağ, SGD.

## Madde 2: geri yayılım matematiği ve kod

**Tanım (kullanılan ölçek).** W = A·f(Θ/A). Gerçek türev J = ∂W/∂Θ = f′(Θ/A). (Raporunuzdaki W = A·sin(θ)
ölçeğinde J = A·cos θ ve adım çarpanı A²·cos²θ olur; A² katman başına sabittir. Kodda bu `theta_scale='unit'`
seçeneğidir; ana deneyler `'amplitude'` ölçeğini kullanır.)

**Birinci derece ağırlık adımı.** Geri geçişte J̃ kullanılırsa ΔΘ = −η·J̃·g ve ΔW ≈ −η·J·J̃·g (g = ∂L/∂W).

| Model | J (gerçek) | J̃ (geri geçişte) | J·J̃ (ağırlık adımı çarpanı) | \|J̃\| (parametre adımı çarpanı) |
|---|---|---|---|---|
| Sinüs | cos | cos | cos² | \|cos\| |
| Yansımalı sinüs (ε = 1) | cos | sign(cos) | **\|cos\|** | **1** |
| Üçgen dalga | ±1 (köşeler dışında) | ±1 | 1 | 1 |

**Düzeltme.** Raporunuzdaki uyarı doğrudur: yansımalı sinüste ağırlık adımı hâlâ |cos| ile ölçeklenir;
"Jacobian sönümü tamamen kaldırıldı" ifadesi yanlıştır ve makaleden çıkarıldı. Doğru ifade: ağırlık adımının
sönümü cos²'den |cos|'a *zayıflar*; kaldırılan şey serbest parametrenin (Θ) adımındaki sönümdür, yani
parametre tepe noktasında hapsolmaz, tepeyi aşar ve ağırlık iç bölgeye geri döner. Bu kural gerçek türev
değil, düz-geçişli (surrogate) bir gradyandır (Bengio vd., 2013; Yin vd., 2019).

**Sayısal doğrulama** (`tests/test_update_rule.py`; tek SGD adımı, η = 10⁻⁴, A = 0.3; ölçülen ΔW/(−η·g)):

| Θ/A | cos² | \|cos\| | Sinüs | Yansımalı sinüs | Üçgen dalga |
|---|---|---|---|---|---|
| 0 | 1.000 | 1.000 | 1.0000 | 1.0000 | 1.0000 |
| 0.25π | 0.500 | 0.707 | 0.5001 | 0.7072 | 1.0000 |
| 0.40π | 0.095 | 0.309 | 0.0955 | 0.3092 | 1.0000 |
| 0.45π | 0.024 | 0.156 | 0.0245 | 0.1566 | 1.0000 |
| 0.49π | 0.001 | 0.031 | 0.0010 | 0.0316 | 1.0000 |
| 0.50π | 0.000 | 0.000 | 0.0000 | 0.0002 | 1.0000 |

**Kod ve testler.**
* İleri/geri geçiş: `plasticity/models/reparam.py`, `_BoundedMap` (tek özel autograd fonksiyonu).
* Sayısal türev kontrolü (sonlu farklar): `tests/test_reparam_nondamped.py::test_gradcheck_away_from_kinks`,
  `tests/test_reparam.py::test_jacobian_matches_autograd`.
* Surrogate kuralı: `tests/test_reparam_nondamped.py::test_jacobian_floor_backward`, `::test_reflection_vs_freeze`.
* Gerçek ΔW ile birinci derece formülün karşılaştırması: `tests/test_update_rule.py` (yeni, 5 test).
* Üçgen dalga tanımı ve köşe kuralı: RAPOR §13.2 (periyot 4, değer aralığı [−1, 1], köşelerde sağ türev,
  köşeye düşen parametre oranı: 99.710 parametrenin 0'ı tam köşede, 2'si 10⁻⁶·A içinde);
  `tests/test_reparam_nondamped.py::test_tri_closed_form_matches_reference_bitwise`.
* "Türevi her yerde birdir" ifadesi makalede "köşe noktaları dışında birim büyüklükte" olarak düzeltildi.

## Madde 3: yapılanlar (19 Kasım 2026, `reports/denetim/`)

1. Tohum bazlı tek CSV: model × tohum × (AUC, erken/son pencere, koruma, eğim, taze model farkı, ölçüt 1/2/3).
2. TOST: iki tek yönlü testin t istatistikleri ve p değerleri ayrı ayrı; %90 GA; Δ = 0.005 ve 0.01.
3. Birincil ölçüt: normalize AUC; diğer üçü ikincil; çoklu çıkarım için Bonferroni/Holm notu.
4. Marj gerekçesi; eşdeğerlik kapsamının sınırı (SGD, Permuted MNIST, 3×100, bu ölçütler).
5. Ortam kaydı: Python/PyTorch sürümleri, donanım, komutlar; hiperparametre arama kayıtları
   (`reports/pmnist_dev_hparams.md`, `suites/selected_pmnist.yaml`).

## Deney sürüm kaydı

| Tarih | Değişiklik | Neden |
|---|---|---|
| 14.11.2026 | 800 görevlik paket 3 → 10 tohum; LN + WD (10) ve yansımalı sinüs (3) eklendi | 31 Ekim raporu, C paketi |
| 16.11.2026 | `sin_refl_matched` (10 tohum) eklendi; mevcut koşulara dokunulmadı | 16 Kasım raporu, madde 2.1 |
| 16.11.2026 | Makalede yansımalı sinüsün matematiksel tanımı düzeltildi (ağırlık adımı \|cos\| ile ölçeklenir) | 16 Kasım raporu, madde 2.2 |
| 19.11.2026 | 200 görevlik pakette bütün karşılaştırma yöntemleri 10 tohuma çıkarılıyor (eski tohumlar yeniden koşulmadı); girdi standardizasyonu testi ve Random Label MNIST pilotu eklendi | Makale sınırlılıkları; ikinci veri akışı |
| 20.11.2026 | `sin_refl_matched` 10 tohum tamamlandı; RAPOR §14.2, makale taslağı (nedensel test bölümü, Tablo) ve denetim paketi güncellendi; 3 tohumluk eski sonuç korunmuştur | 16 Kasım raporu, madde 2.1 |
| 20.11.2026 | 200 görevlik pakette bütün yöntemler 10 tohuma tamamlandı (RAPOR §6.3). Adam rejiminde beş temel model 10 eşleştirilmiş tohumla tekrarlandı (RAPOR §14.1 güncellemesi, makale Adam eki, denetim paketi): sinüs kırpmayı ve üçgen dalgayı 0.034 geçiyor (10/10 tohum); 3 tohumda görülen "üçgen dalga ile kırpma son pencerede ayrışıyor" gözlemi 10 tohumda doğrulanmadı ve düzeltildi. Random Label MNIST geliştirme taramasında (36 ayar, tek tohum) bütün modeller şans düzeyine düştü; 10 tohumluk ana deney ve L2 Init kontrol deneyi sürüyor | 16 Kasım raporu, bölüm 4 (en az 10 tohum); ikinci veri akışı |
