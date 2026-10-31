# Yapılacaklar — danışman raporu (31 Ekim 2026) sonrası tez ve makale planı

Kaynak: "Tez Çalışması Ara Değerlendirme ve Öneri Raporu" (Prof. Dr. E. U. Küçüksille, 31.10.2026).
Durum sütunu: ✔ hazır · ◐ kısmen var · ✗ yeni. Hesap sütunu bu depodaki ölçümlere dayanır (4 çekirdek CPU);
12 çekirdekli dizüstünde süreler ~3 kat kısalır. Tüm koşular `scripts/run_suite.py` ile, checkpoint'ten devam eder.

## A. Hesap gerektirmeyen işler — 31 Ekim 2026'da tamamlandı (commit d503451)

| # | İş | Danışman maddesi | Durum | Çıktı |
|---|---|---|---|---|
| A1 ✔ | İddia ifadelerini düzelt: "periyodiklik belirleyici" → "sınırlılık tek başına yetmez; Jacobian sönümü…"; "tamamen korudu" → "800 görevde belirgin düşüş gözlenmedi"; "WC ile eşit" → "ayırt edilemeyecek kadar yakın" (TOST'a kadar) | 4.1, 4.3, 4.4 | ✗ | RAPOR §1, §13, §14.2, README |
| A2 ✔ | "Plastisite korunmuş" için operasyonel ölçüt tanımla (koruma oranı ≥ 0.99, eğim/100 görev ≥ −0.002, taze model farkı artmıyor) ve tüm modellere uygula | 4.3 | ◐ (ölçütler var, eşik yok) | RAPOR §2, tablo |
| A3 ✔ | Üçgen dalga – Weight Clipping eşdeğerlik testi: önceden belirlenmiş marj Δ = 0.005, eşleştirilmiş %95 GA, TOST (`plasticity/analysis/stats.py`'ye `tost_paired` ekle) | 4.4, deney 3 | ✗ | `scripts/equivalence.py`, RAPOR §14.2 |
| A4 ✔ | Donma oranı × görev × katman × doğruluk şekli (sat_frac/L0..L3 ve online_accuracy aynı eksende, sin/tanh/tri) | deney 5 | ◐ (veri var: tasks.jsonl) | `reports/ozet/6_donma_orani.png` |
| A5 ✔ | Üçgen dalga köşe noktaları: tam tanım (tri(x) = (2/π)·asin(sin(πx/2)), periyot 4, aralık [−1,1]), köşede türev kuralı (floor tabanlı: d = ±1, 0 olmaz), köşeye tam düşen parametre oranını ölç (|ang mod 2 − 1| < 1e-6) | 5.2 | ✗ | RAPOR §13 alt bölüm |
| A6 ✔ | Özgünlük ifadesini daralt; Dai vd. 2025 (Neuroshaper, üçgen dalga koordinat haritası) atıfı | 4.2 | ✗ | RAPOR §13, literatür |
| A7 ✔ | Teorik çerçeve bölümü: "pürüzsüz + sınırlı + periyodik ⇒ bir yerde türev sıfır" (ekstremum argümanı); ana soru: sınırlılık mı, Jacobian'ın sıfırdan uzak kalması mı | 5.1 | ✗ | RAPOR §9'a ek, tez Bölüm 3 |
| A8 ✔ | Tohum ifadesini netleştir: "15 koşu = 5 model × 3 tohum"; 3 tohumun pilot düzeyi olduğunu belirt | 4.5 | ✗ | RAPOR §14.2 |
| A9 ✔ | Telafili sinüs sonuçlarını danışmanın istediği biçimde öne çıkar: çarpımsal telafi (§8, §11: sınıra yığılma, çöküş) ve yansıma tipi telafi (§13, ε = 1: kırpma düzeyi) — H3'ün nedensel testi | deney 2, 6.1 | ✔ (var, görünür değil) | RAPOR §13 başına özet tablo |

## B. Ucuz deneyler (dizüstünde 1–2 gün)

| # | Deney | Danışman maddesi | Hesap | Paket |
|---|---|---|---|---|
| B1 ✔ | Üçgen dalga γ duyarlılığı: γ ∈ {1.0, 1.25, 1.5, 2.0, 3.0}, 200 görev × 5.000, 5 tohum | deney 4 | 2.9 saat (4 çekirdek) | `suites/pmnist_tri_gamma.yaml`, RAPOR §13.4, `reports/pmnist_tri_gamma_vs_main.png` |
| B2 ✔ | Üçgen dalga CIFAR-100 çevrimiçi akış, 5 tohum (γ = 1.0 ve 1.5; WC referansı ana paketten) | deney 6 | 1.3 saat (4 çekirdek) | `suites/cifar_tri.yaml`, RAPOR §7.4, `reports/cifar_tri_vs_main.png` |
| B3 ✔ | Yansıma sıklığı (`cross_frac`) metriği eklendi; CIFAR'da ölçüldü (RAPOR §7.4), PMNIST γ taramasında loglanıyor | 5.2 | — | `plasticity/metrics/mechanism.py` |

## C. Ana hesap paketi (dizüstünde ~1 hafta, burada ~2 hafta)

| # | Deney | Danışman maddesi | Hesap | Paket |
|---|---|---|---|---|
| C1 | 800 görev × 60.000: standart / sinüs / tanh / üçgen / WC, tohum 3–9 (7 ek tohum, 35 koşu) → toplam 10 eşleştirilmiş tohum | deney 1 (kritik) | ~650 CPU-saat | `suites/pmnist_fullstream.yaml` seeds: [0..9] |
| C2 | LN + Weight Decay baseline, 800 görev, 10 tohum | deney 7 | ~150 CPU-saat | aynı pakete `ln_wd` koşusu |
| C3 | Yansımalı sinüs (jacobian_floor ε = 1), 800 görev, 3 tohum — H3 nedensel testinin uzun ufuk hâli | deney 2 | ~65 CPU-saat | aynı pakete `sin_refl` |
| C4 | C1–C3 bitince: analyze.py, scaleup_tables.py, TOST, RAPOR §14.2 güncelle | — | — | reports/, RAPOR |

## D. İsteğe bağlı (GPU, tezden sonra da olabilir)

| # | Deney | Danışman maddesi | Not |
|---|---|---|---|
| D1 | Kapasite × ufuk matrisi: genişlik {100, 200, 400, 800, 1600} × görev {200, 400, 800}, standart + üçgen, 3 tohum | deney 8 ("yararlı") | GPU portu (vmap) + ~2 hafta |
| D2 | 3×2000 × 800 görev | deney 9 ("opsiyonel") | zorunlu değil |

## E. Tez metni (deneyler koşarken başlanır)

1. Giriş ve problem: plastisite kaybı, Dohare 2024, Lyle 2024/2025, Elsayed 2024 (Weight Clipping), Lillo & Cheney 2026.
2. Yöntem: genel çerçeve W = A·f(Θ); sinüs (öneri) ve tanh (kontrol); eşleştirilmiş başlangıç; H1–H3; teorik çerçeve (A7).
3. Deney düzeni: akışlar, ölçütler, operasyonel plastisite tanımı (A2), istatistik (eşleştirilmiş test, bootstrap GA, TOST).
4. Sonuçlar 1: küçültülmüş protokol (RAPOR §4–§11) — H1 kısmen, H2, H3 ölçümü.
5. Sonuçlar 2: mekanizmadan türetilen harita (§13), telafili sinüs (A9), kanonik uzunluk (§14.2 + C1), rejim bağımlılığı: Adam (§14.1) ve genişlik (§14.3).
6. Tartışma: sınırlılık vs Jacobian; özgünlük (A6); sınırlılıklar (§12).
7. Sonuç ve gelecek çalışma (§14.4, D1).
Kronoloji korunur: sinüs ana hipotez testi, üçgen dalga mekanizmadan türetilen ikinci aşama.

## F. Makale (tezden çıkarılan, İngilizce, 8–12 sayfa)

- Mesaj: "Element-wise boundedness alone is insufficient…; a non-saturating piecewise-linear periodic map preserves bounded weights while maintaining update magnitude." (danışman §7.2)
- Başlık adayı: "Beyond Bounded Weights: Non-Saturating Periodic Reparameterization for Preserving Plasticity in Continual Learning".
- Şekiller: (1) 800 görev eğrileri 10 tohum, (2) mekanizma: donma oranı + adım çarpanı, (3) telafili sinüs (H3 nedensel), (4) tri vs WC eşdeğerlik + γ duyarlılığı; tablo: paired farklar + TOST.
- Ek: Adam rejimi, geniş ağ sınır koşulu, CIFAR.
- Hedef: CoLLAs / TMLR / RLC; danışman ortak yazar. Yazım, C1 tamamlanınca (hakemler 10 tohum ve TOST'u isteyecek).

## Sıra ve takvim

1. Hafta 1: dizüstü kurulumu; B1–B2 başlat; C1–C3 paketini başlat; A1–A10 yazım işleri; danışmana mesaj.
2. Hafta 2: C paketi biter; C4 analiz; RAPOR ve README güncelle; tez Bölüm 1–3 taslağı.
3. Hafta 3–4: tez Bölüm 4–7; makale taslağı; D1 kararı.
