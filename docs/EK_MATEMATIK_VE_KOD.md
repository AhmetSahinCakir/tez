# Ek: Matematiksel yöntem ve kod eşlemesi

Danışman raporu (16 Kasım 2026), teslim listesi madde 2. Bu ek, yöntemin bütün formüllerini tek yerde toplar,
her formülün kodda nerede olduğunu ve hangi testle doğrulandığını gösterir. Sayısal tablolar bu belge yazılırken
kodun kendisiyle yeniden hesaplanmıştır (19 Kasım 2026).

## 1. Gösterim

| Simge | Anlamı |
|---|---|
| n | Katmanın giriş sayısı |
| b = √(6/n) | Başlangıç dağılımının sınırı (Kaiming-düzgün; W₀ ~ U(−b, b), varyans 2/n) |
| γ ≥ 1 | Sınır genişliği katsayısı |
| A = γ·b | Genlik (katmana özgü, sabit) |
| Θ | Eniyileyicinin güncellediği serbest parametre |
| W | Ağın kullandığı (etkin) ağırlık |
| L, g = ∂L/∂W | Kayıp ve etkin ağırlığa göre gradyan |
| η | Öğrenme oranı |

## 2. İleri harita

Her ağırlık ve her yanlılık, serbest bir parametrenin sınırlı bir fonksiyonudur:

    W = A · f(Θ / A),        f: ℝ → [−1, 1]                                   (1)

Dolayısıyla Θ ne olursa olsun |W| ≤ A'dır; ayrıca bir izdüşüm adımı gerekmez. İncelenen üç harita:

| Harita | f(x) | f′(x) | Sınıra ulaşır mı? |
|---|---|---|---|
| Sinüs | sin x | cos x | Evet (x = π/2 + kπ) |
| Tanh | tanh x | 1 − tanh² x | Hayır (yalnızca yaklaşır) |
| Üçgen dalga | tri(x) | ±1 (köşeler dışında) | Evet (x = 1 + 2k) |

Kodda ana deneyler bu ölçeği kullanır (`theta_scale: amplitude`). W = A·sin(Θ) biçimindeki ölçekte
(`theta_scale: unit`) türev A·cos Θ olur ve aşağıdaki adım çarpanına katman başına sabit bir A² çarpanı eklenir;
bu, sınırlılıkla ilgisi olmayan bir öğrenme oranı farkıdır ve ana deneylerde bu yüzden kullanılmamıştır.

## 3. Başlangıcın eşleştirilmesi

Bütün modeller aynı etkin ağırlıklardan başlar: W₀ bir kez çekilir ve

    Θ₀ = A · f⁻¹(W₀ / A)                                                     (2)

alınır (sinüs için arcsin, tanh için artanh, üçgen dalga için özdeşlik; |W₀| ≤ b ≤ A olduğundan tanımlıdır).
Böylece ilk adımdaki fark yalnızca haritadan gelir. Aynı tohum aynı W₀'ı ve aynı görev akışını verir.

## 4. Geri yayılım ve birinci derece ağırlık adımı

Zincir kuralı: ∂L/∂Θ = (∂L/∂W)·(∂W/∂Θ) = g · f′(Θ/A). SGD adımı ΔΘ = −η · f′(Θ/A) · g'dir. Etkin ağırlıktaki
değişim, birinci derece Taylor açılımıyla:

    ΔW ≈ f′(Θ/A) · ΔΘ = −η · f′(Θ/A)² · g                                    (3)

f′² ∈ [0, 1] çarpanı her parametre için ayrı bir öğrenme oranı çarpanıdır; metinde **adım çarpanı** denir.

| Harita | Adım çarpanı f′² | Ağırlık cinsinden |
|---|---|---|
| Sinüs | cos²(Θ/A) | 1 − (W/A)² |
| Tanh | (1 − tanh²(Θ/A))² | (1 − (W/A)²)² |
| Üçgen dalga | 1 (köşeler dışında) | 1 |

Sonuç: sinüs ve tanh haritalarında bir ağırlık sınıra yaklaştıkça adımı küçülür. Sinüste çarpan sonlu bir Θ
değerinde sıfır olur; tanh'ta sıfıra hiç ulaşmaz ama |Θ| büyüdükçe küçülmeyi sürdürür.

## 5. Pürüzsüz ve sınırlı bir haritada türev neden sıfırlanır?

**İddia.** f türevlenebilir olsun ve bir x* noktasında sınır değerine ulaşsın (f(x*) = 1 veya −1). O zaman
f′(x*) = 0'dır.

**Gerekçe.** f'nin değerleri [−1, 1] aralığındadır, dolayısıyla x* bir en büyük (veya en küçük) noktadır.
Türevlenebilir bir fonksiyonun iç noktadaki ekstremumunda türev sıfırdır (Fermat teoremi).

**Sonuç.** Bir harita aynı anda (i) türevlenebilir, (ii) sınırına ulaşan ve (iii) türevi sıfırdan uzak kalan bir
harita olamaz. Sinüs (i) ve (ii)'yi sağlar, (iii)'ü sağlamaz. Tanh sınırına ulaşmaz ama türevi yine sıfıra
yaklaşır. Üçgen dalga (ii) ve (iii)'ü sağlar, karşılığında köşe noktalarında türevlenebilirlikten vazgeçer.

## 6. Üçgen dalga

    tri(x) = (2/π) · arcsin(sin(πx/2))                                        (4)

* Periyot 4; değer aralığı [−1, 1]; [−1, 1] üzerinde tri(x) = x (bu aralıkta katman, standart katmanla aynıdır).
* Yükselen kollarda türev +1, alçalan kollarda −1; köşe noktalarında (x = ±1, ±3, …) türev tanımsızdır.
* Kodda trigonometrik fonksiyon kullanılmaz: u = x + 1, y = u − 4·⌊u/4⌋, tri(x) = 1 − |y − 2|,
  türev = 1 − 2·⌊y/2⌋. Bu kural köşe noktasında sağ türevi verir; türev hiçbir noktada 0 değildir.
* Köşeye tam düşen parametre uygulamada görülmemiştir (RAPOR §13.2: 99.710 parametrenin 0'ı tam köşede).

## 7. Yansıma ve izdüşüm: üçgen dalga ile Weight Clipping arasındaki fark

Aralığın içinde iki yöntem de aynı adımı atar: ΔW = −η·g. Fark yalnızca sınırı aşan adımda ortaya çıkar.
Adımdan önce |W| ≤ A olsun ve adım ağırlığı A + δ'ya taşısın (0 < δ < 2A):

| Yöntem | Adımdan sonraki ağırlık | Ad |
|---|---|---|
| Üçgen dalga | A − δ | yansıma |
| Weight Clipping (κ = γ) | A | izdüşüm |

İki sonuç arasındaki fark δ'dır. Adımdan önce W ≤ A olduğundan δ = W − η·g − A ≤ η·|g|, yani **tek adımdan sonra
iki yöntemin farkı adımın kendisinden büyük olamaz.** İki yöntem de gradyanı ölçeklemez. Bu, birkaç adım
sonrası için bir eşitlik kanıtı değildir; yalnızca adımlar aralığın genişliğine göre küçükken iki yöntemin yakın
yollar izlemesinin beklendiğini söyler. Deneyde gözlenen de budur (SGD, öğrenme oranı 0,003). Adam ile adım
aralığa göre büyük olabilir; orada bu beklenti geçerli değildir.

## 8. Yansımalı sinüs (tek etkenli nedensel test)

İleri harita sinüsle aynıdır; yalnızca geri geçişte kullanılan türev değişir:

    J = cos(Θ/A)   (gerçek türev),        J̃ = sign(J) · max(|J|, ε)   (geri geçişte kullanılan)

ΔΘ = −η·J̃·g ve ΔW ≈ J·ΔΘ = −η·J·J̃·g olur. ε = 1 için J̃ = sign(cos) ve

| Model | Parametre adımı çarpanı \|J̃\| | Ağırlık adımı çarpanı J·J̃ |
|---|---|---|
| Sinüs (ε = 0) | \|cos\| | cos² |
| Yansımalı sinüs (ε = 1) | 1 | \|cos\| |
| Üçgen dalga | 1 | 1 |

Dikkat edilecek iki nokta: (a) yansımalı sinüste ağırlık adımındaki sönüm **kalkmaz**, cos²'den |cos|'a
zayıflar; kalkan şey serbest parametrenin adımındaki sönümdür (parametre tepe noktasında takılmaz, tepeyi aşar
ve ağırlık iç bölgeye döner). (b) J̃ gerçek türev değildir; düz-geçişli türden bir vekil gradyandır. Bu yüzden
deney "ileri harita aynı, yalnızca geri türev farklı" biçiminde tek etkenli bir müdahaledir, yeni bir yöntem
önerisi değildir.

## 9. Sayısal doğrulama

Tek SGD adımı, η = 10⁻⁴, A = 0,3, g = 1; ölçülen ΔW / (−η·g). Değerler kodun kendisiyle hesaplanmıştır.

| Θ/A | cos² | \|cos\| | Sinüs | Yansımalı sinüs | Üçgen dalga (iç bölge) |
|---|---|---|---|---|---|
| 0 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 |
| 0,25π | 0,5000 | 0,7071 | 0,5001 | 0,7072 | 1,0000 |
| 0,40π | 0,0955 | 0,3090 | 0,0955 | 0,3092 | 1,0000 |
| 0,45π | 0,0245 | 0,1564 | 0,0245 | 0,1566 | 1,0000 |
| 0,49π | 0,0010 | 0,0314 | 0,0010 | 0,0316 | 1,0000 |
| 0,50π | 0,0000 | 0,0000 | 0,0000 | 0,0002 | 1,0000 |

Yansıma ve izdüşüm (A = 0,3): W = 0,29 iken +0,03'lük adım üçgen dalgada 0,28, kırpmada 0,30 verir (fark 0,02 ≤
adım 0,03); W = 0,2999 iken +0,0005'lik adımda 0,2996 ve 0,30 (fark 0,0004 ≤ 0,0005).

## 10. Formül – kod – test eşlemesi

| Konu | Kod | Test |
|---|---|---|
| İleri harita (1), türevler | `plasticity/models/reparam.py`, `_BoundedMap.forward` | `tests/test_reparam.py::test_effective_weights_are_bounded_always`, `::test_jacobian_matches_autograd` |
| Başlangıç eşleştirmesi (2) | `ReparamLinear.inverse` | `tests/test_reparam.py::test_matched_initialisation_across_modes`, `::test_inverse_forward_round_trip` |
| Geri geçiş, sonlu farklarla türev kontrolü | `_BoundedMap.backward` | `tests/test_reparam_nondamped.py::test_gradcheck_away_from_kinks` |
| Birinci derece adım (3) | — | `tests/test_update_rule.py::test_sine_step_is_cos_squared` |
| Üçgen dalga (4), kapalı biçim | `_BoundedMap.forward` (`mode == "tri"`) | `tests/test_reparam_nondamped.py::test_tri_values_and_period`, `::test_tri_closed_form_matches_reference_bitwise` |
| Üçgen dalgada sönümsüz adım | — | `tests/test_update_rule.py::test_triangle_step_is_undamped_away_from_corners` |
| Yansımalı sinüs (§8) | `_BoundedMap.backward` (`floor > 0`), ayar: `model.jacobian_floor` | `tests/test_reparam_nondamped.py::test_jacobian_floor_backward`, `::test_reflection_vs_freeze`; `tests/test_update_rule.py::test_reflected_sine_step_is_abs_cos_not_one` |
| Weight Clipping | `plasticity/methods/weight_clipping.py` | `tests/test_methods_wc_l2init.py` |
| Adım çarpanı ve doyma oranı ölçümü | `plasticity/metrics/mechanism.py` | `tests/test_metrics.py` |

Bütün testler `python3 -m pytest -q tests` komutuyla çalışır.
