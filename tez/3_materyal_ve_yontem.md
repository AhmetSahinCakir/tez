# 3. MATERYAL VE YÖNTEM

> Taslak (19 Kasım 2026). Sayılar ve ayarlar depodaki yapılandırma dosyalarından alınmıştır (her alt bölümün
> sonunda kaynağı yazılıdır). Formüllerin türetmesi ve kod eşlemesi `docs/EK_MATEMATIK_VE_KOD.md` dosyasındadır;
> tezde ek olarak verilecektir. Cümleler tez yazarı tarafından yeniden yazılacaktır.

## 3.1. Sınırlı ağırlık yeniden parametrizasyonu

n girişli tam bağlantılı bir katmanın ağırlıkları [−b, b] aralığında düzgün dağılımdan başlatılır; b = √(6/n)
değeri He vd. (2015) tarafından önerilen 2/n varyansını verir. Genlik A = γ·b (γ ≥ 1) olarak sabitlenir ve
her ağırlık ile her yanlılık serbest bir Θ parametresine bağlanır:

    W = A · f(Θ / A),        f: ℝ → [−1, 1].                                  (3.1)

Üç harita incelenmiştir:

| Harita | f(x) | f′(x) |
|---|---|---|
| Sinüs | sin x | cos x |
| Tanh | tanh x | 1 − tanh² x |
| Üçgen dalga | tri(x) = (2/π)·arcsin(sin(πx/2)) | ±1 (köşe noktaları dışında) |

Üçgen dalga 4 periyotludur, [−1, 1] aralığında özdeşlik fonksiyonuna eşittir ve ±1'de yansır. Bütün modeller
aynı etkin ağırlıklardan başlar: W₀ bir kez çekilir ve Θ₀ = A·f⁻¹(W₀/A) alınır. (Kod:
`plasticity/models/reparam.py`.)

## 3.2. Adım çarpanı

L kayıp, g = ∂L/∂W ve η öğrenme oranı olmak üzere, Θ üzerindeki bir gradyan adımı etkin ağırlığı birinci
dereceden

    ΔW ≈ −η · f′(Θ/A)² · g                                                    (3.2)

kadar değiştirir. f′² ∈ [0, 1] çarpanı her parametre için ayrı bir öğrenme oranı çarpanıdır ve tezde adım
çarpanı olarak adlandırılmıştır. Sinüs haritasında bu çarpan cos²(Θ/A) = 1 − (W/A)² değerine eşittir: ağırlık
sınıra yaklaştıkça adımı küçülür. Türevlenebilir bir fonksiyon sınır değerine ulaştığı noktada en büyük ya da
en küçük değerini alır ve türevi orada sıfırdır; bu nedenle bir harita aynı anda pürüzsüz, sınırına ulaşan ve
türevi sıfırdan uzak kalan bir harita olamaz. Üçgen dalga pürüzsüzlükten vazgeçerek bu kısıtı aşar: sınırlıdır
ve köşeler dışında |f′| = 1'dir.

## 3.3. Yansıma ve izdüşüm

Üçgen dalgada (3.2) aralığın içinde ΔW = −η·g verir. Ağırlığı A + δ'ya (0 < δ < 2A) taşıyacak bir adım onu
A − δ'da bırakır (yansıma); ağırlık kırpma aynı adımı A'ya eşler (izdüşüm). Tek adımdan sonra iki sonuç en çok
δ ≤ η·|g| kadar ayrılır. Adımlar aralığın genişliğine göre küçük olduğunda iki yöntemin yakın yollar izlemesi
beklenir.

## 3.4. Yansımalı sinüs: tek etkenli müdahale

Adım çarpanının plastisite kaybının nedeni olup olmadığını sınamak için ileri harita sinüs olarak bırakılmış,
yalnızca geri geçişte kullanılan türev değiştirilmiştir: cos(Θ/A) yerine sign(cos(Θ/A)) kullanılır. Bu durumda
serbest parametrenin adımı sönümlenmez, ağırlık adımının çarpanı cos²'den |cos|'a zayıflar. Bu kural gerçek
türev değil, düz-geçişli türden bir vekil gradyandır (Bengio vd., 2013). Karşılaştırılan iki modelde γ, öğrenme
oranı, başlangıç ağırlıkları ve görev akışı aynıdır; tek fark geri türevdir. (Ayar: `model.jacobian_floor: 1.0`.)

## 3.5. Veri akışları

**Online Permuted MNIST** (Dohare vd., 2024). Her görev, MNIST eğitim görüntülerinin (LeCun vd., 1998)
piksellerine yeni bir rastgele permütasyon uygular. Görüntüler teker teker (yığın boyutu 1) ve yalnızca bir kez
sunulur; görev sınırı ve görev kimliği ağa verilmez. İki ölçek kullanılmıştır:

| Ölçek | Görev sayısı | Görev başına örnek | Toplam güncelleme |
|---|---|---|---|
| Kısa akış | 200 | 5.000 | 10⁶ |
| Kanonik uzunluk | 800 | 60.000 | 4,8 × 10⁷ |

**Ardışık CIFAR-100 ikili sınıflandırma** (Chen ve Zhang, 2026). Her görevde 100 sınıftan rastgele iki sınıf
seçilir; görüntüler gri seviyeye çevrilir ve 1.024 boyutlu vektör olarak verilir. Ölçüt, görevin iki sınıfına
ait test görüntülerindeki doğruluktur.

**Random Label MNIST** (Kumar vd., 2025). 1.200 sabit MNIST görüntüsüne her görevde yeniden rastgele etiket
atanır; ağ her görevde 400 tur eğitilir (yığın boyutu 16, 50 görev). Ölçüt, görev boyunca bütün yığınlarda
güncellemeden önce ölçülen ortalama doğruluktur. [Pilot aşamasında; ana deney kurulursa bu paragraf
güncellenecek.]

Akışlar (tohum, görev) çiftinin belirlenimci fonksiyonudur: aynı tohum aynı permütasyonları ve aynı örnek
sırasını verir. (Kod: `plasticity/data/streams.py`.)

## 3.6. Ağ ve eniyileyici

Üç gizli katmanlı, katman başına 100 birimli, ReLU aktivasyonlu çok katmanlı algılayıcı kullanılmıştır.
Ana deneylerde eniyileyici momentumsuz stokastik gradyan inişidir. İki ek rejim ayrı deneylerle incelenmiştir:
Adam eniyileyicisi (öğrenme oranı 0,001) ve geniş ağ (üç gizli katman, 2.000 birim).

## 3.7. Karşılaştırılan modeller ve yöntemler

Ana karşılaştırma beş modelden oluşur: standart ağ; sinüs, tanh ve üçgen dalga haritalarıyla bütün ağırlıkların
ve yanlılıkların yeniden parametrizasyonu; ağırlık kırpma (Elsayed vd., 2024). Ek olarak yayımlanmış şu
yöntemler aynı akışlarda çalıştırılmıştır:

| Yöntem | Kaynak | Müdahale |
|---|---|---|
| L2 Init | Kumar vd., 2025 | Parametreleri başlangıç değerlerine çeken ceza terimi |
| Continual Backprop | Dohare vd., 2024 | Düşük yararlılıklı birimlerin yeniden başlatılması |
| Normalize-and-Project | Lyle vd., 2024 | Katman normalizasyonu ve ağırlık normunun izdüşümü |
| Katman normalizasyonu + ağırlık sönümü | Lyle vd., 2025 | Katman normalizasyonu ve ağırlık sönümü |
| Shrink and Perturb | Ash ve Adams, 2020 | Ağırlıkların küçültülüp gürültü eklenmesi |
| UPGD | Elsayed ve Mahmood, 2024 | Yararlılıkla ölçeklenen pertürbasyonlu gradyan |
| Parseval düzenlileştirmesi | Chung vd., 2024 | Ağırlık matrislerinin ortogonalliğe yaklaştırılması |

Yöntemler kaynak makalelerdeki tanımlara göre yazılmıştır; kaynak kodlarla birebir karşılaştırma
yapılmamıştır (bu bir sınırlılıktır).

## 3.8. Hiperparametre seçimi

Bütün değerler, değerlendirme akışlarından farklı bir rastgele tohumla üretilen ayrı bir geliştirme akışında
(100 görev, tek tohum) seçilmiş ve değerlendirme sırasında değiştirilmemiştir. Her yönteme aynı arama bütçesi
verilmiştir (5 öğrenme oranı ya da 4 öğrenme oranı × 2 yönteme özgü değer). Seçim ölçütü AUC'dir. Seçilen
değerler:

| Model | Öğrenme oranı | Diğer |
|---|---|---|
| Standart ağ | 0,003 | — |
| Sinüs, tanh | 0,003 | γ = 1,5 |
| Üçgen dalga | 0,003 | γ = 1 |
| Ağırlık kırpma | 0,003 | κ = 1 |

(Kaynak: `suites/pmnist_dev.yaml`, `suites/selected_pmnist.yaml`, `reports/pmnist_dev_hparams.md`.)
Seçimin tek tohumlu kısa bir akışta yapılmış olması bir sınırlılıktır.

## 3.9. Ölçütler

**Performans ölçütleri**

* Çevrimiçi doğruluk: bir görevde, güncellemeden önce doğru sınıflandırılan görüntülerin oranı.
* AUC: çevrimiçi doğruluğun bütün görevler üzerindeki ortalaması (birincil ölçüt).
* Erken ve son pencere: ilk ve son %10 görevin ortalaması.
* Koruma oranı: son pencere / erken pencere.
* Taze model farkı: son görevde sıfırdan başlatılan ağın doğruluğu eksi sürekli eğitilen ağın doğruluğu.
  Pozitif değer, sürekli eğitilen ağın başka hiçbir görev görmemiş bir ağdan kötü olduğunu gösterir.

**Plastisitenin korunması için işlemsel ölçüt.** Bir koşu, şu üç koşul birlikte sağlanırsa plastisiteyi
korumuş sayılır: (1) koruma oranı ≥ 0,99; (2) doğruluk eğrisinin eğimi ≥ −0,002 / 100 görev; (3) taze model
farkı ≤ 0,01.

**Mekanizma ölçütleri.** Ortalama adım çarpanı ⟨f′²⟩; doymuş parametre oranı (|f′| < 0,1); ortalama ağırlık
büyüklüğü; gerçek ağırlık adımının normu; ölü birim oranı; son gizli katman temsilinin etkin kertesi (Roy ve
Vetterli, 2007).

## 3.10. İstatistiksel değerlendirme

Deney birimi tohumdur. Bir tohum hem başlangıç ağırlıklarını hem görev akışını belirler; modeller aynı
akışlarda karşılaştırıldığı için bütün farklar eşleştirilmiştir. Ortalamalar tohumlar üzerinden %95 güven
aralığıyla, farklar eşleştirilmiş testlerle verilmiştir. Birden fazla karşılaştırma yapıldığında Holm (1979)
düzeltmesi uygulanmıştır.

İki yöntem arasında anlamlı fark bulunmaması eşitlik kanıtı değildir. Bu nedenle üçgen dalga ile ağırlık
kırpma için iki tek yönlü test (TOST; Schuirmann, 1987) uygulanmıştır. Eşdeğerlik marjı Δ = 0,005 olarak,
kanonik uzunluktaki deneyin on tohumu tamamlanmadan önce belirlenmiştir; ikincil marj 0,01'dir. Farkın %90
güven aralığı (−Δ, +Δ) içinde kalıyorsa eşdeğerlik ilan edilir. Ayrıntılar `reports/denetim/DENETIM.md`
dosyasındadır.

## 3.11. Yazılım, donanım ve yeniden üretilebilirlik

Deneyler Python ve PyTorch ile yazılmış, yapılandırma dosyalarıyla yönetilen bir altyapıda yürütülmüştür.
Her koşu kendi yapılandırmasını kaydeder ve görev sınırında alınan denetim noktasından birebir devam edebilir.
Bütün koşular 4 çekirdekli bir işlemcide, grafik işlemci olmadan çalıştırılmıştır. Sürüm ve donanım kaydı
`reports/denetim/DENETIM.md` dosyasının son bölümündedir.
