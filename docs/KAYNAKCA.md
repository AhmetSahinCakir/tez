# Kaynakça doğrulama kaydı (makale ve tez)

Tam kayıtlar: `paper/references.bib` (BibTeX, 49 kaynak; hepsi makalede atıflı). Doğrulama tarihi: 16 Kasım 2026.

**Yöntem.** Her kaynak birincil bir kaynaktan tek tek bulundu; künye (başlık, yazarlar, yer, yıl, sayfa) o
kaynaktaki kayıtla karşılaştırıldı ve makalede o kaynağa atfedilen ifade, özet ya da tam metin okunarak
kontrol edildi. Hiçbir künye hafızadan bırakılmadı.

Künye kaynakları: arXiv API (40 kayıt), PMLR sayfaları, NeurIPS ve ICLR resmî bildiri siteleri (BibTeX
kayıtları), Reinforcement Learning Journal sayfası, Crossref (DOI kayıtları), yayıncı PDF'leri.

## Doğrulamanın bulduğu hatalar (düzeltildi)

| Kaynak | Hata | Düzeltme |
|---|---|---|
| Zilly vd. 2021 | Yazar listesi yanlıştı (iki yazar hatalı yazılmıştı) | Zilly, Achille, Censi, Frazzoli; NeurIPS 34, s. 12386–12399 |
| Chen ve Zhang | arXiv ön baskısı (2025) olarak girilmişti | PRX Quantum 7(3), 033003 (2026), DOI 10.1103/ry52-85ll |
| Klein vd. (derleme) | Yazar listesi eski sürüme aitti | Güncel sürüm (v3): Klein, Luther, McAuliffe, Miklautz, Plant, Tschiatschek |
| Joudaki vd. 2026 | Yazar listesi eksikti | 8 yazar, ICLR 2026, s. 48999–49027 |
| Wang vd. 2025 | İkinci yazarın adı "Wei P. Dai" yazılmıştı | Bildiri kaydındaki biçim: "Wei Dai"; s. 69412–69444 |
| Roy ve Vetterli 2007 | Sayfa numarası doğrulanamadı | Sayfa numarası çıkarıldı (başlık, yazar, konferans doğrulandı) |
| Makale metni | "Ağırlık normunun büyümesi" için Lyle vd. 2023'e atıf yapılmıştı; o çalışmanın özeti eğrilik değişimini vurguluyor | Atıf Dohare vd. 2024, Elsayed vd. 2024, Lyle vd. 2024'e taşındı; Lyle vd. 2023 eğrilik için kullanıldı |
| Makale metni | Dohare vd. için "doğrusal ağdan iyi öğrenemez" yazılmıştı | Nature özetindeki ifade: "sığ (shallow) bir ağdan iyi öğrenemez" |
| Goodfellow vd. | Yalnızca arXiv ön baskısı (2013) olarak girilmişti; arXiv kaydında yayın bilgisi yok | ICLR 2014 konferans bildirisi (resmî ICLR arşivindeki kabul listesinden doğrulandı, 18 Kasım 2026); arXiv numarası notta bırakıldı |
| Kullanılmayan iki kayıt | Agarwal vd. 2021 ve Dohare vd. 2023 metinde atıflı değildi | Kaynakçadan çıkarıldı |

## Kaynak kaynak doğrulama

"İçerik" sütunu, makalede o kaynağa dayandırılan ifadenin hangi metinle doğrulandığını gösterir.

### Plastisite kaybı: olgu ve açıklamalar

| Kaynak | Künye kaynağı | İçerik kontrolü |
|---|---|---|
| Dohare vd. 2024, Nature 632(8026):768–774 | Crossref | Tam metin (Europe PMC): 800 görev, görev başına 60.000 görüntü tek tek; ölü birim artar, ağırlık büyüklüğü artar, etkin kerte düşer; "shallow network" |
| Dohare vd. 2021, arXiv:2108.06325 | arXiv | Özet: sürekli öğrenmede backprop'un zamanla bozulması |
| Ash ve Adams 2020, NeurIPS 33:3884–3894 | NeurIPS | Özet + tam metin: warm-start daha kötü genelleme; "shrink and perturb" |
| Berariu vd. 2021, arXiv:2106.00042 | arXiv | Özet: önceden eğitilmiş model taze modelin genellemesine ulaşamayabilir |
| Lyle vd. 2022, ICLR | arXiv ("Presented at ICLR 2022") | Tam metin: "feature rank" ölçümü, kapasite kaybı |
| Lyle vd. 2023, ICML, PMLR 202:23190–23211 | PMLR | Özet: kayıp, kayıp yüzeyinin eğriliğindeki değişimle bağlantılı |
| Lyle vd., CoLLAs, PMLR 274:750–783 (2025) | PMLR | Özet + tam metin: bağımsız mekanizmalar; layer norm + weight decay; "parameter norm growth" |
| Nikishin vd. 2022, ICML, PMLR 162:16828–16847 | PMLR | Özet: ajanın bir bölümünü periyodik sıfırlama |
| Sokar vd. 2023, ICML, PMLR 202:32145–32168 | PMLR | Özet: uyuyan nöronlar, ReDo |
| Abbas vd. 2023, CoLLAs, PMLR 232:620–636 | PMLR | Tam metin: aktivasyon ayak izi seyrekleşir; CReLU |
| Zilly vd. 2021, NeurIPS 34:12386–12399 | NeurIPS | Özet: ardışık katmanlarda "karşılıklı donmuş" ağırlıklar |
| Lewandowski vd. 2023, arXiv:2312.00246 | arXiv | Özet: eğrilik yönlerinin kaybı |
| Juliani ve Ash 2024, NeurIPS 37:113884–113910 | NeurIPS | Özet: on-policy derin RL'de plastisite kaybı |
| Klein vd. 2024, arXiv:2411.04832 (v3) | arXiv | Özet: derleme, 50'den fazla yöntem |
| Joudaki vd. 2026, ICLR, s. 48999–49027 | ICLR | Özet: gradyan yörüngelerini tutan manifoldlar; donmuş ve klon birimler |
| Wang vd. 2025, NeurIPS 38:69412–69444 | NeurIPS | Özet: iki tür plastisite kaybı |
| Liu ve Mou 2026, arXiv:2602.09234 | arXiv | Özet: görev geçişinin ani olmasının rolü |

### Çözüm yöntemleri

| Kaynak | Künye kaynağı | İçerik kontrolü |
|---|---|---|
| Elsayed vd. 2024, RLJ 5:2198–2217 | RLJ sayfası | Tam metin: [−κ·s_l, κ·s_l] aralığına kırpma, s_l başlatma sınırı |
| Elsayed ve Mahmood 2024, ICLR, s. 33399–33430 | ICLR | Özet: yararlılık ölçekli pertürbasyon (UPGD) |
| Kumar vd., CoLLAs, PMLR 274:410–430 (2025) | PMLR | Özet: başlangıç parametrelerine doğru L2 (L2 Init) |
| Lyle vd. 2024, NeurIPS 37:106440–106473 | NeurIPS | Özet: norm büyümesi ile etkin öğrenme hızı düşüşü; Normalize-and-Project |
| Chung vd. 2024, NeurIPS 37:127937–127967 | NeurIPS | Özet: Parseval düzenlileştirmesi, ortogonallik |
| Lewandowski vd. 2025, ICLR, s. 34504–34530 | ICLR | Özet: spektral düzenlileştirme |
| Lewandowski vd. 2025, ICLR, s. 31112–31137 | ICLR | Özet: her katmanda sinüs + kosinüs (deep Fourier features) |
| Galashov vd. 2024, NeurIPS 37:83197–83234 | NeurIPS | Özet: başlangıç dağılımına doğru yumuşak sıfırlama |
| Farias ve Jozefiak 2025, ICLR, s. 20852–20886 | ICLR | Özet: nöron sıfırlama (SNR) |
| Lee vd. 2025, ICML, PMLR 267:33352–33403 | PMLR | Özet: hiperküresel normalizasyon ile norm büyümesinin sınırlanması |
| Oh vd. 2025, arXiv:2507.04683 | arXiv | Özet: Soft Weight Rescaling |
| Lillo ve Cheney 2026, ICLR, s. 96814–96850 | ICLR | Tam metin: "non-zero derivative floor", Smooth-Leaky |
| Chen ve Zhang 2026, PRX Quantum 7:033003 | Crossref | Tam metin (arXiv v2): 3.000 ikili CIFAR-100 görevi; sinüs aktivasyonlu MLP plastisite kaybeder; kompakt manifold |

### Periyodik fonksiyonlar, sınırlı ağırlıklar, yeniden parametrizasyon

| Kaynak | Künye kaynağı | İçerik kontrolü |
|---|---|---|
| Dai vd. 2025, npj Comput. Mater. 11(1):259 | Crossref | Tam metin: üçgen dalga koordinat haritası; "constant gradient magnitudes … in contrast with sinusoidal mappings, whose varying gradients…" |
| Sitzmann vd. 2020, NeurIPS 33:7462–7473 | NeurIPS | Özet: periyodik aktivasyonlar |
| Ziyin vd. 2020, NeurIPS 33:1583–1594 | NeurIPS | Özet: periyodik fonksiyonların öğrenilmesi |
| Naumov vd. 2018, arXiv:1811.09862 | arXiv | Özet: ağırlıkların periyodik fonksiyonları düzenlileştirici olarak |
| Salimans ve Kingma 2016, NeurIPS 29 | NeurIPS | Özet: ağırlık yeniden parametrizasyonu |
| Courbariaux vd. 2015, NeurIPS 28 | NeurIPS | Tam metin: ağırlıkların kırpılması (clip) |
| Bengio vd. 2013, arXiv:1308.3432 | arXiv | Özet: "straight-through estimator" |

### Veri kümeleri, mimari, eniyileme

| Kaynak | Künye kaynağı | İçerik kontrolü |
|---|---|---|
| Goodfellow vd. 2014, ICLR (arXiv:1312.6211) | ICLR 2014 resmî arşivi (iclr.cc/archive/2014/conference-proceedings: kabul edilen konferans bildirileri listesi, arXiv:1312.6211 bağlantısıyla) + arXiv | Tam metin: piksel permütasyonlu MNIST görevleri |
| Zenke vd. 2017, ICML, PMLR 70:3987–3995 | PMLR | Tam metin: "permuted MNIST" |
| LeCun vd. 1998, Proc. IEEE 86(11):2278–2324 | Crossref | Künye |
| Krizhevsky 2009, teknik rapor | Rapor PDF'i (Toronto Üniversitesi) | Künye |
| He vd. 2015, ICCV, s. 1026–1034 | Crossref | Özet: doğrultucular için başlatma |
| Ba vd. 2016, arXiv:1607.06450 | arXiv | Özet |
| Kingma ve Ba 2015, ICLR | arXiv ("3rd ICLR, San Diego, 2015") | Özet |
| Roy ve Vetterli 2007, EUSIPCO | Bildiri PDF'i (EURASIP) | Başlık, yazar, konferans; sayfa numarası verilmedi |

### İstatistik

| Kaynak | Künye kaynağı | Not |
|---|---|---|
| Schuirmann 1987, J. Pharmacokinet. Biopharm. 15(6):657–680 | Crossref | TOST |
| Lakens 2017, Soc. Psychol. Personal. Sci. 8(4):355–362 | Crossref | Eşdeğerlik testleri |
| Efron ve Tibshirani 1993 | Crossref (DOI 10.1007/978-1-4899-4541-9) | Bootstrap |
| Holm 1979, Scand. J. Statist. 6(2):65–70 | **İkincil kaynak** | JSTOR otomatik erişimi engelliyor, Crossref'te kayıt yok; künye arama sonuçlarındaki JSTOR kaydından (stable/4615733) alındı. Elle bir kez kontrol edilmeli |

## Sınırlar

* İçerik kontrolü, makalede kullanılan ifadeyle sınırlıdır; makalelerin tamamı baştan sona okunmadı.
* Holm (1979) birincil kaynaktan doğrulanamayan tek künyedir.
* "Sınırlı-periyodik ağırlık yeniden parametrizasyonunu plastisite kaybı için inceleyen bir çalışma bulunamadı"
  ifadesi kapsamlı bir sistematik taramaya değil, yapılan aramalara dayanır; makalede de bu şekilde yazıldı.

## Bildiri (IISEC) için yeniden doğrulama: 18 Kasım 2026

Bildirideki 21 kaynak birincil kayıtlardan yeniden çekildi (arXiv API, PMLR cilt sayfaları v70/v202/v232/v274,
NeurIPS ve ICLR bildiri sitelerinin BibTeX kayıtları, Crossref, Reinforcement Learning Journal 2024 sayısı, EURASIP
arşivindeki EUSIPCO 2007 PDF'i, ICLR 2014 arşivi). Yazar, başlık, yer, cilt, sayfa ve yıl bilgilerinin hepsi tuttu.

İçerik kontrolü (bildiride o kaynağa dayandırılan cümle, özet ya da tam metinle karşılaştırıldı):

* Tam metinden: Dohare vd. 2024 (Europe PMC), Lyle vd. 2022, Elsayed vd. 2024, Lillo ve Cheney 2026, Chen ve Zhang 2026
  (arXiv sürümü), Dai vd. 2025 (arXiv sürümü), He vd. 2015, Goodfellow vd. 2014, Zenke vd. 2017, Roy ve Vetterli 2007.
* Özetten: Ash ve Adams 2020, Lyle vd. 2023, Sokar vd. 2023, Abbas vd. 2023, Lyle vd. 2024 (NeurIPS), Lyle vd. (CoLLAs),
  Kumar vd., Lewandowski vd. 2025, Salimans ve Kingma 2016.
* Yalnızca künye (tam metne erişilemedi): LeCun vd. 1998 (MNIST), Schuirmann 1987 (TOST; başlık yöntemi adıyla anıyor).

Bu kontrolün bildiride düzelttirdiği ifadeler: (1) "nedenler birlikte etki eder" -> Lyle vd.'nin özetindeki gibi
"nedenler ayrıdır, yalnızca birini gidermek her zaman yetmez"; (2) "Kaiming sınırı b=sqrt(6/n)" -> He vd. Gauss
biçimini (varyans 2/n) verir; "aynı varyansı veren düzgün dağılım sınırı" olarak yazıldı; (3) ağırlık normalleştirme
için "optimizasyon geometrisi" -> özetteki "koşullanma" ifadesi.

**Yayımlanmış asıl metinlerden ikinci kontrol (18 Kasım 2026).** Aşağıdaki kaynakların yayıncı/bildiri sitesindeki PDF'i
indirildi ve bildiride dayandırılan cümle o metinde bulundu: Ash ve Adams (NeurIPS), Lyle vd. 2023 ve Sokar vd. (PMLR 202),
Abbas vd. (PMLR 232), Elsayed vd. (RLJ), Lyle vd. 2024 (NeurIPS), Lyle vd. ve Kumar vd. (PMLR 274), Lewandowski vd.
(ICLR 2025), Lillo ve Cheney (ICLR 2026), Salimans ve Kingma (NeurIPS), Dai vd. (npj Comput. Mater.), He vd. (CVF açık
erişim), Zenke vd. (PMLR 70), Roy ve Vetterli (EURASIP). Dohare vd. (Nature) Europe PMC'deki yayımlanmış tam metinden,
Goodfellow vd. ICLR 2014 arşivinin bağlandığı arXiv metninden okundu. Otomatik erişime kapalı olduğu için asıl metni
okunamayanlar: Lyle vd. 2022 (OpenReview), Chen ve Zhang (APS), LeCun vd. 1998 (IEEE Xplore), Schuirmann 1987 (Springer);
ilk ikisinin arXiv sürümü okundu. Salimans ve Kingma cümlesindeki "ağın temsil edebildiği işlevler değişmez" eki kaynakta
açıkça geçmediği için çıkarıldı.

**Kullanıcının indirdiği yayıncı PDF'lerinden kontrol (18 Kasım 2026).** LeCun vd. 1998 (Proc. IEEE 86(11), s. 2278-2324:
"The resulting database was called the modified NIST, or MNIST, dataset"), Schuirmann 1987 (J. Pharmacokinet. Biopharm.
15(6), s. 657-680: "The Two One-Sided Tests Procedure" bölümü) ve Chen ve Zhang 2026 (PRX Quantum 7, 033003: sinüs
aktivasyonlu klasik MLP, ağırlıklar sınırsız, plastisite kaybı; ana metin ve Ek D) yayımlanmış metinden doğrulandı.
Asıl metni henüz okunmayan tek kaynak: Lyle, Rowland ve Dabney 2022 (ICLR; arXiv sürümü okundu).
