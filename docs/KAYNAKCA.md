# Kaynakça durumu (makale ve tez)

Tam kayıtlar: `paper/references.bib` (BibTeX, 49 kaynak). Son tarama: 16 Kasım 2026.

Durum sütunu: **✔** = künye bu taramada yayıncı / bildiri sayfasından doğrulandı; **○** = künye hafızadan yazıldı,
gönderimden önce sayfa ve cilt numaraları kontrol edilmeli (çalışmanın varlığı ve içeriği konusunda şüphe yok).
Not: bu bulut oturumundan arxiv.org açılamıyor (ağ izni yok); doğrulamalar arama sonuçlarından yapıldı.

## 1. Depoda zaten kullanılan kaynaklar (15)

| Kaynak | Makaledeki rolü | Durum |
|---|---|---|
| Dohare vd. 2024, Nature 632 | Olgunun tanımı, 800 görevlik protokol, Continual Backprop | ○ |
| Lyle vd. 2023, ICML | Plastisite kaybının nedenleri, ağırlık normu | ○ |
| Lyle vd., "Disentangling…", CoLLAs (PMLR 274, 2025) | Layer Norm + Weight Decay referansı; "birden fazla mekanizma" | ✔ (yıl: bildiri cildi 2025, konferans 2024) |
| Lyle vd. 2024, NeurIPS (NaP) | Normalize-and-Project | ○ |
| Elsayed vd. 2024, RLJ 5:2198–2217 | Weight Clipping (ana rakip) | ✔ |
| Elsayed ve Mahmood 2024, ICLR | UPGD | ○ |
| Kumar vd., CoLLAs (PMLR 274:410–430) | L2 Init | ✔ |
| Ash ve Adams 2020, NeurIPS | Warm-start, Shrink & Perturb | ○ |
| Chung vd. 2024, NeurIPS | Parseval düzenlileştirmesi | ○ |
| Lee vd. 2025, ICML (PMLR 267) | Hyperspherical normalization | ✔ |
| Lillo ve Cheney, ICLR 2026 | Aktivasyon tasarımı, Smooth-Leaky, "türev tabanı" | ✔ |
| Chen ve Zhang, arXiv:2511.17228 (2025) | Kompakt parametre uzayı, Sin-MLP, CIFAR-100 protokolü | ✔ arXiv; **düzeltme:** depoda "2026" yazıyordu. PRX Quantum'da yayımlandığına dair doğrulanmamış bir bilgi var, kontrol edilmeli |
| Dai vd. 2025, npj Comput. Mater. | Üçgen dalga haritasının daha önceki kullanımı (özgünlük sınırı) | ✔ künye; **üçgen dalga pasajı ve makale no. (259) bu taramada doğrulanamadı**, gönderimden önce metinden alıntılanmalı |
| He vd. 2015, ICCV | Kaiming başlatma | ○ |
| Schuirmann 1987 | TOST eşdeğerlik testi | ○ |

## 2. Bu taramada eklenenler (34)

**Mutlaka gerekli olanlar (hakem arar):**

| Kaynak | Neden gerekli | Durum |
|---|---|---|
| Lewandowski, Schuurmans, Machado, "Plastic learning with deep Fourier features", ICLR 2025 | Sürekli öğrenmede periyodik (sin/cos) *aktivasyon*; bizim çalışmamıza en yakın "periyodiklik" çalışması, farkı açıkça yazılmalı | ✔ |
| Joudaki vd., "Barriers for learning in an evolving world", ICLR 2026 | Plastisite kaybının matematiksel açıklaması (gradyan akışının değişmez manifoldları) | ✔ (yazar listesi eksik: 8 yazar, 4'ü doğrulandı) |
| Wang vd., "The dual nature of plasticity loss…", NeurIPS 2025 | İki tür plastisite kaybı | ✔ |
| Liu ve Mou, "Do neural networks lose plasticity in a gradually changing world?", arXiv 2026 | Karşı görüş: kayıp, ani görev değişiminin ürünü | ✔ |
| Oh vd., "Soft weight rescaling", arXiv 2025 | Ağırlık büyüklüğünü sınırlayan yeni bir yöntem | ✔ |
| Klein vd., "Plasticity loss in deep RL: a survey", arXiv 2024 | Derleme | ✔ |
| Zilly vd. 2021, NeurIPS | "Karşılıklı donmuş ağırlıklar": donma mekanizmasının öncülü | ○ |
| Sokar vd. 2023; Abbas vd. 2023; Nikishin vd. 2022; Lyle vd. 2022; Berariu vd. 2021; Dohare vd. 2021; Juliani ve Ash 2024; Lewandowski vd. 2023 | Olgunun ve mekanizmaların temel literatürü | ○ |
| Lewandowski vd. 2025 (spektral düzenlileştirme); Galashov vd. 2024; Farias ve Jozefiak 2025; Dohare vd. 2023 | Diğer çözüm aileleri | ○ |

**Yöntem ve arka plan:**

| Kaynak | Rolü | Durum |
|---|---|---|
| Sitzmann vd. 2020 (SIREN); Ziyin vd. 2020 | Periyodik aktivasyonlar | ○ |
| Naumov vd. 2018 | Ağırlıkların periyodik fonksiyonu (nicemleme düzenlileştirmesi) | ○ |
| Salimans ve Kingma 2016 | Ağırlık yeniden parametrizasyonu | ○ |
| Courbariaux vd. 2015 | Ağırlık kırpmanın önceki kullanımı | ○ |
| Bengio vd. 2013 | Düz-geçişli kestirici (yansımalı sinüsün geri geçişi) | ○ |
| Goodfellow vd. 2013; Zenke vd. 2017; LeCun vd. 1998; Krizhevsky 2009 | Permuted MNIST, MNIST, CIFAR-100 | ○ |
| Ba vd. 2016; Kingma ve Ba 2015; Roy ve Vetterli 2007 | Layer Norm, Adam, etkin kerte | ○ |
| Lakens 2017; Holm 1979; Efron ve Tibshirani 1993; Agarwal vd. 2021 | İstatistik: eşdeğerlik testi, çoklu test düzeltmesi, bootstrap, tohum sayısı | ○ |

## 3. Taramanın sonucu

* Yapısal olarak sınırlı-periyodik bir *ağırlık* yeniden parametrizasyonunu plastisite kaybı için inceleyen bir
  çalışma bulunamadı; en yakın çalışmalar periyodik *aktivasyon* (Lewandowski vd. 2025), aktivasyon şekli
  (Lillo ve Cheney 2026) ve işlemsel sınır (Elsayed vd. 2024) üzerinedir. Özgünlük iddiası bu sınırla yazıldı.
* Gönderimden önce yapılacaklar: (i) ○ işaretli künyelerin sayfa/cilt kontrolü; (ii) Dai vd. 2025'teki üçgen
  dalga pasajının metinden doğrulanması; (iii) Chen ve Zhang'ın dergi künyesi; (iv) Joudaki vd. yazar listesi.
