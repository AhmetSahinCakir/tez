# 1. GİRİŞ

> Taslak (19 Kasım 2026). Bu metin bir iskelettir: içerik ve sıra doğrudur, cümleler tez yazarı tarafından
> kendi ifadesiyle yeniden yazılacaktır. Kaynak gösterimi enstitü kılavuzundaki (Soyad, Yıl) biçimindedir.

Yapay sinir ağları çoğunlukla sabit bir veri kümesi üzerinde bir kez eğitilir ve sonra kullanılır. Birçok
uygulamada ise veri zamanla değişir ve ağın yeni veriden öğrenmeyi sürdürmesi beklenir. Bu öğrenme biçimine
sürekli öğrenme denir. Sürekli öğrenmede iki ayrı sorun vardır. Birincisi, ağın eski görevlerde öğrendiklerini
unutmasıdır (yıkıcı unutma). İkincisi, ağın yeni görevleri öğrenme yeteneğinin zamanla azalmasıdır. Bu ikinci
soruna plastisite kaybı denir ve bu tezin konusudur.

Plastisite kaybında ağ, yeterince uzun bir görev dizisinden sonra yeni bir görevi, aynı görevi sıfırdan
öğrenen bir ağdan daha yavaş ve daha düşük doğrulukla öğrenir. Dohare vd. (2024), geri yayılımla eğitilen
standart ağların uzun görev dizilerinde bu yeteneği kaybettiğini ve sonunda sığ bir ağdan daha iyi
öğrenemediğini göstermiştir. Benzer bir etki daha önce, önceden eğitilmiş bir ağın yeni veride sıfırdan
başlatılan ağa göre daha kötü genellemesi olarak da gözlenmiştir (Ash ve Adams, 2020).

Plastisite kaybı için birkaç neden önerilmiştir: ölü ya da uyuyan birimler (Sokar vd., 2023; Abbas vd., 2023),
temsilin kertesinin düşmesi (Lyle vd., 2022; Dohare vd., 2024) ve ağırlık büyüklüğünün artması (Dohare vd.,
2024; Elsayed vd., 2024; Lyle vd., 2024). Lyle vd. (2025) bu nedenlerin birbirinden bağımsız olduğunu ve
yalnızca birini gidermenin her zaman yeterli olmadığını bildirmiştir.

Ağırlık büyüklüğünü sınırlayan yöntemler arasında en basiti ağırlık kırpmadır (Weight Clipping): her
güncellemeden sonra ağırlıklar, başlangıç ölçeğiyle belirlenen bir aralığa geri izdüşürülür (Elsayed vd.,
2024). Kırpma, eniyileyicinin adımından sonra uygulanan ayrı bir işlemdir. Aynı kısıt modelin içine de
yerleştirilebilir: her ağırlık, serbest bir parametrenin sınırlı bir fonksiyonu olarak yazılır,

    W = A · f(Θ / A),        f: ℝ → [−1, 1].

Bu durumda Θ ne olursa olsun |W| ≤ A'dır ve ayrıca bir izdüşüm adımına gerek kalmaz. Bu tezde bu yaklaşıma
sınırlı ağırlık yeniden parametrizasyonu denmiştir.

## 1.1. Amaç ve araştırma soruları

Tezin başlangıç fikri, f olarak sinüs fonksiyonunun seçilmesiydi: sinüs hem sınırlıdır hem periyodiktir ve
ağırlıkları yapısal olarak sınırlı tutar. Tez önerisinde üç hipotez yer almıştır:

* **H1.** Sinüs yeniden parametrizasyonu, standart ağa göre plastisiteyi korur.
* **H2.** Bu etki sınırlılıktan mı, periyodiklikten mi gelir? Karşılaştırma için sınırlı ama periyodik olmayan
  tanh haritası kullanılır.
* **H3.** Yeniden parametrizasyon eniyileme geometrisini değiştirir: etkin ağırlık adımı haritanın türevinin
  karesiyle çarpılır; bu çarpan sınıra yaklaşan ağırlıklarda küçülür.

Deneyler H1'i yalnızca kısmen desteklemiştir ve çalışmanın ağırlık merkezi H3'e kaymıştır. Tezin yanıt
aradığı asıl soru şudur: **sınırlı bir ağırlık parametrizasyonunda plastisiteyi belirleyen özellik ağırlığın
sınırlı olması mıdır, yoksa güncelleme adımının büyüklüğünün korunması mıdır?**

## 1.2. Kapsam

* Veri akışları: Online Permuted MNIST (ana deney), ardışık CIFAR-100 ikili sınıflandırma ve
  [pilotu süren] Random Label MNIST.
* Ağ: tam bağlantılı, ReLU aktivasyonlu, küçük bir ağ (üç gizli katman, 100 birim); geniş ağ (2.000 birim) ayrı
  bir sınır koşulu olarak incelenmiştir.
* Eniyileyici: ana deneylerde stokastik gradyan inişi; Adam ayrı bir rejim olarak incelenmiştir.
* Tez yeni bir yöntem önermekten çok, sınırlı parametrizasyonun hangi özelliğinin işe yaradığını ölçen
  deneysel bir çalışmadır. Üçgen dalga haritası ağırlık kırpmadan daha iyi değildir; onunla aynı sonucu verir.

## 1.3. Tezin katkıları

1. Pürüzsüz sınırlı haritaların (sinüs, tanh) ağırlıkları sınırlı tuttuğu hâlde plastisite kaybını
   önlemediğinin, eşleştirilmiş tohumlarla gösterilmesi.
2. Bunun nedenine ilişkin ölçüm: sınıra yaklaşan ağırlıklarda adım çarpanının küçülmesi ve bu etkinin
   yalnızca geri türevi değiştiren tek etkenli bir deneyle sınanması.
3. Türevi sıfıra yaklaşmayan sınırlı ve periyodik bir haritanın (üçgen dalga) kayıp göstermediğinin ve ağırlık
   kırpmayla önceden belirlenmiş bir marj içinde eşdeğer sonuç verdiğinin gösterilmesi.
4. Sonuçların geçerlilik sınırlarının (eniyileyici, ağ genişliği, veri akışı) ayrı deneylerle belirlenmesi.

## 1.4. Tezin düzeni

İkinci bölümde konuyla ilgili çalışmalar tarih sırasıyla özetlenmiştir. Üçüncü bölümde yöntem, veri akışları,
karşılaştırılan modeller, ölçütler ve istatistiksel değerlendirme anlatılmıştır. Dördüncü bölümde bulgular
verilmiş ve tartışılmıştır. Beşinci bölümde sonuçlar ve öneriler yer almaktadır.
