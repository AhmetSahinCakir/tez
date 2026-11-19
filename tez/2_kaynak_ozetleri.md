# 2. KAYNAK ÖZETLERİ

> Taslak (19 Kasım 2026). Enstitü kılavuzuna göre bu bölümde kaynaklara tarih sırasıyla değinilir. Her cümle,
> `docs/KAYNAKCA.md` dosyasında doğrulanmış içerikle sınırlı tutulmuştur; bir kaynağın yalnızca özeti
> okunduysa o kaynak hakkında özetin söylediğinden fazlası yazılmamıştır. Cümleler tez yazarı tarafından
> yeniden yazılacaktır.

LeCun vd. (1998), el yazısı rakam görüntülerinden oluşan MNIST veri kümesini tanıtmıştır. Bu tezdeki ana
deneyler bu veri kümesinden üretilen görev akışlarını kullanır.

Bengio vd. (2013), türevi sıfır olan ya da tanımsız olan işlemlerden gradyan geçirmek için düz-geçişli
kestirici (straight-through estimator) yaklaşımını ele almıştır. Tezdeki yansımalı sinüs deneyinde geri
geçişte gerçek türev yerine kullanılan kural bu türden bir vekil gradyandır.

Goodfellow vd. (2014), sinir ağlarında yıkıcı unutmayı incelerken piksellerin rastgele yer değiştirmesiyle
üretilen MNIST görevlerini kullanmıştır. Zenke vd. (2017) aynı tür görev dizisini "permuted MNIST" adıyla
kullanmıştır. Bu tezdeki ana veri akışı bu görev ailesine dayanır.

Courbariaux vd. (2015), ikili ağırlıklı ağların eğitiminde gerçek değerli ağırlıkları her güncellemeden sonra
sabit bir aralığa kırpmıştır. He vd. (2015), doğrultucu (ReLU) aktivasyonlu ağlar için bir başlatma yöntemi
önermiştir; tezdeki ağların başlangıç dağılımı bu yönteme dayanır.

Salimans ve Kingma (2016), ağırlıkların başka parametreler cinsinden yazılmasının (ağırlık normalizasyonu)
eğitimi hızlandırdığını göstermiştir. Bu çalışma, ağırlık yeniden parametrizasyonunun eniyileme üzerindeki
etkisine ilişkin temel kaynaklardan biridir.

Ash ve Adams (2020), önceden eğitilmiş bir ağdan devam etmenin (warm start), sıfırdan eğitilen ağa göre daha
kötü genellemeye yol açabildiğini göstermiş ve "shrink and perturb" yöntemini önermiştir. Sitzmann vd. (2020)
ve Ziyin vd. (2020) periyodik fonksiyonları aktivasyon olarak incelemiştir; bu çalışmalar sürekli öğrenmeyle
ilgili değildir.

Dohare vd. (2021), sürekli öğrenmede geri yayılımın zamanla bozulduğunu bildirmiştir. Berariu vd. (2021),
önceden eğitilmiş bir modelin taze bir modelin genelleme düzeyine ulaşamayabildiğini; Zilly vd. (2021) ise
ardışık katmanlardaki ağırlıkların karşılıklı olarak donabildiğini göstermiştir.

Lyle vd. (2022), pekiştirmeli öğrenmede ağın yeni hedeflere uyum kapasitesinin azaldığını ve bunun temsil
kertesiyle ilişkisini incelemiştir. Nikishin vd. (2022), ajanın bir bölümünün belirli aralıklarla yeniden
başlatılmasını önermiştir.

Lyle vd. (2023), plastisite kaybını kayıp yüzeyinin eğriliğindeki değişimle ilişkilendirmiştir. Sokar vd.
(2023), derin pekiştirmeli öğrenmede etkinliği çok düşük "uyuyan" nöronların zamanla arttığını göstermiş ve bu
nöronları yeniden başlatan ReDo yöntemini önermiştir. Abbas vd. (2023), sürekli pekiştirmeli öğrenmede
aktivasyonların seyrekleştiğini göstermiş ve CReLU aktivasyonunu önermiştir.

Dohare vd. (2024), 800 görevlik Online Permuted MNIST akışında (görev başına 60.000 görüntü, tek tek ve bir
kez) standart ağların plastisite kaybettiğini göstermiştir. Bu çalışmada kayıp sırasında ölü birimlerin
arttığı, ağırlık büyüklüğünün arttığı ve temsilin etkin kertesinin düştüğü bildirilmiş; düşük yararlılıklı
birimleri seçerek yeniden başlatan "continual backpropagation" yöntemi önerilmiştir. Tezdeki ana deney
düzeni bu çalışmanın protokolünü izler.

Elsayed vd. (2024), her güncellemeden sonra ağırlıkları başlatma sınırının κ katı olan bir aralığa kırpan
Weight Clipping yöntemini önermiştir. Bu yöntem, tezdeki sınırlı parametrizasyonun en yakın karşılığıdır ve
ana karşılaştırma yöntemidir. Elsayed ve Mahmood (2024), yararlılıkla ölçeklenen pertürbasyonlu gradyan
yöntemini (UPGD) önermiştir.

Lyle vd. (2024), ağırlık normunun büyümesinin etkin öğrenme hızını düşürdüğünü göstermiş ve katman
normalizasyonunu norm izdüşümüyle birleştiren "Normalize-and-Project" yöntemini önermiştir. Chung vd. (2024)
ağırlık matrislerini ortogonalliğe yaklaştıran Parseval düzenlileştirmesini önermiştir. Klein vd. (2024),
plastisite kaybı üzerine 50'den fazla yöntemi kapsayan bir derleme hazırlamıştır.

Kumar vd. (2025), parametreleri başlangıç değerlerine doğru çeken bir L2 düzenlileştirmesi (L2 Init)
önermiştir. Bu çalışma ayrıca, sabit bir görüntü kümesine her görevde rastgele etiket atanan Random Label
MNIST problemini kullanmıştır; tezdeki ikinci veri akışı [pilot aşamasında] bu problemin protokolünü izler.

Lyle vd. (2025), plastisite kaybının birbirinden bağımsız birden fazla mekanizması olduğunu ve katman
normalizasyonu ile ağırlık sönümünün birlikte kullanılmasının çeşitli görevlerde plastisiteyi korumada
etkili olduğunu bildirmiştir. Lewandowski vd. (2025), her katmanda sinüs ve kosinüs aktivasyonlarını
birlikte kullanan ağların eğitilebilir kaldığını göstermiştir. Lee vd. (2025) hiperküresel normalizasyonla
norm büyümesini sınırlamıştır.

Dai vd. (2025), nanofotonik aygıt tasarımında kullanılan bir sinir ağı parametrizasyonunda koordinat haritası
olarak üçgen dalgayı kullanmış ve bu haritanın gradyan büyüklüğünün sabit olduğunu, sinüs haritasında ise
gradyanın değiştiğini belirtmiştir. Üçgen dalganın sabit gradyan büyüklüklü bir harita olarak kullanımı
dolayısıyla yeni değildir; bu tezde yeni olan, haritanın ağırlıklara uygulanması ve plastisite kaybı
bağlamında incelenmesidir.

Lillo ve Cheney (2026), aktivasyon fonksiyonunun türevinin sıfırdan uzak kalmasının plastisitenin korunmasıyla
ilişkili olduğunu göstermiştir. Chen ve Zhang (2026), 3.000 ikili CIFAR-100 görevinden oluşan bir dizide
sinüs aktivasyonlu ve ağırlıkları sınırlanmamış klasik bir ağın plastisite kaybettiğini bildirmiştir. Bu iki
çalışma periyodik fonksiyonu ya da türev koşulunu aktivasyona uygular; bu tezde ise harita ağırlıklara
uygulanır ve ağın hesapladığı fonksiyon sınıfı ReLU ağı olarak kalır.

Yapılan aramalarda, sınırlı ve periyodik bir yeniden parametrizasyonu ağ ağırlıklarına plastisite kaybı
bağlamında uygulayan bir çalışmaya rastlanmamıştır. Bu ifade sistematik bir taramaya değil, yapılan aramalara
dayanır.

İstatistiksel değerlendirmede, iki yöntemin eşdeğerliğini sınamak için iki tek yönlü test yöntemi
(Schuirmann, 1987), çoklu karşılaştırmalar için Holm (1979) düzeltmesi ve güven aralıkları için bootstrap
yöntemi (Efron ve Tibshirani, 1993) kullanılmıştır.
