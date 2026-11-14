// Builds docs/danisman_bilgilendirme_raporu.docx (advisor briefing, 12 pt body).  Usage: node scripts/danisman_raporu_docx.js docs/danisman_bilgilendirme_raporu.docx
const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ImageRun, AlignmentType, LevelFormat, BorderStyle, ShadingType, VerticalAlign } = require('docx');
const FONT = 'Times New Roman', BODY = 24, TBL = 20, CAP = 20, H1 = 32, H2 = 28, W = 9638;
const runs = (s, size, extra = {}) => s.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map(p =>
  p.startsWith('**') ? new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, size, ...extra }) : new TextRun({ text: p, font: FONT, size, ...extra }));
const P = (s, o = {}) => new Paragraph({ children: runs(s, BODY), alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 120, line: 276 }, ...o });
const H = s => new Paragraph({ children: [new TextRun({ text: s, bold: true, font: FONT, size: H2 })], spacing: { before: 200, after: 100 }, keepNext: true });
const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' }, borders = { top: border, bottom: border, left: border, right: border };
const table = (widths, rows) => new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths,
  rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, cantSplit: true, children: r.map((c, j) => new TableCell({
    width: { size: widths[j], type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: i === 0 ? { type: ShadingType.CLEAR, fill: 'E7E6E6', color: 'auto' } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ children: runs(i === 0 ? `**${c}**` : c, TBL), alignment: j === 0 || widths.length === 3 ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { before: 0, after: 0 } })] })) })) });
const gap = () => new Paragraph({ children: [], spacing: { before: 0, after: 80 } });
const LI = (s, ref) => new Paragraph({ children: runs(s, BODY), numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 80, line: 276 } });

const children = [
  new Paragraph({ children: [new TextRun({ text: 'Tez Bilgilendirme Raporu: 31 Ekim değerlendirmesinden sonra yapılanlar ve sonuçlar', bold: true, font: FONT, size: H1 })], alignment: AlignmentType.CENTER, spacing: { before: 0, after: 120 } }),
  new Paragraph({ children: runs('Tez: Sürekli Öğrenmede Plastisite Kaybının Azaltılması için Sınırlı ve Periyodik Ağırlık Yeniden Parametrizasyonu', BODY, { italics: true }), alignment: AlignmentType.CENTER, spacing: { before: 0, after: 160 } }),
  H('Özet'),
  P('31 Ekim 2026 tarihli değerlendirme raporunuzda istenen zorunlu işlerin tamamı bitti. Kanonik uzunluktaki akışta (800 görev × 60.000 örnek, 10 eşleştirilmiş tohum) üçgen dalga haritası plastisite kaybı göstermiyor ve Weight Clipping ile istatistiksel olarak eşdeğer (TOST, Δ = 0.005). Pürüzsüz sinüs haritası bu ölçekte standart ağın gerisine düşüyor; aynı haritada Jacobian sönümü kaldırılınca kayıp ortadan kalkıyor. Bundan sonraki adım için görüşünüzü rica ediyorum: önce tez mi, önce makale mi?'),
  H('Yapılanlar'),
  table([3300, 4838, 1500], [
    ['Değerlendirme raporundaki madde', 'Yapılan iş', 'Durum'],
    ['Yazım ve tanımlar (iddia ifadeleri, plastisite ölçütü, özgünlük sınırı, teorik çerçeve)', 'Rapor yeniden yazıldı; ölçüt: koruma oranı ≥ 0.99, eğim ≥ −0.002 / 100 görev, taze model farkı ≤ 0.01', 'Tamamlandı'],
    ['Deney 1, 2, 7: 800 görevlik akış', 'Beş model × 10 tohum, Layer Norm + Weight Decay × 10 tohum, yansımalı sinüs × 3 tohum (63 koşu)', 'Tamamlandı'],
    ['Deney 3: üçgen dalga – Weight Clipping eşdeğerliği', 'Önceden belirlenmiş marjla TOST (Δ = 0.005 ve 0.01)', 'Tamamlandı'],
    ['Deney 4, 5, 6: γ duyarlılığı, donma oranı şekli, CIFAR-100', 'γ ∈ {1.0 … 3.0} taraması; katman başına donma oranı; üçgen dalga CIFAR-100 akışında (5 tohum)', 'Tamamlandı'],
    ['Deney 8, 9: kapasite × ufuk matrisi, 3×2000 ağ × 800 görev', 'GPU gerektiriyor; isteğe bağlı olarak işaretlenmişti', 'Yapılmadı'],
  ]), gap(),
  P('Ana paket 63 koşudan oluşuyor ve yaklaşık 1.130 işlemci-saati sürdü. Bütün koşular hatasız tamamlandı; kod, ham sonuçlar ve analiz tabloları depoda duruyor.'),
  H('Ana sonuç: 800 görevlik akış'),
  P('Protokol Dohare vd. (2024) ile aynı uzunlukta: Online Permuted MNIST, 800 görev × 60.000 örnek, yığın boyutu 1, 3×100 ağ, SGD. Hiperparametreler 200 görevlik geliştirme akışında seçildi ve bu ölçek için yeniden ayarlanmadı.'),
  table([2638, 800, 1100, 1300, 1100, 1200, 1500], [
    ['Model', 'Tohum', 'Normalize AUC', 'Son pencere doğruluğu', 'Koruma oranı', 'Taze model farkı', 'Plastisite ölçütünü sağlayan tohum'],
    ['**Üçgen dalga (önerilen)**', '10', '**0.939**', '**0.940**', '**1.005**', '−0.023', '10 / 10'],
    ['Weight Clipping', '10', '0.939', '0.940', '1.006', '−0.022', '10 / 10'],
    ['Layer Norm + Weight Decay', '10', '0.925', '0.923', '0.992', '+0.008', '8 / 10'],
    ['Yansımalı sinüs', '3', '0.924', '0.920', '0.994', '+0.005', '3 / 3'],
    ['tanh', '10', '0.859', '0.831', '0.912', '+0.085', '0 / 10'],
    ['Standart ağ', '10', '0.856', '0.835', '0.924', '+0.084', '0 / 10'],
    ['Sinüs', '10', '0.818', '0.720', '0.792', '+0.212', '0 / 10'],
  ]), gap(),
  P('Koruma oranı, son 40 görevin doğruluğunun ilk 40 göreve oranıdır. Taze model farkı, 800. görevde sıfırdan eğitilen ağ ile sürekli öğrenen ağ arasındaki farktır; negatif değer sürekli öğrenen ağın daha iyi olduğunu gösterir. AUC\'de %95 güven aralığı bütün modellerde ±0.002 veya daha dardır.'),
  P('Üçgen dalga standart ağı her tohumda geçiyor: AUC farkı +0.083 [+0.082, +0.083]. Sinüs standart ağın her tohumda gerisinde: −0.039 [−0.040, −0.037]. Yani H1, pürüzsüz sinüs haritası için bu ölçekte reddediliyor; 200 görevdeki küçük olumlu fark (+0.011) uzun ufukta tersine dönüyor.'),
  new Paragraph({ children: [new ImageRun({ type: 'png', data: fs.readFileSync('/home/user/tez/reports/ozet/7_tum_modeller_800_gorev.png'), transformation: { width: 605, height: 327 } })], alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true }),
  new Paragraph({ children: runs('**Şekil 1.** 800 görev boyunca her görevdeki çevrimiçi doğruluk. Üçgen dalga ve Weight Clipping eğrileri üst üste biniyor ve düz kalıyor; standart ağ ve tanh yavaşça, sinüs hızlanarak düşüyor.', CAP), alignment: AlignmentType.CENTER, spacing: { before: 0, after: 160 } }),
  H('Üç bulgu'),
  P('**1. Üçgen dalga, Weight Clipping ile eşdeğer.** Önceden belirlenen Δ = 0.005 marjıyla yapılan TOST testi, 10 eşleştirilmiş tohumda dört ölçütün dördünde de eşdeğerlik veriyor. %90 güven aralıklarının tamamı ±0.003\'ün içinde kalıyor.'),
  table([2638, 1700, 2300, 1700, 1300], [
    ['Ölçüt', 'Fark (üçgen − kırpma)', '%90 güven aralığı', 'p (TOST, Δ = 0.005)', 'Karar'],
    ['Normalize AUC', '+0.0003', '[+0.0000, +0.0006]', '< 0.001', 'Eşdeğer'],
    ['Son pencere doğruluğu', '−0.0001', '[−0.0006, +0.0005]', '< 0.001', 'Eşdeğer'],
    ['Koruma oranı', '−0.0003', '[−0.0015, +0.0008]', '< 0.001', 'Eşdeğer'],
    ['Taze model farkı', '−0.0012', '[−0.0028, +0.0005]', '0.001', 'Eşdeğer'],
  ]), gap(),
  P('Katkı kırpmayı geçmek değil; aynı etkiyi ayrı bir izdüşüm adımı olmadan, türevi her yerde birim olan tek bir parametrizasyonla elde etmek ve pürüzsüz haritanın neden başarısız olduğunu açıklamak.'),
  P('**2. H3\'ün nedensel testi uzun ufukta da tutuyor.** Sinüs haritası aynen korunup yalnızca geri geçişte türevin büyüklüğü 1\'e sabitlendiğinde (yansımalı sinüs) AUC 0.818\'den 0.924\'e, koruma oranı 0.79\'dan 0.99\'a çıkıyor. Doyan parametre payı %78\'den %14\'e iniyor. Pürüzsüz sınırın başarısızlığının nedeni Jacobian sönümü; sınırlılık tek başına yeterli değil.'),
  P('**3. Üçgen dalga, Layer Norm + Weight Decay referansını geçiyor.** Fark AUC\'de +0.014 [+0.014, +0.015] ve on tohumun onunda aynı yönde. Layer Norm + Weight Decay standart ağın kaybının büyük bölümünü gideriyor ama eğrisi tam düz değil ve ölçütü 10 tohumun 8\'inde sağlıyor.'),
  H('Sınırlılıklar'),
  LI('**Yansımalı sinüs yalnızca 3 tohumla koşuldu.** Ayrıca pürüzsüz sinüsle hiperparametreleri de farklı (her biri kendi seçilmiş γ ve öğrenme hızıyla); aynı ayarla tek etkenli karşılaştırma yalnızca 200 görevde var.', 'bul'),
  LI('**Eşdeğerlik sonucu SGD için geçerli.** Adam altında sıralama değişiyor: sert sınırlar (üçgen dalga, kırpma) geriliyor, pürüzsüz sinüs öne geçiyor (3 tohum).', 'bul'),
  LI('**Fayda küçük ağa bağlı.** 3×2000 ağda 200 görevlik akışta standart ağ plastisite kaybetmiyor ve sınırlı haritalar fayda sağlamıyor (3 tohum). Geniş ağ × 800 görev kombinasyonu koşulmadı.', 'bul'),
  LI('**Tek veri kümesinde kanonik ölçek.** 800 görevlik sonuç yalnızca Permuted MNIST için; CIFAR-100 sonuçları küçültülmüş protokolden.', 'bul'),
  H('Görüşünüze sunulan sorular'),
  LI('Önce tez metnini mi, önce makaleyi mi yazmamı önerirsiniz?', 'num'),
  LI('Yukarıdaki sınırlılıklardan tez savunması veya makale için tamamlanmasını gerekli gördüğünüz var mı? En ucuzu yansımalı sinüsün 10 tohuma tamamlanması (yaklaşık 4 gün).', 'num'),
  LI('Makale için hedef olarak CoLLAs, TMLR ve RLC düşünülüyor; tercihiniz hangisi olur?', 'num'),
  LI('Makalenin ana mesajı olarak şunu öneriyorum: "Eleman bazında sınırlılık tek başına yeterli değildir; doymayan, parçalı doğrusal ve periyodik bir harita ağırlıkları sınırlı tutarken güncelleme büyüklüğünü korur." Uygun buluyor musunuz?', 'num'),
];
const lvl = (format, text) => [{ level: 0, format, text, alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } }, run: { font: FONT, size: BODY } } }];
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  numbering: { config: [{ reference: 'bul', levels: lvl(LevelFormat.BULLET, '•') }, { reference: 'num', levels: lvl(LevelFormat.DECIMAL, '%1.') }] },
  sections: [{ properties: { page: { margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } }, children }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(process.argv[2], b); console.log('ok', b.length); });
