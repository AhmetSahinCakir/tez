# Ingilizce bildiriyi IISEC'in resmi Word sablonuna (IEEE konferans sablonu, A4) yerlestirir.
# Kullanim: python3 scripts/bildiri_word.py && python3 scripts/bildiri_word_sablon.py
# Okur:  bildiri/word/template-a4.docx  (https://iisec.tbdakademi.org.tr/2026/format/template-a4.docx, degistirilmemis)
#        bildiri/word/icerik.docx       (bildiri_word.py ciktisi: metin, Word denklemleri, tablolar)
#        bildiri/word/fig*.png
# Yazar: bildiri/word/iisec2027_paper.docx
# Sablonun stilleri, numaralandirmasi, sayfa ve sutun ayarlari aynen kalir; sablonun kilavuz metinlerinin tamami silinir.
import copy, datetime, os, re, shutil, struct, sys, tempfile, zipfile
from lxml import etree
from PIL import ImageFont

D = "bildiri/word"
TEMPLATE, CONTENT, OUT = f"{D}/template-a4.docx", f"{D}/icerik.docx", f"{D}/iisec2027_paper.docx"
FIGS = ["fig1_maps.png", "fig2_accuracy.png", "fig3_mechanism.png"]
ANON = "Author information removed for double-blind review"
COLW = 4866          # sutun genisligi (twip): (11906 - 2*907 - 360) / 2
FIGW = 3060000       # sekil genisligi: 8.5 cm (EMU)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
NS = {"w": W, "m": M}
w = lambda t: f"{{{W}}}{t}"

PPR_ORDER = ["pStyle", "keepNext", "keepLines", "numPr", "tabs", "spacing", "ind", "jc", "rPr", "sectPr"]


def E(tag, **attrs):
    e = etree.Element(w(tag))
    for k, v in attrs.items():
        e.set(w(k), str(v))
    return e


def para(style=None, children=(), **ppr):
    """Sablon stiliyle paragraf; ppr: keepNext=True, spacing={...}, ind={...}, jc='center', tabs=[...], sectPr=<el>"""
    p = etree.Element(w("p"))
    pPr = etree.SubElement(p, w("pPr"))
    items = dict(ppr)
    if style: items["pStyle"] = style
    for key in PPR_ORDER:
        if key not in items: continue
        v = items[key]
        if key == "pStyle": pPr.append(E("pStyle", val=v))
        elif key in ("keepNext", "keepLines"): pPr.append(E(key))
        elif key == "tabs":
            t = etree.SubElement(pPr, w("tabs"))
            for val, pos in v: t.append(E("tab", val=val, pos=pos))
        elif key in ("spacing", "ind"): pPr.append(E(key, **v))
        elif key == "jc": pPr.append(E("jc", val=v))
        else: pPr.append(v)  # rPr, sectPr hazir eleman
    for c in children: p.append(c)
    return p


def run(text, bold=None, italic=False, font=None, size=None):
    r = etree.Element(w("r"))
    rPr = etree.SubElement(r, w("rPr"))
    if font: rPr.append(E("rFonts", ascii=font, hAnsi=font, cs=font))
    if bold is not None: rPr.append(E("b") if bold else E("b", val="0"))
    if italic: rPr.append(E("i")); rPr.append(E("iCs"))
    if size: rPr.append(E("sz", val=size)); rPr.append(E("szCs", val=size))
    rPr.append(E("lang", val="en-US"))
    t = etree.SubElement(r, w("t")); t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return r


def tab():
    r = etree.Element(w("r")); r.append(E("tab")); return r


def content(p):
    """Pandoc paragrafinin icerigi (calisma parcalari ve satir ici denklemler); dil Ingilizce olarak isaretlenir."""
    out = []
    for c in p:
        tag = etree.QName(c).localname
        if tag == "r":
            c = copy.deepcopy(c)
            rPr = c.find("w:rPr", NS)
            if rPr is None:
                rPr = etree.Element(w("rPr")); c.insert(0, rPr)
            if rPr.find("w:lang", NS) is None: rPr.append(E("lang", val="en-US"))
            for t in c.iter(w("t")):                  # "(ii-iv)" gibi parantezlerden sonraki bolunmez bosluk: iki yana yasli satiri germesin
                if t.text: t.text = t.text.replace(")\u00a0", ") ")
            out.append(c)
        elif tag == "oMath" and text_of(c).strip() == "\u00b1":
            out.append(run("\u00b1"))                                 # tek basina arti-eksi: duz metin
        elif tag == "oMath":
            out.append(fix_math(copy.deepcopy(c)))
        elif tag == "oMathPara":
            out += [fix_math(copy.deepcopy(m)) for m in c.findall("m:oMath", NS)]
    return out


M_ORDER = {"dPr": ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"], "mcPr": ["count", "mcJc"],
           "rPr": ["lit", "nor", "scr", "sty", "brk", "aln"]}


def fix_math(el):
    """Pandoc'un denklem ozelliklerini Word semasinin istedigi siraya koyar (icerik degismez)."""
    for pr in el.iter(*[f"{{{M}}}{k}" for k in M_ORDER]):
        order = M_ORDER[etree.QName(pr).localname]
        kids = sorted(pr, key=lambda c: order.index(etree.QName(c).localname))
        if any(etree.QName(c).localname == "nor" for c in kids):      # duz metin: "nor" ile "sty" birlikte olmaz
            kids = [c for c in kids if etree.QName(c).localname != "sty"]
        for c in list(pr): pr.remove(c)
        for c in kids: pr.append(c)
    for r in el.iter(f"{{{M}}}r"):                                    # denklem icindeki duz metin ("sine:" vb.)
        mr = r.find("m:rPr", NS)
        if mr is not None and mr.find("m:nor", NS) is not None:
            for t in r.iter(f"{{{M}}}t"): t.text = t.text.replace("\u2001", "")    # ek bosluk gereksiz: sutun araligi yeter
            if not text_of(r):
                r.getparent().remove(r); continue
            wr = etree.Element(w("rPr"))                              # "equation" stilinin Symbol yazi tipi Yunan harfi yapmasin
            wr.append(E("rFonts", ascii="Times New Roman", hAnsi="Times New Roman", cs="Times New Roman"))
            mr.addnext(wr)
    for r in el.iter(f"{{{M}}}r"):                                    # cift cizgili R: dogrudan U+211D karakteri
        scr = r.find("m:rPr/m:scr", NS)
        if scr is not None and scr.get(f"{{{M}}}val") == "double-struck":
            scr.getparent().remove(scr)
            for t in r.iter(f"{{{M}}}t"): t.text = t.text.replace("R", "\u211d")
    FUNCS = ("sin", "cos", "tanh")                                    # "sinx" yerine "sin x": islev adindan sonra ince bosluk
    for r in list(el.iter(f"{{{M}}}r", f"{{{M}}}sSup")):
        nxt = r.getnext()
        if nxt is None or etree.QName(nxt).localname != "r" or not text_of(nxt).strip(): continue
        if etree.QName(r).localname == "r" and text_of(r) in FUNCS:
            r.find("m:t", NS).text += "\u2009"
        elif etree.QName(r).localname == "sSup" and text_of(r.find("m:e", NS)) in FUNCS:
            sp = etree.Element(f"{{{M}}}r"); t = etree.SubElement(sp, f"{{{M}}}t"); t.text = "\u2009"
            r.addnext(sp)
    for mat in el.iter(f"{{{M}}}m"):                                  # hizali denklem: "f'(x)" ve "= ..." hucrelerini birlestir
        for r in list(mat.iter(f"{{{M}}}r")):                         # etiketten sonraki ek bosluk: sutun araligi yeter
            if text_of(r) and not text_of(r).strip("\u2001"): r.getparent().remove(r)
        rows = mat.findall("m:mr", NS)
        if all(len(r.findall("m:e", NS)) == 4 for r in rows):
            for r in rows:
                es = r.findall("m:e", NS)
                for c in list(es[3]): es[2].append(c)
                r.remove(es[3])
            mcs = mat.find("m:mPr/m:mcs", NS)
            mcs.remove(mcs.findall("m:mc", NS)[3])
            mcs.findall("m:mc", NS)[2].find("m:mcPr/m:mcJc", NS).set(f"{{{M}}}val", "left")
    return el


def text_of(el):
    return "".join(el.itertext())


# ---------------------------------------------------------------- tablolar
FONT = {b: ImageFont.truetype(f"/usr/share/fonts/truetype/liberation/LiberationSerif-{n}.ttf", 160)
        for b, n in ((False, "Regular"), (True, "Bold"))}  # Times New Roman ile ayni olcu; 160 birim = 8 pt
PAD = 40   # hucre ic boslugu (twip, her yan)
TSZ = 16   # tablo yazisi (yarim punto): sablonun "table copy" / "table col head" stillerindeki 8 punto


def twips(s, bold=False):
    return FONT[bold].getlength(s) * TSZ / 16  # 160 birim = 8 pt = 160 twip


def small(p):
    """Hucre yazisini TSZ boyutuna getirir (rPr icinde semanin istedigi sirayla)."""
    if TSZ == 16: return p                      # stilin kendi boyutu: dogrudan bicim eklenmez
    for r in p.iter(w("r")):
        rPr = r.find("w:rPr", NS)
        if rPr is None:
            rPr = etree.Element(w("rPr")); r.insert(0, rPr)
        at = next((i for i, c in enumerate(rPr) if etree.QName(c).localname in ("vertAlign", "lang")), len(rPr))
        rPr.insert(at, E("szCs", val=TSZ)); rPr.insert(at, E("sz", val=TSZ))
    return p


def table(tbl, borders):
    rows = [[tc.find("w:p", NS) for tc in tr.findall("w:tc", NS)] for tr in tbl.findall("w:tr", NS)]
    for r in rows:                                   # sutuna sigmasi icin: "0.836 ± 0.002" -> "0.836±0.002"
        for p in r:
            for t in p.iter(w("t")): t.text = (t.text or "").replace(" \u00b1 ", "\u00b1")
    while len(rows) > 1 and not text_of(rows[1][0]).strip():   # iki satirli baslik -> tek baslik satiri
        for p0, p1 in zip(rows[0], rows.pop(1)):
            if text_of(p1).strip():
                p0.append(run(" "))
                for c in content(p1): p0.append(c)
    ncol = len(rows[0])
    need = []
    for j in range(ncol):
        head = max(twips(x, True) for x in text_of(rows[0][j]).split())   # baslik satiri kelimelerden bolunebilir
        cells = max(twips(text_of(r[j])) for r in rows[1:])               # veri satirlari tek satir
        need.append(max(head, cells) + 2 * PAD + 10)
    extra = COLW - sum(need)
    if extra < 0: sys.exit(f"tablo sutuna sigmiyor: {sum(need):.0f} > {COLW}")
    widths = [int(n + extra / ncol) for n in need]
    widths[0] += COLW - sum(widths)
    t = etree.Element(w("tbl"))
    pr = etree.SubElement(t, w("tblPr"))
    pr.append(E("tblW", w=COLW, type="dxa")); pr.append(E("jc", val="center")); pr.append(copy.deepcopy(borders))
    pr.append(E("tblLayout", type="fixed"))
    mar = etree.SubElement(pr, w("tblCellMar"))
    mar.append(E("left", w=PAD, type="dxa")); mar.append(E("right", w=PAD, type="dxa"))
    grid = etree.SubElement(t, w("tblGrid"))
    for x in widths: grid.append(E("gridCol", w=x))
    for i, r in enumerate(rows):
        tr = etree.SubElement(t, w("tr"))
        trPr = etree.SubElement(tr, w("trPr")); trPr.append(E("cantSplit"))
        if i == 0: trPr.append(E("tblHeader"))
        trPr.append(E("jc", val="center"))
        for j, p in enumerate(r):
            tc = etree.SubElement(tr, w("tc"))
            tcPr = etree.SubElement(tc, w("tcPr")); tcPr.append(E("tcW", w=widths[j], type="dxa")); tcPr.append(E("vAlign", val="center"))
            style = "tablecolhead" if i == 0 else "tablecopy"
            tc.append(small(para(style, content(p), jc="left" if (j == 0 and i > 0) else "center")))
    return t


# ---------------------------------------------------------------- sekiller
def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def picture(rid, n, path, alt):
    px, py = png_size(path)
    cx = min(FIGW, int(px / 600 * 914400))            # sekiller 600 dpi: yazi boyutlari (7-8 pt) korunur
    cy = int(cx * py / px)
    xml = f"""<w:r xmlns:w="{W}" xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}" xmlns:r="{R}"><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
<wp:docPr id="{n}" name="Figure {n}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic><a:graphicData uri="{PIC}"><pic:pic><pic:nvPicPr><pic:cNvPr id="{n}" name="{os.path.basename(path)}"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"""
    r = etree.fromstring(re.sub(r">\s+<", "><", xml))
    r.find(f".//{{{WP}}}docPr").set("descr", alt)
    return r


# ---------------------------------------------------------------- sablonu ac
tmp = tempfile.mkdtemp()
zipfile.ZipFile(TEMPLATE).extractall(tmp)
doc = etree.parse(f"{tmp}/word/document.xml")
body = doc.getroot().find("w:body", NS)
old = list(body)
src = etree.parse(zipfile.ZipFile(CONTENT).open("word/document.xml")).getroot().find("w:body", NS)

# sablondan korunan parcalar: baslik paragrafinin bicimi, uc bolum ayari, ornek tablonun kenarliklari
title_pPr = copy.deepcopy(old[0].find("w:pPr", NS))
sect_title = copy.deepcopy(old[3].find("w:pPr/w:sectPr", NS))            # baslik bolumu (tek sutun, ilk sayfa alt bilgisi)
sect_author = copy.deepcopy(old[8].find("w:pPr/w:sectPr", NS))           # yazar bolumu
sect_body = copy.deepcopy([p for p in old if p.find("w:pPr/w:sectPr/w:cols[@w:num='2']", NS) is not None][0].find("w:pPr/w:sectPr", NS))
sect_last = copy.deepcopy(old[-1])                                        # belge sonu bolumu
borders = [e for e in old if etree.QName(e).localname == "tbl"][0].find("w:tblPr/w:tblBorders", NS)
assert etree.QName(sect_last).localname == "sectPr" and sect_body is not None
sect_author.find("w:cols", NS).attrib.pop(w("num"))                       # kor hakemlik: tek satir, tek sutun
for e in old: body.remove(e)                                              # sablonun butun metni silinir

# ---------------------------------------------------------------- icerigi yerlestir
new, rels, eq, fig, prev = [], [], 0, 0, None
STYLE = {"Heading1": "Balk1", "Heading2": "Balk2"}
state, wait, held = None, False, []       # wait: basliktan sonra henuz metin gelmedi; held: bekletilen sekil/tablolar


def floaty(el):
    (held if wait else new).append(el)    # IEEE: sekil ve tablo, metinde anildiktan sonra yerlestirilir


for e in src:
    tag = etree.QName(e).localname
    if tag == "tbl":
        floaty(table(e, borders))
        if not wait: prev = "tbl"
        continue
    if tag != "p": continue
    st = e.find("w:pPr/w:pStyle", NS); st = st.get(w("val")) if st is not None else ""
    txt = text_of(e).strip()
    if st == "Title":
        p = etree.Element(w("p")); p.append(title_pPr)
        for c in content(e): p.append(c)
        new.append(p)
        new.append(para("Author", spacing=dict(before="0", after="0"), sectPr=sect_title))
        new.append(para("Author", [run(ANON)]))
        new.append(para(None, sectPr=sect_author))
    elif st == "Heading4":
        state = txt                                    # "Abstract" / "Keywords": sonraki paragrafin basina eklenir
    elif st in STYLE:
        assert not held, "bekleyen sekil/tablo yerlestirilmedi"
        if txt == "References":
            new.append(para("Balk5", content(e))); state = "refs"
        else:
            new.append(para(STYLE[st], content(e))); wait = True
    elif state == "Abstract":
        new.append(para("Abstract", [run("Abstract", italic=True), run("\u2014")] + content(e))); state = None
    elif state == "Keywords":
        new.append(para("Keywords", [run("Keywords\u2014")] + content(e))); state = None
    elif state == "refs":
        new.append(para("references", content(e)))
    elif e.find("m:oMathPara", NS) is not None:
        eq += 1
        new.append(para("equation", [tab()] + content(e) + [tab(), run(f"({eq})", font="Times New Roman")],
                        tabs=[("clear", 2520), ("clear", 5040), ("center", COLW // 2), ("right", COLW)]))
        prev = "eq"; continue
    elif st == "CaptionedFigure":
        fig += 1
        rid = f"rIdFig{fig}"; rels.append((rid, FIGS[fig - 1]))
        cap = e.getnext()
        floaty(para(None, [picture(rid, fig, f"{D}/{FIGS[fig - 1]}", text_of(cap).strip())], keepNext=True,
                    spacing=dict(before="120", after="0"), jc="center"))
    elif st == "ImageCaption":
        floaty(para("figurecaption", content(e)))
    elif st == "TableCaption":
        floaty(para("tablehead", content(e), keepNext=True))
    else:
        extra = {}
        if prev == "eq" and st == "FirstParagraph": extra["ind"] = dict(firstLine="0")   # denklemden sonra ayni paragraf suruyor
        if prev == "tbl": extra["spacing"] = dict(before="160")
        new.append(para("GvdeMetni", content(e), **extra))
        wait = False
        if held:
            prev = "tbl" if etree.QName(held[-1]).localname == "tbl" else None
            new += held; held = []
            continue
    prev = None

assert eq == 3 and fig == 3, (eq, fig)
refs = [p for p in new if p.find("w:pPr/w:pStyle[@w:val='references']", NS) is not None]
nref = len(re.findall(r"\\bibitem\{", open("bildiri/paper_en.bbl", encoding="utf-8").read()))
assert len(refs) == nref, (len(refs), nref)   # LaTeX kaynakcasiyla ayni sayida kaynak
refs[-1].find("w:pPr", NS).append(sect_body)           # iki sutunlu govde burada biter (son sayfada sutunlar dengelenir)
tiny = etree.Element(w("rPr")); tiny.append(E("sz", val=2)); tiny.append(E("szCs", val=2))
new.append(para(None, spacing=dict(before="0", after="0", line="20", lineRule="exact"), rPr=tiny))
for p in new: body.append(p)
body.append(sect_last)
doc.write(f"{tmp}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# ---------------------------------------------------------------- sekil dosyalari ve iliskiler
os.makedirs(f"{tmp}/word/media", exist_ok=True)
rp = f"{tmp}/word/_rels/document.xml.rels"
x = open(rp, encoding="utf-8").read()
for rid, name in rels:
    shutil.copy(f"{D}/{name}", f"{tmp}/word/media/{name}")
    x = x.replace("</Relationships>", f'<Relationship Id="{rid}" Type="{R}/image" Target="media/{name}"/></Relationships>')
open(rp, "w", encoding="utf-8").write(x)
cp = f"{tmp}/[Content_Types].xml"
x = open(cp, encoding="utf-8").read()
if 'Extension="png"' not in x:
    x = x.replace("<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1)
open(cp, "w", encoding="utf-8").write(x)

# Word resimleri kaydederken/PDF'e cevirirken kucultmesin ("Do not compress images in file")
sp = f"{tmp}/word/settings.xml"
x = open(sp, encoding="utf-8").read()
if "doNotAutoCompressPictures" not in x:
    for anchor in ("<w:shapeDefaults>", "<w:decimalSymbol "):
        if anchor in x:
            x = x.replace(anchor, "<w:doNotAutoCompressPictures/>" + anchor, 1); break
    else: sys.exit("settings.xml: yer bulunamadi")
open(sp, "w", encoding="utf-8").write(x)

# ---------------------------------------------------------------- belge ozellikleri: sablon metni ve kisi adi kalmasin
title = text_of(new[0])
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
open(f"{tmp}/docProps/core.xml", "w", encoding="utf-8").write(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    f'<dc:title>{title}</dc:title><dc:creator></dc:creator><cp:lastModifiedBy></cp:lastModifiedBy>'
    f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
    f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>')
open(f"{tmp}/docProps/app.xml", "w", encoding="utf-8").write(
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
    '<Application>Microsoft Office Word</Application></Properties>')

# ---------------------------------------------------------------- paketle
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    names = sorted(os.path.relpath(os.path.join(dp, f), tmp) for dp, _, fs in os.walk(tmp) for f in fs)
    names.sort(key=lambda n: n != "[Content_Types].xml")
    for n in names: z.write(os.path.join(tmp, n), n)
shutil.rmtree(tmp)
print("yazildi:", OUT, f"({len(new)} paragraf/tablo, {eq} denklem, {fig} sekil, {len(refs)} kaynak)")
