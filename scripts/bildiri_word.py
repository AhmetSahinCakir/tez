# Ingilizce bildirinin (bildiri/paper_en.tex) Word'e elle aktarilmasi icin yardimci dosyalari uretir.
# Kullanim: python3 scripts/bildiri_word.py   (once: cd bildiri && latexmk -pdf paper_en.tex)
# Yazar:  bildiri/word/icerik.docx  (metin, Word denklemleri, tablolar, sekiller, kaynakca; kopyalama kaynagi)
#         bildiri/word/fig*.png     (600 dpi sekiller)
# Gerekir: pandoc, pdftoppm
import os, re, subprocess

B, OUT = "bildiri", "bildiri/word"
os.makedirs(OUT, exist_ok=True)
tex = open(f"{B}/paper_en.tex", encoding="utf-8").read()
bbl = open(f"{B}/paper_en.bbl", encoding="utf-8").read()

# ---------------------------------------------------------------- kaynak numaralari (bbl sirasi)
keys = re.findall(r"\\bibitem\{([^}]+)\}", bbl)
num = {k: i + 1 for i, k in enumerate(keys)}


def cite(m):
    ns = sorted(num[k.strip()] for k in m.group(1).split(","))
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out += [f"[{ns[i]}]--[{ns[j]}]"] if j - i >= 2 else [f"[{n}]" for n in ns[i:j + 1]]
        i = j + 1
    return ", ".join(out)


REF = {"fig:maps": "1", "fig:accuracy": "2", "fig:mechanism": "3", "tab:results": "I", "tab:mechanism": "II",
       "eq:reparam": "1", "eq:damping": "3", "sec:reflection": "III-C"}
FIG = {"fig_maps": "fig1_maps", "fig_accuracy": "fig2_accuracy", "fig_mechanism": "fig3_mechanism"}

# ---------------------------------------------------------------- sekiller: PDF -> 600 dpi PNG
for src, dst in FIG.items():
    subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", f"{B}/figures/{src}.pdf", f"{OUT}/{dst}"], check=True)

# ---------------------------------------------------------------- LaTeX on isleme
body = tex.split(r"\maketitle", 1)[1].split(r"\bibliographystyle", 1)[0]
title = re.search(r"\\title\{(.*?)\}\s*\n\s*\\author", tex, re.S).group(1).replace("\n", " ")
body = re.sub(r"\\cite\{([^}]+)\}", cite, body)
body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + REF[m.group(1)] + ")", body)
body = re.sub(r"\\ref\{([^}]+)\}", lambda m: REF[m.group(1)], body)
body = re.sub(r"\\label\{[^}]+\}", "", body)


def plain(m):  # yalnizca sayi iceren matematik -> duz metin (Word'de Times ile yazilsin, denklem nesnesi olmasin)
    t = m.group(1).replace(r"\pm", " ± ").replace(r"\to", " → ").replace("-", "−")
    return t.replace("+", "+").replace(",", ", ") if "," in t else t


NUMERIC = r"\$([+\-]?[0-9][0-9.,+\-\[\]]*(?:\\pm[0-9.]+|\\to[0-9.]+)?|\[[+\-0-9.,]+\])\$"
rows = lambda m: re.sub(NUMERIC, plain, open(f"{B}/{m.group(1)}", encoding="utf-8").read())
body = re.sub(NUMERIC, plain, body)
body = body.replace("$10^6$", r"10\textsuperscript{6}").replace("$n$: number", r"\emph{n}: number").replace(" & $n$ & ", r" & \emph{n} & ")
body = re.sub(r"\$p=([0-9.]+)\$", r"\\emph{p} = \1", body)
body = re.sub(r"\\tablerows\{([^}]+)\}", rows, body)
body = re.sub(r"\\resizebox\{\\linewidth\}\{!\}\{%\s*(\\begin\{tabular\}.*?\\end\{tabular\})\}", lambda m: m.group(1), body, flags=re.S)
body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]+\}\\footnotesize", "", body)
body = body.replace(r"\begin{IEEEkeywords}", r"\paragraph{Keywords}").replace(r"\end{IEEEkeywords}", "")
body = body.replace(r"\begin{abstract}", r"\paragraph{Abstract}").replace(r"\end{abstract}", "")
for src, dst in FIG.items():
    body = body.replace(f"figures/{src}.pdf", f"{dst}.png")
body = body.replace(r"\includegraphics[width=\linewidth]", r"\includegraphics[width=8.5cm]")
# denklem numaralari Word'de elle yazilir; hizali denklemi uc ayri satir olarak ver
body = body.replace(r", \nonumber\\", r"\\").replace(r"\nonumber", "")

refs = bbl.rsplit(r"\BIBdecl", 1)[1].replace(r"\end{thebibliography}", "")
refs = re.sub(r"\\bibitem\{([^}]+)\}", lambda m: "\n\n", refs)  # numara yok: sablonun "references" stili [1], [2]... numaralarini kendisi verir
refs = refs.replace(r"\,", "").replace(r"\BIBentryALTinterwordspacing", "").replace(r"\BIBentrySTDinterwordspacing", "")

doc = r"""\documentclass{article}
\usepackage{amsmath,amssymb,graphicx,booktabs}
\newcommand{\tri}{\operatorname{tri}}
\title{%s}
\begin{document}
\maketitle
%s
\section*{References}
%s
\end{document}
""" % (title, body, refs)
open(f"{OUT}/_icerik.tex", "w", encoding="utf-8").write(doc)
subprocess.run(["pandoc", "_icerik.tex", "-f", "latex", "-t", "docx", "-o", "icerik.docx"], cwd=OUT, check=True)
os.remove(f"{OUT}/_icerik.tex")
print("yazildi:", sorted(os.listdir(OUT)))
