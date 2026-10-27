"""prep_docx.py - turn main.tex into a pandoc-friendly LaTeX file with resolved numbers (from main.aux/main.bbl)."""
import re
src = open("main.tex").read(); aux = open("main.aux").read(); bbl = open("main.bbl").read()
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{(.*?)\}\{", aux):
    labels[m.group(1)] = m.group(2).replace("\\mbox  {", "").replace("{", "").replace("}", "").strip()
cites = dict(re.findall(r"\\bibcite\{([^}]*)\}\{(\d+)\}", aux))
s = src
# body only + simple title block
body = s[s.index(r"\maketitle") + len(r"\maketitle"):s.index(r"\bibliographystyle")]
# citations
body = re.sub(r"~?\\cite\{([^}]*)\}", lambda m: " " + ", ".join(f"[{cites[k.strip()]}]" for k in m.group(1).split(",")), body)
# references
body = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], body)
# captions: prefix numbers
def cap(env, prefix):
    global body
    def f(m):
        blk = m.group(0)
        lab = re.search(r"\\label\{([^}]*)\}", blk)
        if not lab: return blk
        n = labels[lab.group(1)]
        return blk.replace(r"\caption{", r"\caption{" + f"{prefix} {n}. ", 1)
    body = re.sub(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", f, body, flags=re.S)
cap("figure", "Fig."); cap("table", "TABLE")
# tables: put the caption above as a paragraph so Word shows it over the table
def tabcap(m):
    blk = m.group(0); c = re.search(r"\\caption\{(.*?)\}\\label", blk, re.S)
    if not c: return blk
    inner = blk.replace(c.group(0), r"\label")
    inner = re.sub(r"\\begin\{table\}\[t\]|\\end\{table\}|\\centering|\\label\{[^}]*\}|\\setlength\{\\tabcolsep\}\{[^}]*\}", "", inner)
    return "\n\n\\textbf{" + c.group(1) + "}\n\n" + inner + "\n\n"
body = re.sub(r"\\begin\{table\}\[t\].*?\\end\{table\}", tabcap, body, flags=re.S)
# table-safe text for pandoc: unicode minus, no @{} in column specs, plain p*
body = body.replace("$-$", "\u2212").replace("$\\pm0.8\\%$", "\u00b10.8\\%")
body = body.replace("@{}", "").replace("log-AR($p^*$)", "log-AR(p*)")
body = re.sub(r"\\multicolumn\{(\d)\}\{l\}", r"\\multicolumn{\1}{l}", body)
# multirow -> plain
body = re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", r"\1", body)
# algorithm -> numbered list
def alg(m):
    blk = m.group(0); c = re.search(r"\\caption\{(.*?)\}", blk).group(1)
    steps = re.findall(r"\\State (.*)", blk)
    items = "\n".join(r"\item " + st.replace(r"\quad ", r"\hspace{1em}") for st in steps)
    return f"\n\n\\textbf{{Algorithm {labels['alg:ctl']}. {c}}}\n\n\\begin{{enumerate}}\n{items}\n\\end{{enumerate}}\n\n"
body = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", alg, body, flags=re.S)
# keywords
body = body.replace(r"\begin{IEEEkeywords}", r"\noindent\textbf{Index Terms---}").replace(r"\end{IEEEkeywords}", "")
body = body.replace(r"\section*{Data and Code Availability}", r"\section{Data and Code Availability}")
# bibliography from .bbl
items = re.split(r"\\bibitem\{([^}]*)\}", bbl)[1:]
refs = []
for k, txt in zip(items[0::2], items[1::2]):
    txt = txt.replace(r"\end{thebibliography}", "")
    txt = re.sub(r"\\hskip[^\\]*\\relax", " ", txt)
    txt = re.sub(r"\\BIBentry(ALT|STD)interwordspacing", "", txt)
    txt = txt.replace(r"\newblock", "").replace("~", " ")
    refs.append("{[}" + cites[k] + "{]}~" + " ".join(txt.split()))
bib = "\n\\section{References}\n\n" + "\n\n".join(r"\noindent{}" + r for r in refs) + "\n"
head = r"""\documentclass{article}
\usepackage{amsmath,amssymb,graphicx,booktabs,url,hyperref}
\newcommand{\K}{\mathbf{K}}
\graphicspath{{figs/}}
\title{Steering the Linux Scheduler with Koopman Burst Prediction under EEVDF}
\author{Saarthak Manocha, Prathima Tirumalareddy and A Sirisha\\Chaitanya Bharathi Institute of Technology, Hyderabad, India\\saarthakmanocha@gmail.com, prathimareddy.t@gmail.com, asirishanaidu@gmail.com}
\date{}
\begin{document}
\maketitle
"""
out = head + body + bib + "\n\\end{document}\n"
# section numbering in Word: pandoc --number-sections uses arabic; keep LaTeX roman refs consistent by numbering headings ourselves
open("word/main_docx.tex", "w").write(out)
print("labels", len(labels), "cites", len(cites), "refs", len(refs))

# number headings like IEEEtran: I., II. ...; subsections A., B. ...
def roman(n):
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]; r = ""
    for v, c in vals:
        while n >= v: r += c; n -= v
    return r
s2 = open("word/main_docx.tex").read()
sec = sub = 0
def num(m):
    global sec, sub
    kind, title = m.group(1), m.group(2)
    if title in ("References", "Data and Code Availability"):
        return "\\" + kind + "*{" + title + "}"
    if kind == "section":
        sec += 1; sub = 0; return "\\section*{" + roman(sec) + ". " + title + "}"
    sub += 1; return "\\subsection*{" + chr(64 + sub) + ". " + title + "}"
s2 = re.sub(r"\\(section|subsection)\{(.*?)\}(?=\\label|\n|$)", num, s2)
s2 = re.sub(r"\\(section|subsection)\*\{(.*?)\}\\label\{[^}]*\}", r"\\\1*{\2}", s2)
open("word/main_docx.tex", "w").write(s2)
print("sections", sec)
