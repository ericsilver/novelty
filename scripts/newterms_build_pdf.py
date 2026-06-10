"""Build the newterms_report.pdf: cover + methods + overall chart + 10 line charts + top-100 longtable."""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\shared-secure\research\novelty")
PAPER = REPO / "paper"
RESULTS = PAPER / "results"
TOP10_DIR = RESULTS / "newterms_top10"
CSV = RESULTS / "newterms_top100.csv"
TEX = PAPER / "newterms_report.tex"
PDF = PAPER / "newterms_report.pdf"


def latex_escape(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&")
                   .replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")
                   .replace("%", "\\%").replace("{", "\\{").replace("}", "\\}"))


def main() -> int:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    print(f"loaded {len(rows)} rows", file=sys.stderr)

    # ----- longtable rows -----
    longtable_rows = []
    for r in rows:
        ys = " / ".join(f"{int(r[f'y{i}']):,}" for i in range(1, 11))
        longtable_rows.append(
            f"{latex_escape(r['theme'])} & {r['first_year']} & "
            f"{int(r['origin_klass']):03d} {latex_escape(r['origin_industry'])[:18]} & "
            f"{int(r['origin_5y']):,} & "
            f"\\scriptsize{{{ys}}} & "
            f"{float(r['other_hhi']):.2f} & "
            f"{int(r['other_total']):,} & "
            f"{int(r['total_study_cnt']):,} \\\\"
        )

    # ----- top-10 charts list (in order) -----
    top10 = rows[:10]
    chart_blocks = []
    for r in top10:
        safe = r["theme"].replace("/", "_").replace(" ", "_")[:40]
        path = TOP10_DIR / f"{safe}.png"
        if not path.exists():
            continue
        chart_blocks.append(
            "\\begin{figure}[H]\\centering\n"
            f"\\includegraphics[width=0.92\\textwidth]{{results/newterms_top10/{safe}.png}}\n"
            f"\\caption{{\\textbf{{{latex_escape(r['theme'])}}} (born {r['first_year']}, origin "
            f"{int(r['origin_klass']):03d} {latex_escape(r['origin_industry'])}). "
            f"Total study-period registrations: {int(r['total_study_cnt']):,}; "
            f"HHI across other industries {float(r['other_hhi']):.2f}.}}\n"
            "\\end{figure}\n"
        )

    tex = r"""
\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{caption}
\usepackage{float}
\usepackage{pdflscape}
\usepackage[hidelinks]{hyperref}

\title{\bfseries New trademark vocabulary, 1995--2021\\
\large Tokens introduced after a 1990--1994 burn-in, where they originated,\\ and how they spread across industries}
\author{Eric Silver}
\date{June 2026 --- working note}

\begin{document}
\maketitle

\section*{What this is}
A pass over the full cross-industry token panel
(\texttt{\_cct\_year\_class\_token\_counts.parquet}, 10{,}617{,}183 rows;
1990--2021; 42{,}645 distinct tokens across 45 NICE classes) identifying
vocabulary that was introduced \emph{after} a 1990--1994 burn-in and that
crossed a substantive-use threshold in the study period 1995--2021. The
panel reports, for each (year, NICE class, token), the number of granted
filings whose goods/services text contained that token.

\paragraph{New-term definition.} A token qualifies if its total count over
1990--1994 is exactly 0, its total count over 1995--2021 is at least 200,
and it is present in at least 5 distinct study-period years.
\textbf{Result:} 3{,}653 tokens meet the criteria. The top 100 by total
study-period count are below.

\paragraph{Origin industry.} The NICE class with the largest count for the
token in its first two appearance years. This is informative but noisy at
very small first-window counts (e.g., \textbf{e-commerce} shows origin
``Household Utensils 21'' on just 2 stray filings in 1995--1996 before
1997's class 9/42 explosion --- the \texttt{origin\_5y=2} next to
\texttt{other\_total=146k} makes the noise visible).

\paragraph{Truncation.} Terms born $\geq$2013 have trailing zero
\texttt{y8}/\texttt{y9}/\texttt{y10} entries because the panel ends in 2021;
this is data-truncation, not absence (e.g., blockchain $y_8 = 29{,}314$,
$y_9 = y_{10} = 0$).

\paragraph{HHI.} Herfindahl on non-origin-industry shares. Higher = adoption
concentrated in a small number of other industries; lower = adoption spread.

\clearpage
\section*{Overall: top-10 themes' total utilization}
\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{results/newterms_overall.png}
\caption{Total registrations across all industries, 1995--2021, for the top
10 new themes by study-period total. Bar color = origin industry.}
\end{figure}

\clearpage
\section*{Per-theme yearly registrations by industry (top 10)}

""" + "\n".join(chart_blocks) + r"""

\clearpage
\begin{landscape}
\section*{Top 100 new themes (full table)}
{\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{@{}l c p{2.6cm} r p{6.2cm} r r r@{}}
\toprule
theme & born & origin industry & 5y orig & y1 / y2 / \dots / y10 (count) & HHI\textsubscript{other} & other total & total \\
\midrule
\endhead
""" + "\n".join(longtable_rows) + r"""
\bottomrule
\end{longtable}
}
\end{landscape}

\end{document}
"""
    TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {TEX}", file=sys.stderr)

    # compile twice for refs
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(TEX.name)],
            cwd=str(PAPER), capture_output=True, text=True,
        )
    if not PDF.exists():
        print("COMPILE FAILED; last stdout:", file=sys.stderr)
        print(r.stdout[-3000:], file=sys.stderr)
        return 2
    print(f"compiled -> {PDF}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
