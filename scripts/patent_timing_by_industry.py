"""Patent timing and industry ordering: does patenting PRECEDE forward-leaning
trademark language, and does the industry ordering look like IPR complementarity?

Motivation. A patent must be applied for before the invention is in public use,
while a trademark is filed when the firm is ready to commercialise. If dKL and
patenting were both tracking one underlying innovation event, patents should
therefore lead trademark language rather than move with it. The contemporaneous
regression in scripts/patent_complementarity_by_sector.py cannot see a lead, so
this script re-estimates the same within-firm specification with patents entered
at t-2, t-1, t, and t+1 (t+1 is a reverse-timing placebo).

Lags are built by joining on the patent year rather than by positional shift, so
gap years in a firm's filing history do not silently mis-align.

The industry block re-estimates the lead specification separately in every NICE
class and orders the classes by coefficient, so the ordering can be inspected
directly rather than through an a-priori complement/substitute grouping.

Inputs :  as scripts/patent_complementarity_by_sector.py (imported)
Output :  paper/results/patent_timing_by_industry.json
          paper/results/patent_timing_note.{tex,pdf}
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from patent_complementarity_by_sector import (  # noqa: E402
    COMPLEMENT_CLASSES, SUBSTITUTE_CLASSES, MIN_FIRMS, MIN_OBS,
    owner_year_class_panel, dominant_class, collapse_owner_year,
    load_patents, two_way_fe, nice_group,
)

NICE_LABELS = {
    "001": "Chemicals", "002": "Paints", "003": "Cosmetics", "004": "Lubricants",
    "005": "Pharma", "006": "Metals", "007": "Machines", "008": "Hand tools",
    "009": "Software/Electronics", "010": "Medical devices", "011": "Lighting/HVAC",
    "012": "Vehicles", "013": "Firearms", "014": "Jewelry", "015": "Music",
    "016": "Paper", "017": "Rubber/Plastics", "018": "Leather", "019": "Building",
    "020": "Furniture", "021": "Housewares", "022": "Cordage", "023": "Yarns",
    "024": "Textiles", "025": "Clothing", "026": "Lace", "027": "Carpets",
    "028": "Toys/Games", "029": "Meat/Foods", "030": "Coffee/Staples",
    "031": "Agriculture", "032": "Beverages", "033": "Wine/Spirits",
    "034": "Tobacco", "035": "Advertising/Retail", "036": "Insurance/Finance",
    "037": "Construction", "038": "Telecom", "039": "Transport",
    "040": "Materials treatment", "041": "Education/Entertainment",
    "042": "Scientific/Tech services", "043": "Food services",
    "044": "Medical services", "045": "Legal/Personal services",
}

# k = number of years by which patents PRECEDE the trademark filing year.
LAGS = [2, 1, 0, -1]
LAG_LABEL = {
    2: "Patents at $t-2$",
    1: "Patents at $t-1$ (patents lead)",
    0: "Patents at $t$ (contemporaneous)",
    -1: "Patents at $t+1$ (reverse-timing placebo)",
}
LEAD = 1  # the specification the industry block uses


def frames_by_lag(kind: str = "t200"):
    """Owner-year frames for each lag, with dkl standardized ONCE on the
    contemporaneous panel so every coefficient is in the published sigma units."""
    oyc = owner_year_class_panel(kind)
    dom = dominant_class(oyc).to_pandas()
    oy = collapse_owner_year(oyc).to_pandas()
    pat = load_patents()

    base = oy.merge(pat, left_on=["owner_name", "year"],
                    right_on=["uspto_owner_name", "year"], how="inner")
    mu, sd = float(base.mean_dkl.mean()), float(base.mean_dkl.std())

    out = {}
    for k in LAGS:
        p = pat.copy()
        # a patent granted in year Y is entered against the filing year Y + k
        p["year"] = p["year"] + k
        m = oy.merge(p, left_on=["owner_name", "year"],
                     right_on=["uspto_owner_name", "year"], how="inner")
        m["logpat"] = np.log1p(m.n_patents)
        m["dkl_z"] = (m.mean_dkl - mu) / sd
        keep = m.groupby("owner_name").year.transform("count") >= 3
        f = m[keep].copy()
        f = f.merge(dom[["owner_name", "nice_class"]], on="owner_name", how="left")
        f["nice_group"] = f.nice_class.map(nice_group)
        f["is_complement"] = (f.nice_group == "complement").astype(float)
        f["logpat_x_comp"] = f.logpat * f.is_complement
        out[k] = f
    return out, sd


def fe(df, xvars=("logpat",)):
    return two_way_fe(df, "dkl_z", list(xvars), "owner_name", "year")


def guarded(df, xvars=("logpat",)):
    d = df[["owner_name", "year", "dkl_z"] + list(xvars)].dropna()
    if d.owner_name.nunique() < MIN_FIRMS or len(d) < MIN_OBS:
        return {"skipped": "cell too small", "n_obs": int(len(d)),
                "n_firms": int(d.owner_name.nunique())}
    try:
        return fe(df, xvars)
    except Exception as exc:
        return {"skipped": f"{type(exc).__name__}: {exc}",
                "n_obs": int(len(d)), "n_firms": int(d.owner_name.nunique())}


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def build(kind: str = "t200") -> dict:
    frames, sd = frames_by_lag(kind)
    res: dict = {"scoring": kind, "sigma_dkl": sd, "timing": {}, "groups": {},
                 "interaction": {}, "by_class": {}}

    for k, f in frames.items():
        r = fe(f)
        res["timing"][str(k)] = r
        print(f"[timing] k={k:+d}  coef={r['logpat']['coef']:+.4f} "
              f"t={r['logpat']['t']:+.2f}  n={r['n_obs']:,} firms={r['n_firms']:,}",
              file=sys.stderr)

    lead = frames[LEAD]
    for g in ("complement", "neutral", "substitute"):
        res["groups"][g] = guarded(lead[lead.nice_group == g])
    res["interaction"] = guarded(lead, ("logpat", "logpat_x_comp"))

    for cls in sorted(NICE_LABELS):
        sub = lead[lead.nice_class == cls]
        r = guarded(sub)
        r["group"] = nice_group(cls)
        r["label"] = NICE_LABELS[cls]
        res["by_class"][cls] = r

    (RES / "patent_timing_by_industry.json").write_text(json.dumps(res, indent=1))
    return res


def latex(res: dict) -> str:
    L = []
    A = L.append
    A(r"\documentclass[11pt]{article}")
    A(r"\usepackage[margin=1in]{geometry}\usepackage{booktabs}\usepackage{amsmath}")
    A(r"\usepackage[hidelinks]{hyperref}\usepackage{longtable}")
    A(r"\newcommand{\dkl}{\Delta\mathrm{KL}}")
    A(r"\title{\vspace{-2.5em}\bfseries Patent timing and industry ordering:"
      r"\\ does patenting lead trademark vocabulary?}")
    A(r"\author{Eric Silver}\date{\today}\begin{document}\maketitle\vspace{-1.5em}")

    A(r"\noindent A patent is applied for before an invention is in public use; a "
      r"trademark is filed when the firm is ready to sell. If $\dkl$ and patenting "
      r"tracked one underlying event, patents should therefore \emph{lead} "
      r"trademark language. Table~\ref{tab:timing} enters patents at four "
      r"timings in the paper's within-firm specification. Table~\ref{tab:classes} "
      r"orders the NICE classes by the lead coefficient.")

    A(r"\begin{table}[h]\centering\small")
    A(r"\caption{Within-firm coefficient of $\dkl$ on $\log(1+\text{patents})$ at "
      r"four patent timings. Firm and year fixed effects, standard errors "
      r"clustered on owner, owners with at least three years. $\dkl$ is "
      r"standardized once on the contemporaneous panel, so all coefficients are "
      r"in the same units as the published $-0.010$. Topic scoring, $T=200$.}")
    A(r"\label{tab:timing}")
    A(r"\begin{tabular}{lrrrrr}\toprule")
    A(r"Patent timing & Coef. ($\sigma$) & SE & $t$ & Owner-years & Owners \\\midrule")
    for k in LAGS:
        r = res["timing"][str(k)]
        lp = r["logpat"]
        A(f"{LAG_LABEL[k]} & ${lp['coef']:+.4f}$\\textsuperscript{{{stars(lp['p'])}}} & "
          f"{lp['se']:.4f} & ${lp['t']:+.2f}$ & {r['n_obs']:,} & {r['n_firms']:,} \\\\")
    A(r"\bottomrule\end{tabular}\end{table}")

    A(r"\vspace{-0.8em}")
    A(r"\begin{table}[h]\centering\small")
    A(r"\caption{The same lead specification ($t-1$) inside the a-priori IPR "
      r"groups, and the pooled interaction. The firm fixed effect absorbs the "
      r"level of the firm-constant complement dummy, so the interaction is "
      r"identified off the slope difference alone.}")
    A(r"\label{tab:groups}")
    A(r"\begin{tabular}{lrrrrr}\toprule")
    A(r"Stratum & Coef. ($\sigma$) & SE & $t$ & Owner-years & Owners \\\midrule")
    for g, lab in (("complement", "Complement sectors"),
                   ("neutral", "Neutral sectors"),
                   ("substitute", "Substitute sectors")):
        r = res["groups"][g]
        if "logpat" not in r:
            A(f"{lab} & \\multicolumn{{3}}{{c}}{{cell too small}} & "
              f"{r['n_obs']:,} & {r['n_firms']:,} \\\\")
            continue
        lp = r["logpat"]
        A(f"{lab} & ${lp['coef']:+.4f}$\\textsuperscript{{{stars(lp['p'])}}} & "
          f"{lp['se']:.4f} & ${lp['t']:+.2f}$ & {r['n_obs']:,} & {r['n_firms']:,} \\\\")
    it = res["interaction"]
    if "logpat_x_comp" in it:
        lp = it["logpat_x_comp"]
        A(r"\midrule")
        A(f"$\\log(1+\\text{{pat}})\\times$complement & ${lp['coef']:+.4f}$"
          f"\\textsuperscript{{{stars(lp['p'])}}} & {lp['se']:.4f} & "
          f"${lp['t']:+.2f}$ & {it['n_obs']:,} & {it['n_firms']:,} \\\\")
    A(r"\bottomrule\end{tabular}\end{table}")

    rows = [(c, r) for c, r in res["by_class"].items() if "logpat" in r]
    rows.sort(key=lambda cr: -cr[1]["logpat"]["coef"])
    A(r"\begin{longtable}{llrrrr}")
    A(r"\caption{NICE classes ordered by the lead ($t-1$) coefficient. "
      r"\textsc{c}~=~a-priori complement, \textsc{s}~=~substitute, "
      r"\textsc{n}~=~neutral. Classes with fewer than "
      f"{MIN_FIRMS} owners are omitted."
      r"}\label{tab:classes}\\\toprule")
    A(r"Class & Group & Coef. ($\sigma$) & $t$ & Owner-years & Owners \\\midrule")
    A(r"\endfirsthead\toprule Class & Group & Coef. ($\sigma$) & $t$ & "
      r"Owner-years & Owners \\\midrule\endhead")
    for cls, r in rows:
        lp = r["logpat"]
        tag = {"complement": r"\textsc{c}", "substitute": r"\textsc{s}",
               "neutral": r"\textsc{n}"}[r["group"]]
        A(f"{cls} {r['label']} & {tag} & ${lp['coef']:+.4f}$"
          f"\\textsuperscript{{{stars(lp['p'])}}} & ${lp['t']:+.2f}$ & "
          f"{r['n_obs']:,} & {r['n_firms']:,} \\\\")
    A(r"\bottomrule\end{longtable}")

    A(r"\vspace{-1em}\noindent\footnotesize\textsuperscript{***}$p<0.01$, "
      r"\textsuperscript{**}$p<0.05$, \textsuperscript{*}$p<0.10$. "
      r"Every owner in the panel is matched to a PatentsView assignee and so "
      r"patents at least once; the substitute cells are patenting firms in "
      r"low-patenting sectors rather than typical firms in those sectors, which "
      r"makes the two groups more alike and biases the interaction toward zero.")

    A(r"\vspace{1.2em}\noindent\rule{\linewidth}{0.4pt}\\[0.4em]")
    A(r"\noindent\footnotesize\textbf{Provenance.} This note was produced with "
      r"AI assistance. The author specified the research question, the timing "
      r"logic, the estimating specification, and the a-priori sector grouping, "
      r"and is responsible for its contents. Claude (Anthropic), model "
      r"Claude Fable 5, wrote \texttt{scripts/patent\_timing\_by\_industry.py}, "
      r"ran the estimations, and drafted this note. Nothing here rests on that "
      r"tooling: the script, the panel build, and the underlying corpus are "
      r"public at \url{https://github.com/ericsilver/novelty}, so every "
      r"coefficient in these tables can be reproduced independently.")
    A(r"\end{document}")
    return "\n".join(L)


def main() -> int:
    res = build("t200")
    tex = RES / "patent_timing_note.tex"
    tex.write_text(latex(res), encoding="utf-8")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode",
                        "-output-directory", str(RES), str(tex)],
                       capture_output=True)
    print(f"[done] -> {RES / 'patent_timing_note.pdf'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
