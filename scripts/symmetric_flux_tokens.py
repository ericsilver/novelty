"""Symmetric companion to vanishing_trend_examples.py.

For each of three flux strata (negative ΔKL = harvesting receding
vocabulary; neutral ΔKL ≈ 0 = aligned with the field; positive ΔKL =
trend-setting), compute:
  - the unigrams most over-represented in that stratum vs corpus baseline
  - for each surfaced token, its survival rate across ALL filings
    containing the token (filing-level, CLEAN H=2y panel)

Output a paper-ready three-column table:
  | NEGATIVE (harvest) | NEUTRAL | POSITIVE (innovate) |

Each cell: 'token (XX%)' where XX% is mean survived_5y across all
filings containing that token.

Outputs:
  paper/results/symmetric_flux_tokens.txt
  paper/results/symmetric_flux_tokens.tex   (paper-ready LaTeX table)
  paper/results/symmetric_flux_tokens.csv
"""

from __future__ import annotations

import gc
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")  # unigrams >=3 chars, alpha + hyphen


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def stream_panel():
    """Yield (year, dkl, survived_5y, gs_text_lower) per filing.
    Streaming to keep memory low."""
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        op = PROC / f"outcomes_class{cls}.parquet"
        tm = PROC / f"tm_class{cls}.parquet"
        if not (sp.exists() and op.exists() and tm.exists()): continue
        s = pl.read_parquet(
            sp,
            columns=["serial_number","year","kl_vs_past","kl_vs_future",
                     "n_ref_past","n_ref_future","n_terms"],
        ).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 5)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
            & pl.col("year").is_between(1985, 2021)
        ).with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        )
        o = pl.read_parquet(op, columns=["serial_number","survived_5y"])
        t = pl.read_parquet(tm, columns=["serial_number","goods_services"])
        j = s.join(o, on="serial_number", how="inner").join(t, on="serial_number", how="inner").select(
            "year","dkl","survived_5y","goods_services"
        )
        for r in j.iter_rows(named=True):
            yield r
        del s, o, t, j
        gc.collect()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[pass 1] streaming for ΔKL distribution …", flush=True)
    dkls = []
    rows = []
    for r in stream_panel():
        dkls.append(r["dkl"])
        rows.append(r)
    import numpy as np
    arr = np.array(dkls)
    p10 = float(np.quantile(arr, 0.10))
    p40 = float(np.quantile(arr, 0.40))
    p60 = float(np.quantile(arr, 0.60))
    p90 = float(np.quantile(arr, 0.90))
    print(f"[stratify] ΔKL p10={p10:.3f}  p40={p40:.3f}  p60={p60:.3f}  p90={p90:.3f}", flush=True)
    print(f"           n_total = {len(rows):,}", flush=True)

    # Token counts per stratum + per-token survival sums
    tok_total = Counter()
    tok_neg = Counter(); tok_neu = Counter(); tok_pos = Counter()
    n_neg = n_neu = n_pos = 0
    tok_surv_sum = defaultdict(float); tok_surv_n = defaultdict(int)

    print("[pass 2] tokenizing + counting …", flush=True)
    for i, r in enumerate(rows):
        if i and i % 500_000 == 0:
            print(f"   {i:,}/{len(rows):,}", flush=True)
        gs = (r["goods_services"] or "").lower()
        toks = set(TOKEN_RE.findall(gs))
        surv = 1.0 if r["survived_5y"] else 0.0
        for t in toks:
            tok_total[t] += 1
            tok_surv_sum[t] += surv
            tok_surv_n[t]   += 1
        dkl = r["dkl"]
        if dkl <= p10:
            n_neg += 1
            for t in toks: tok_neg[t] += 1
        elif p40 <= dkl <= p60:
            n_neu += 1
            for t in toks: tok_neu[t] += 1
        elif dkl >= p90:
            n_pos += 1
            for t in toks: tok_pos[t] += 1

    print(f"[strata] n_neg={n_neg:,}  n_neu={n_neu:,}  n_pos={n_pos:,}", flush=True)
    n_total = len(rows)

    def enrich(stratum_counts: Counter, n_stratum: int, exclude_others: Counter = None,
               min_pop: int = 2000):
        # rank tokens by lift over corpus, gated on popularity
        out = []
        for t, c in stratum_counts.items():
            tot = tok_total[t]
            if tot < min_pop: continue
            p_stratum = c / n_stratum if n_stratum else 0
            p_pop     = tot / n_total
            if p_pop == 0: continue
            lift = p_stratum / p_pop
            # filter pure boilerplate ("services", "the", etc.) by requiring distinctiveness
            if p_pop > 0.20: continue
            out.append((t, lift, p_stratum, p_pop, tot))
        out.sort(key=lambda x: -x[1])
        return out

    enr_neg = enrich(tok_neg, n_neg)
    enr_neu = enrich(tok_neu, n_neu)
    enr_pos = enrich(tok_pos, n_pos)

    def fmt(rows, top=25):
        out_rows = []
        for t, lift, ps, pp, tot in rows[:top]:
            surv = tok_surv_sum[t] / tok_surv_n[t] if tok_surv_n[t] else float("nan")
            out_rows.append({"token": t, "lift": lift, "survival": surv,
                             "n_token": tot, "p_stratum": ps, "p_pop": pp})
        return out_rows

    neg_top = fmt(enr_neg, 25)
    neu_top = fmt(enr_neu, 25)
    pos_top = fmt(enr_pos, 25)

    # Three-column readable text
    lines = ["Symmetric flux tokens with token-level survival rates\n",
             "Each cell: token (survival% across all CLEAN filings containing the token).",
             "Lift = rate in stratum / rate in corpus; min token-pop 2,000.\n"]
    lines.append(f"{'NEGATIVE  ΔKL ≤ {:.2f}  (harvest)'.format(p10):<46s}"
                 f"{'NEUTRAL  |ΔKL| < {:.2f}  (aligned)'.format(p60):<46s}"
                 f"{'POSITIVE  ΔKL ≥ {:.2f}  (innovate)'.format(p90):<46s}")
    lines.append("-" * 138)
    for i in range(max(len(neg_top), len(neu_top), len(pos_top))):
        cells = []
        for col in (neg_top, neu_top, pos_top):
            if i < len(col):
                r = col[i]
                cells.append(f"{r['token']:<22s} {r['survival']*100:5.1f}%  ({r['lift']:.2f}x)")
            else:
                cells.append("")
        lines.append(f"{cells[0]:<46s}{cells[1]:<46s}{cells[2]:<46s}")
    txt = "\n".join(lines)
    (OUT / "symmetric_flux_tokens.txt").write_text(txt)
    print("\n" + txt[:6000])

    # CSV for paper
    csv_rows = []
    for label, col in [("negative", neg_top), ("neutral", neu_top), ("positive", pos_top)]:
        for r in col:
            csv_rows.append({"stratum": label, **r})
    pl.from_dicts(csv_rows).write_csv(OUT / "symmetric_flux_tokens.csv")

    # LaTeX table (top 15 per column for paper)
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Tokens over-represented at the negative, neutral, and positive extremes of $\Delta\mathrm{KL}$ (vocabulary flux). Each cell: token followed by survival rate across all filings containing the token, with enrichment lift in parentheses. Top 15 per stratum, ranked by lift; minimum token popularity 2{,}000 filings.}")
    tex.append(r"\label{tab:symmetric-flux-tokens}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{lll}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Negative $\Delta\mathrm{KL}$ (harvest)} & \textbf{Neutral $\Delta\mathrm{KL}\!\approx\!0$} & \textbf{Positive $\Delta\mathrm{KL}$ (innovate)} \\")
    tex.append(r"\midrule")
    for i in range(15):
        cells = []
        for col in (neg_top[:15], neu_top[:15], pos_top[:15]):
            if i < len(col):
                r = col[i]
                cells.append(f"\\texttt{{{r['token']}}}\\quad {r['survival']*100:.0f}\\% ({r['lift']:.2f}$\\times$)")
            else:
                cells.append("")
        tex.append(" & ".join(cells) + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    (OUT / "symmetric_flux_tokens.tex").write_text("\n".join(tex))
    print(f"\n[done] wrote .txt, .csv, .tex", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
