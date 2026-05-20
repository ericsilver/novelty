"""Per-industry deep-dive appendix generator.

For each NICE class with enough data, emit one appendix section containing:
  1. Summary stats (n, owners, KL means, survival, SEC share).
  2. Trending tokens (top tokens enriched at positive-ΔKL extreme) with their
     survival rates and corpus frequency.
  3. Ending tokens (top tokens enriched at negative-ΔKL extreme) with survival.
  4. Flux-neutral most-frequent tokens at |ΔKL|≈0 with survival.
  5. Scatter plot in (KL_pros, KL_retr) space showing all clean filings,
     ΔKL=0 diagonal, and the top-10 most-consequential SEC-listed firms
     that DEBUTED in 1990-2021 labeled by name.
  6. Table of those top-10 firms' first filings (mark, year, g/s excerpt).

Outputs:
  paper/results/per_industry_deep/class<NNN>_scatter.png  (one per class)
  paper/results/per_industry_deep_appendix.tex             (concatenated)
"""

from __future__ import annotations

import gc
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from novelty.industries import name as industry_name

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"
PNG_DIR = OUT / "per_industry_deep"


# Curated stopword set: bigram-joiner words and templated boilerplate
STOPWORDS = frozenset({
    "the","a","an","of","in","for","and","or","by","to","on","with","at","from",
    "namely","including","etc","other","via","as","not","is","are","be","such",
    "wherein","also","whereby","both","its","than","then","that","this","these",
    "those","into","over","under","between","through","upon","without","among",
    "all","any","each","more","most","some","one","two","three","four","five",
    "out","up","down","new","being","been","being","does","do","did","has","have",
    "providing","using","based","relating","made","used","include","includes",
    "included","comprised","composed","consisting","containing","goods","services",
})
WORD_RE = re.compile(r"[a-z][a-z\-]{2,}")


def texescape(s: str | None) -> str:
    if s is None: return ""
    s = s.replace("\\", r"\textbackslash{}")
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return s


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def tokens_of(doc: str) -> set[str]:
    if not doc: return set()
    return {t for t in WORD_RE.findall(doc.lower()) if t not in STOPWORDS}


def render_one(cls: str, sec_owners: set[str], cik_max_rev: dict, owner_to_cik: dict) -> tuple[str | None, str | None]:
    sp = PROC / f"surprise_class{cls}.parquet"
    op = PROC / f"outcomes_class{cls}.parquet"
    tm = PROC / f"tm_class{cls}.parquet"
    if not (sp.exists() and op.exists() and tm.exists()):
        return None, None

    s = pl.read_parquet(sp,
        columns=["serial_number","owner_name","year","prospective_kl","retrospective_kl",
                 "n_ref_prospective","n_ref_retrospective","n_terms"]).filter(
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
        & pl.col("year").is_between(1990, 2021)
    ).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
    )
    if s.height < 5_000:
        return None, None
    o = pl.read_parquet(op, columns=["serial_number","survived_5y","reached_registration"])
    t = pl.read_parquet(tm, columns=["serial_number","mark_identification","goods_services"])
    j = s.join(o, on="serial_number", how="inner").join(t, on="serial_number", how="inner")
    n = j.height

    # ---- 1. Summary stats ----
    mean_pros = float(j["prospective_kl"].mean())
    mean_retr = float(j["retrospective_kl"].mean())
    mean_dkl  = float(j["dkl"].mean())
    survival = float(j["survived_5y"].cast(pl.Float64).mean())
    sec_share = float(j["owner_name"].is_in(sec_owners).cast(pl.Float64).mean())

    # ---- 2-4. Tokens at positive / negative / neutral ΔKL strata ----
    dkl_arr = j["dkl"].to_numpy()
    surv_arr = j["survived_5y"].to_numpy().astype(bool)
    gs_arr = j["goods_services"].to_list()

    p10 = np.quantile(dkl_arr, 0.10)
    p90 = np.quantile(dkl_arr, 0.90)

    counts = {"pos": Counter(), "neg": Counter(), "neu": Counter(), "all": Counter()}
    surv_sum = {"pos": Counter(), "neg": Counter(), "neu": Counter(), "all": Counter()}

    for i, doc in enumerate(gs_arr):
        toks = tokens_of(doc)
        is_survived = surv_arr[i]
        d = dkl_arr[i]
        for stratum in ("all", ):
            for tk in toks:
                counts[stratum][tk] += 1
                if is_survived: surv_sum[stratum][tk] += 1
        if d >= p90:
            for tk in toks:
                counts["pos"][tk] += 1
                if is_survived: surv_sum["pos"][tk] += 1
        elif d <= p10:
            for tk in toks:
                counts["neg"][tk] += 1
                if is_survived: surv_sum["neg"][tk] += 1
        elif abs(d) <= 0.02:
            for tk in toks:
                counts["neu"][tk] += 1
                if is_survived: surv_sum["neu"][tk] += 1

    n_pos = int(((dkl_arr >= p90)).sum())
    n_neg = int(((dkl_arr <= p10)).sum())
    n_neu = int((np.abs(dkl_arr) <= 0.02).sum())
    n_all = len(dkl_arr)

    def top_enriched(stratum: str, n_stratum: int, top_n: int = 10, min_count: int = max(30, n//1000)) -> list[dict]:
        if n_stratum == 0: return []
        rows = []
        for tk, c in counts[stratum].items():
            tot = counts["all"][tk]
            if tot < min_count: continue
            if tot < 50: continue
            p_stratum = c / n_stratum
            p_all     = tot / n_all
            if p_all <= 0: continue
            lift = p_stratum / p_all
            surv_rate = surv_sum["all"][tk] / tot if tot > 0 else float("nan")
            rows.append({"token": tk, "lift": lift, "n_with_token": tot,
                         "intra_class_surv": surv_rate})
        rows.sort(key=lambda r: -r["lift"])
        return rows[:top_n]

    trending = top_enriched("pos", n_pos, top_n=10)
    ending   = top_enriched("neg", n_neg, top_n=10)
    # For flux-neutral, sort by frequency in the neutral stratum
    neu_rows = []
    if n_neu > 0:
        for tk, c in counts["neu"].most_common(50):
            tot = counts["all"][tk]
            if tot < 50: continue
            surv = surv_sum["all"][tk] / tot
            neu_rows.append({"token": tk, "n_in_neutral": c, "n_with_token": tot,
                             "intra_class_surv": surv})
        neu_rows = sorted(neu_rows, key=lambda r: -r["n_in_neutral"])[:10]

    # ---- 5. Public firms that DEBUTED in 1990-2021 in this class ----
    firm_first = (j.filter(pl.col("owner_name").is_not_null())
                   .sort(["owner_name", "year"])
                   .group_by("owner_name", maintain_order=True)
                   .agg([pl.col("year").min().alias("first_year"),
                         pl.col("serial_number").first().alias("first_serial"),
                         pl.col("prospective_kl").first().alias("first_pros"),
                         pl.col("retrospective_kl").first().alias("first_retr"),
                         pl.col("dkl").first().alias("first_dkl"),
                         pl.col("mark_identification").first().alias("first_mark"),
                         pl.col("goods_services").first().alias("first_gs"),
                         pl.len().alias("n_filings_in_class")])
                   .filter(pl.col("first_year").is_between(1990, 2021))
                   .filter(pl.col("owner_name").is_in(list(sec_owners))))
    # Annotate with max_revenue for ranking
    firm_first = firm_first.with_columns(
        pl.col("owner_name").map_elements(
            lambda o: float(cik_max_rev.get(owner_to_cik.get(o, -1), 0.0) or 0.0),
            return_dtype=pl.Float64
        ).alias("max_revenue")
    )
    top_public = firm_first.sort("max_revenue", descending=True).head(10)

    # ---- 6. Scatter plot ----
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    sample_n = min(5000, j.height)
    if j.height > sample_n:
        idx = np.random.default_rng(42).choice(j.height, sample_n, replace=False)
    else:
        idx = np.arange(j.height)
    px = j["prospective_kl"].to_numpy()[idx]
    py = j["retrospective_kl"].to_numpy()[idx]
    ps = surv_arr[idx]
    # Survived vs failed background
    ax.scatter(px[~ps], py[~ps], s=2, alpha=0.06, color="#cc4444", label="background failed")
    ax.scatter(px[ps],  py[ps],  s=2, alpha=0.10, color="#229922", label="background survived")
    # diagonal
    lo = min(px.min(), py.min())
    hi = max(px.max(), py.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="#444", lw=1, label=r"$\Delta KL = 0$")
    # Labeled top public firms
    for r in top_public.iter_rows(named=True):
        x = r["first_pros"]; y = r["first_retr"]
        if x is None or y is None: continue
        ax.scatter([x], [y], s=72, marker="*", color="#2244cc",
                   edgecolor="black", linewidths=1, zorder=5)
        # short label
        short = (r["owner_name"] or "")[:30]
        ax.annotate(short, (x, y), fontsize=8, alpha=0.9, color="#1a3380",
                    xytext=(6, 4), textcoords="offset points", zorder=6)
    ax.set_xlabel(r"$KL_{\mathrm{pros}}$ (nats)")
    ax.set_ylabel(r"$KL_{\mathrm{retr}}$ (nats)")
    ax.set_title(f"Class {cls}: {industry_name(cls)}\n"
                 f"All clean filings (sample={sample_n:,}); blue stars = top-10 public debuts 1990-2021")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    png_path = PNG_DIR / f"class{cls}_scatter.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Build the section text ----
    name = texescape(industry_name(cls))
    lines = [
        f"\\subsection*{{Class {cls}: {name}}}",
        f"\\label{{app:deep-{cls}}}",
        "",
        # Summary stats
        f"\\begin{{tabular}}{{ll}}",
        f"Clean filings (1990--2021): & {n:,} \\\\",
        f"Mean $KL_\\text{{pros}}$ / $KL_\\text{{retr}}$ / $\\Delta KL$: & "
        f"{mean_pros:.2f} / {mean_retr:.2f} / {mean_dkl:+.3f} nats \\\\",
        f"5y survival rate: & {survival*100:.1f}\\% \\\\",
        f"Share with SEC-listed owner: & {sec_share*100:.2f}\\% \\\\",
        f"\\end{{tabular}}",
        "",
        # Scatter
        f"\\begin{{center}}",
        f"\\includegraphics[width=0.7\\linewidth]{{results/per_industry_deep/class{cls}_scatter.png}}",
        f"\\end{{center}}",
        "",
        # Trending tokens
        f"\\textbf{{Trending tokens (top-decile $\\Delta KL$)}} --- "
        f"with intra-class 5y survival rate.",
        "",
    ]
    if trending:
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"token & lift & $n$ filings & intra-class survival \\")
        lines.append(r"\midrule")
        for r in trending:
            lines.append(f"\\texttt{{{texescape(r['token'])}}} & "
                         f"{r['lift']:.2f}$\\times$ & {r['n_with_token']:,} & "
                         f"{r['intra_class_surv']*100:.0f}\\% \\\\")
        lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    else:
        lines.append("(no enriched tokens above threshold)")
    lines.append("")

    lines.append(f"\\textbf{{Ending tokens (bottom-decile $\\Delta KL$)}} --- "
                 f"intra-class 5y survival rate.")
    lines.append("")
    if ending:
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"token & lift & $n$ filings & intra-class survival \\")
        lines.append(r"\midrule")
        for r in ending:
            lines.append(f"\\texttt{{{texescape(r['token'])}}} & "
                         f"{r['lift']:.2f}$\\times$ & {r['n_with_token']:,} & "
                         f"{r['intra_class_surv']*100:.0f}\\% \\\\")
        lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    else:
        lines.append("(no enriched tokens above threshold)")
    lines.append("")

    lines.append(f"\\textbf{{Flux-neutral most-frequent tokens ($|\\Delta KL|\\leq 0.02$)}}.")
    lines.append("")
    if neu_rows:
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"token & $n$ in stratum & $n$ in class & intra-class survival \\")
        lines.append(r"\midrule")
        for r in neu_rows:
            lines.append(f"\\texttt{{{texescape(r['token'])}}} & {r['n_in_neutral']:,} & "
                         f"{r['n_with_token']:,} & {r['intra_class_surv']*100:.0f}\\% \\\\")
        lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    lines.append("")

    lines.append(f"\\textbf{{Top-10 SEC-listed firms with debut in 1990--2021}} "
                 f"(ranked by max observed revenue, debut filing g/s shown).")
    lines.append("")
    if top_public.height > 0:
        lines.append(r"\begin{tabular}{p{4.0cm}p{2.2cm}rrrp{6.5cm}}")
        lines.append(r"\toprule")
        lines.append(r"Firm & Mark & year & $KL_\text{pros}$ & $\Delta KL$ & First-filing g/s excerpt \\")
        lines.append(r"\midrule")
        for r in top_public.iter_rows(named=True):
            own = texescape((r["owner_name"] or "")[:55])
            mk  = texescape((r["first_mark"] or "")[:30])
            yr  = r["first_year"]
            pkl = r["first_pros"] if r["first_pros"] is not None else float("nan")
            dkl = r["first_dkl"] if r["first_dkl"] is not None else float("nan")
            gs  = texescape((r["first_gs"] or "").replace("\n"," ")[:200])
            lines.append(f"{own} & {mk} & {yr} & {pkl:.2f} & {dkl:+.2f} & {gs} \\\\")
        lines.append(r"\bottomrule"); lines.append(r"\end{tabular}")
    else:
        lines.append("(no SEC-listed firms with debut in this class in window)")
    lines.append(r"\clearpage")
    lines.append("")

    n_pub = top_public.height
    del s, o, t, j
    gc.collect()
    return "\n".join(lines), f"  class {cls}: n={n:,}, sec_debuts={n_pub}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    # crosswalk + financials for revenue rank
    xw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name","cik")
    sec_owners = set(xw["owner_name"].to_list())
    owner_to_cik = dict(zip(xw["owner_name"].to_list(), xw["cik"].to_list()))
    sec_fin = pl.read_parquet(PROC / "sec_firm_year.parquet", columns=["cik","revenue"]).group_by("cik").agg(
        pl.col("revenue").max().alias("max_revenue"))
    cik_max_rev = dict(zip(sec_fin["cik"].to_list(),
                            sec_fin["max_revenue"].to_list()))

    sections = []
    for cls in class_list():
        print(f"[render] class {cls}…", flush=True)
        section, summary = render_one(cls, sec_owners, cik_max_rev, owner_to_cik)
        if section:
            sections.append(section)
            print(summary, flush=True)

    body = "\n".join(sections)
    (OUT / "per_industry_deep_appendix.tex").write_text(body)
    print(f"\n[done] {len(sections)} sections; wrote per_industry_deep_appendix.tex", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
