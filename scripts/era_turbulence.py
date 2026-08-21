"""Is the present as 'interesting' as the dot-com era?

Produces the paper's era figure and the two claims it carries: that per-year
corpus turbulence shows no AI-era deviation through 2024, and that AI-era
vocabulary at year nine sits roughly where internet vocabulary stood at year
five.

Both panels deliberately sit OUTSIDE the paper's main measure, for the same
reason: they have to reach 2024, and the two-sided score does not. The lead and
atypicality used everywhere else need a full five-year FORWARD reference, which
does not exist after 2020. Past-facing surprise needs only past references, so
it runs to the corpus edge, and phrase prevalence needs no reference at all.

1. Per-year mean past-facing KL, per class for software (009), tech services
   (042), advertising/retail (035), and telecommunications (038), plus all-45
   pooled, 1995-2024. This is TERM-scored: it reads surprise_class*.parquet,
   the class-year token scoring, because there is no topic-side series that
   runs past 2019. Under the clean filter (>= 1,000 filings in the reference,
   >= 3 in-vocabulary terms in the description). The pooled series is weighted
   by each class's filings in that year, so large classes dominate, as they do
   in the corpus.
2. Era phrase prevalence in classes 009+042: internet-era terms vs AI-era
   terms, on aligned year axes. This is a case-insensitive substring match
   against hand-curated term lists, not a model output -- so the comparison is
   sensitive to the lists and to the era-start alignment chosen, and the
   figure's shaded spans are illustrative rather than estimated. A filing
   counts once if it contains any term in the list.

Reads   data/processed/surprise_class{CLS}.parquet   (all 45, for the pooled)
        data/processed/tm_class{009,042}.parquet
Outputs:
  paper/results/era_turbulence.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

FOCUS = {"009": "Software & Electronics", "042": "Sci & Tech Services",
         "035": "Advertising & Retail", "038": "Telecommunications"}

# Hand-curated era vocabularies. Neither list contains a bare acronym ("ai",
# "ml", "web"): substring matching would fire on ordinary words and on other
# marks' text. "generative" is the one loose entry and is kept because
# "generative ai" and "generative model" both appear, in different phrasings.
INTERNET_TERMS = ["internet", "website", "web site", "online", "world wide web"]
START_SHARE = 0.004   # era year 0 = first year >= this share of 009+042 filings
AI_TERMS = ["artificial intelligence", "machine learning", "generative",
            "ai-powered", "large language", "chatbot", "neural network"]


def yearly_pros(cls: str) -> pl.DataFrame:
    return pl.read_parquet(
        PROC / f"surprise_class{cls}.parquet",
        columns=["year", "kl_vs_past", "n_ref_past", "n_terms"],
    ).filter(
        (pl.col("n_terms") >= 3) & (pl.col("n_ref_past") >= 1000)
        & pl.col("kl_vs_past").is_finite()
        & pl.col("year").is_between(1995, 2024)
    ).group_by("year").agg(
        pl.len().alias("n"), pl.col("kl_vs_past").mean().alias("mean_pros")
    ).sort("year")


def main() -> int:
    out = {}

    # ---- Panel 1: per-year mean prospective KL ----
    series = {}
    for cls, label in FOCUS.items():
        g = yearly_pros(cls)
        series[label] = (g["year"].to_list(), g["mean_pros"].to_list())
        print(f"[pros] {cls} {label}: {g.height} years", file=sys.stderr, flush=True)
    # pooled all classes
    pooled_parts = []
    for p in sorted(PROC.glob("surprise_class???.parquet")):
        cls = p.stem.replace("surprise_class", "")
        g = yearly_pros(cls).with_columns(pl.lit(cls).alias("cls"))
        pooled_parts.append(g)
    # Filing-weighted across classes, not a mean of class means: an unweighted
    # mean would let a small class's noise move the pooled series as much as
    # software's, and the question is whether the CORPUS got more turbulent.
    allg = pl.concat(pooled_parts).group_by("year").agg(
        ((pl.col("mean_pros") * pl.col("n")).sum() / pl.col("n").sum()).alias("mean_pros")
    ).sort("year")
    series["All classes (pooled)"] = (allg["year"].to_list(), allg["mean_pros"].to_list())
    out["mean_pros_by_year"] = {k: dict(zip(v[0], v[1])) for k, v in series.items()}

    # ---- Panel 2: era phrase prevalence in 009+042 ----
    tm_parts = []
    for cls in ("009", "042"):
        tm_parts.append(pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["filing_date", "goods_services"]).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
        ).filter(pl.col("year").is_between(1990, 2024)
                 & (pl.col("goods_services").str.len_chars() > 0)))
    tm = pl.concat(tm_parts)
    tot = tm.group_by("year").len().sort("year")
    totals = dict(zip(tot["year"], tot["len"]))

    def prevalence(terms: list[str]) -> dict[int, float]:
        pat = "|".join(terms)
        hits = tm.filter(pl.col("goods_services").str.to_lowercase()
                         .str.contains(pat)).group_by("year").len().sort("year")
        return {int(y): n / totals[y] for y, n in zip(hits["year"], hits["len"])}

    prev_net = prevalence(INTERNET_TERMS)
    prev_ai = prevalence(AI_TERMS)
    out["internet_prevalence"] = prev_net
    out["ai_prevalence"] = prev_ai
    print("[prev] internet 1995-2002:",
          {y: round(prev_net.get(y, 0), 4) for y in range(1995, 2003)},
          file=sys.stderr, flush=True)
    print("[prev] AI 2015-2024:",
          {y: round(prev_ai.get(y, 0), 4) for y in range(2015, 2025)},
          file=sys.stderr, flush=True)

    (RES / "era_turbulence.json").write_text(json.dumps(out, indent=1, default=float))

    # ---- figure ----
    def year_ticks(axis, step: int) -> None:
        """Calendar-year axis: integer ticks only, no decimal years."""
        axis.xaxis.set_major_locator(MultipleLocator(step))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):d}"))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    for label, (ys, ms) in series.items():
        lw = 2.4 if label.startswith("All") else 1.4
        ax.plot(ys, ms, lw=lw, label=label)
    ax.axvspan(1995, 2001, color="#ffcc66", alpha=0.18)
    ax.axvspan(2020, 2024, color="#66cc99", alpha=0.15)
    ax.set_xlabel("Filing year")
    ax.set_ylabel("Mean prospective KL (novelty vs past)")
    ax.set_title("Corpus turbulence by year\n(dot-com span and AI span shaded)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    year_ticks(ax, 5)

    # Era alignment. Year 0 of each era is the first calendar year in which
    # the era's terms appear in at least START_SHARE of the two classes'
    # filings. A share rather than a count, because the two classes filed
    # 38k marks in 1994 and 80k in 2015, and because the AI list carries a
    # ~0.15% background ("neural network") that would put a count threshold
    # years before the vocabulary moved. Both axes are drawn on the same
    # era-year scale; the AI series stops where the record does.
    start_net = min(y for y in sorted(prev_net) if prev_net[y] >= START_SHARE)
    start_ai = min(y for y in sorted(prev_ai) if prev_ai[y] >= START_SHARE)
    out["era_start_rule"] = {"share": START_SHARE, "internet": start_net, "ai": start_ai}
    print(f"[era] year 0: internet {start_net}, AI {start_ai} (first year >= {START_SHARE:.2%})",
          file=sys.stderr, flush=True)

    ax = axes[1]
    k_lo, k_hi = -4, 16
    yrs_n = [y for y in sorted(prev_net) if k_lo <= y - start_net <= k_hi]
    yrs_a = [y for y in sorted(prev_ai) if k_lo <= y - start_ai <= k_hi and y <= 2024]
    ax.plot([y - start_net for y in yrs_n], [100 * prev_net[y] for y in yrs_n], "o-",
            ms=3, color="#cc4444",
            label=f"Internet-era terms (bottom axis; year 0 = {start_net})")
    ax.plot([y - start_ai for y in yrs_a], [100 * prev_ai[y] for y in yrs_a], "s-",
            ms=3, color="#2b6cb0",
            label=f"AI-era terms (top axis; year 0 = {start_ai})")
    ax.axvline(0, color="#718096", lw=0.8, ls=":")
    ax.set_xlim(k_lo - 0.5, k_hi + 0.5)
    ax.set_xlabel("Internet era, calendar year", color="#cc4444")
    ax.set_ylabel("% of class 009+042 filings containing era terms")
    ax.set_title(f"Era vocabulary prevalence, aligned at first year $\geq$ {100*START_SHARE:.1f}% of filings")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    # Bottom axis: internet calendar years at the same era-year positions.
    ticks_n = [y for y in range(1990, 2011) if y % 5 == 0]
    ax.set_xticks([y - start_net for y in ticks_n])
    ax.set_xticklabels([str(y) for y in ticks_n])
    # Top axis: AI calendar years on the identical scale (same xlim), so one
    # unit of width is one year on both axes and the AI series ends early.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks_a = [y for y in range(2011, 2032) if y % 5 == 0]
    ax2.set_xticks([y - start_ai for y in ticks_a])
    ax2.set_xticklabels([str(y) for y in ticks_a])
    ax2.set_xlabel("AI era, calendar year", color="#2b6cb0")
    fig.tight_layout()
    fig.savefig(RES / "era_turbulence.png", dpi=140, bbox_inches="tight")
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
