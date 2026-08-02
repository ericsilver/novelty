"""Is the present as 'interesting' as the dot-com era?

Prospective KL needs only past references, so per-year mean prospective
surprise runs to the corpus edge. Two measures:

1. Per-year mean prospective KL (clean filter), per class for software
   (009), tech services (042), advertising/retail (035), and
   telecommunications (038), plus all-45 pooled, 1995-2024.
2. Era phrase prevalence in classes 009+042: internet-era terms vs
   AI-era terms, aligned years.

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

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

FOCUS = {"009": "Software & Electronics", "042": "Sci & Tech Services",
         "035": "Advertising & Retail", "038": "Telecommunications"}

INTERNET_TERMS = ["internet", "website", "web site", "online", "world wide web"]
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

    ax = axes[1]
    yrs_n = sorted(y for y in prev_net if 1990 <= y <= 2010)
    yrs_a = sorted(y for y in prev_ai if 2010 <= y <= 2024)
    ax.plot(yrs_n, [100 * prev_net[y] for y in yrs_n], "o-", ms=3,
            label="Internet-era terms (bottom axis)", color="#cc4444")
    ax2 = ax.twiny()
    ax2.plot(yrs_a, [100 * prev_ai[y] for y in yrs_a], "s-", ms=3,
             label="AI-era terms (top axis)", color="#2b6cb0")
    ax.set_xlabel("Internet era year", color="#cc4444")
    ax2.set_xlabel("AI era year", color="#2b6cb0")
    ax.set_ylabel("% of class 009+042 filings containing era terms")
    ax.set_title("Era vocabulary prevalence: internet vs AI")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "era_turbulence.png", dpi=140, bbox_inches="tight")
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
