"""B1 baseline: share of filings each year that are owner-debuts.

A debut filing is each owner's first-ever filing across all 45 NICE classes
(matching debut_harvest_fit.py). We compute:

  - Pooled per-year: total filings, debut filings, share = debut/total
  - Per (NICE class x year): same triple, plus the per-class share trend

This gives the flow-level entry compression number. If the share is
falling broadly, that is consistent with declining business dynamism.
If it falls only in some classes, the story is sectoral.

Outputs:
  paper/results/dynamism_firsttimer_share_pooled.csv
  paper/results/dynamism_firsttimer_share_pooled.png
  paper/results/dynamism_firsttimer_share_byclass.csv
  paper/results/dynamism_firsttimer_share_byclass.png
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from novelty.industries import name as industry_name

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

YEAR_MIN = 1985
YEAR_MAX = 2025


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # Pass 1: owner -> earliest filing date across all classes
    print("[pass 1] owner debut dates …", flush=True)
    parts = []
    for cls in class_list():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                              columns=["owner_name","filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8))
        d = tm.group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        parts.append(d)
        del tm, d
        gc.collect()
    debut = (pl.concat(parts).group_by("owner_name")
                .agg(pl.col("debut_date").min().alias("debut_date"))
                .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int64).alias("debut_year")))
    print(f"  {debut.height:,} unique owners", flush=True)

    # Pass 2: per (class, year), total filings + debut filings
    print("[pass 2] per-class panel …", flush=True)
    pool_rows = []
    byclass_rows = []
    for cls in class_list():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                              columns=["owner_name","filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(
            pl.col("filing_date").str.slice(0,4).cast(pl.Int64).alias("year"))
        tm = tm.filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        # flag: is this filing the owner's debut (i.e., does its date match the owner's earliest date)?
        tm = tm.join(debut.select(["owner_name","debut_date"]), on="owner_name", how="left").with_columns(
            (pl.col("filing_date") == pl.col("debut_date")).alias("is_debut"))
        # Per class x year
        g = tm.group_by("year").agg([
            pl.len().alias("n_filings"),
            pl.col("is_debut").sum().alias("n_debuts"),
        ]).with_columns(
            (pl.col("n_debuts").cast(pl.Float64) / pl.col("n_filings").cast(pl.Float64)).alias("share"),
            pl.lit(cls).alias("cls"),
            pl.lit(industry_name(cls)).alias("class_name"),
        ).sort("year")
        byclass_rows.append(g)
        del tm, g
        gc.collect()

    byclass = pl.concat(byclass_rows)
    byclass.write_csv(OUT / "dynamism_firsttimer_share_byclass.csv")

    # Pooled
    pooled = (byclass.group_by("year").agg([
        pl.col("n_filings").sum().alias("n_filings"),
        pl.col("n_debuts").sum().alias("n_debuts"),
    ]).with_columns(
        (pl.col("n_debuts").cast(pl.Float64) / pl.col("n_filings").cast(pl.Float64)).alias("share")
    ).sort("year"))
    pooled.write_csv(OUT / "dynamism_firsttimer_share_pooled.csv")
    print("\nPooled pooled debut share by year:")
    print(pooled.to_pandas().to_string(index=False))

    # Pooled plot: dual axis, total + debut count on left, share on right
    fig, ax1 = plt.subplots(figsize=(12, 6))
    pp = pooled.to_pandas()
    ax1.plot(pp["year"], pp["n_filings"], "-o", color="#888888", lw=1.5, ms=3,
             label="Total filings")
    ax1.plot(pp["year"], pp["n_debuts"], "-o", color="#2b6cb0", lw=2, ms=4,
             label="Debut filings (owner's first ever)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Filings")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(pp["year"], pp["share"], "--s", color="#cc4444", lw=2, ms=4,
             label="Debut share")
    ax2.set_ylabel("Debut share (debuts / total)", color="#cc4444")
    ax2.tick_params(axis="y", labelcolor="#cc4444")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(0, max(0.8, pp["share"].max() * 1.1))

    fig.suptitle(
        f"USPTO trademark filings: total vs first-time-filer (debut) count, {YEAR_MIN}-{YEAR_MAX}\n"
        f"Debut share = debut count / total. Pooled across 45 NICE classes.",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "dynamism_firsttimer_share_pooled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-class small-multiples
    clses = sorted(byclass["cls"].unique().to_list())
    n = len(clses)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*1.9), sharex=True)
    pooled_share = pp.set_index("year")["share"]
    for i, cls in enumerate(clses):
        ax = axes[i // cols, i % cols]
        sub = byclass.filter(pl.col("cls") == cls).sort("year").to_pandas()
        ax.plot(sub["year"], sub["share"], "-", color="#2b6cb0", lw=1.2)
        ax.plot(pooled_share.index, pooled_share.values, "-", color="#cccccc", lw=0.8, alpha=0.8)
        cname = industry_name(cls)
        ax.set_title(f"{cls} {cname[:24]}", fontsize=7)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(alpha=0.2)
    for j in range(n, rows*cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(
        f"Debut share by NICE class, {YEAR_MIN}-{YEAR_MAX}.  "
        f"Blue = class-specific; grey = pooled.",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "dynamism_firsttimer_share_byclass.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] wrote pooled + by-class outputs to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
