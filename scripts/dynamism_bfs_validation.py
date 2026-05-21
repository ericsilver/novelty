"""A1 validation: trademark debut count vs Census BFS business applications.

For each owner in the USPTO trademark corpus, record the earliest filing
date across all 45 NICE classes. Aggregate to per-year debut counts. Compare
against Census Business Formation Statistics (BFS) annual business
applications (BA_BA series, unadjusted monthly summed to annual).

BFS coverage: 2004-2025. The trademark debut series covers 1985+. The
overlap window is what we use for the correlation check.

Outputs:
  paper/results/dynamism_bfs_validation.csv
  paper/results/dynamism_bfs_validation.png
  paper/results/dynamism_bfs_validation.txt
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RAW  = REPO / "data" / "raw"
OUT  = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def compute_debut_year_counts() -> pl.DataFrame:
    """Earliest filing year per owner, across all classes; count by year."""
    print("[debuts] scanning per-class …", flush=True)
    parts = []
    for cls in class_list():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                              columns=["owner_name","filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8))
        d = tm.group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        parts.append(d)
        del tm
        gc.collect()
    debut = (pl.concat(parts).group_by("owner_name")
                .agg(pl.col("debut_date").min().alias("debut_date"))
                .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int64).alias("debut_year")))
    print(f"  {debut.height:,} unique owners", flush=True)
    counts = (debut.group_by("debut_year").agg(pl.len().alias("tm_debut_count"))
              .sort("debut_year"))
    return counts


def load_bfs_annual() -> pl.DataFrame:
    """Annual US business applications, summing monthly BA_BA U series."""
    months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    df = pl.read_csv(RAW / "bfs" / "bfs_monthly.csv",
                     null_values=["NA","","S","D","X"], infer_schema_length=10000,
                     schema_overrides={m: pl.Float64 for m in months})
    us = df.filter((pl.col("geo") == "US")
                   & (pl.col("naics_sector") == "TOTAL")
                   & (pl.col("series") == "BA_BA")
                   & (pl.col("sa") == "U"))
    us = us.with_columns(
        pl.sum_horizontal([pl.col(m) for m in months]).alias("annual_total"),
        pl.sum_horizontal([pl.col(m).is_not_null().cast(pl.Int64) for m in months]).alias("n_months"),
    )
    us = us.filter(pl.col("n_months") == 12).select(["year", "annual_total"]).sort("year")
    return us.rename({"annual_total": "bfs_ba_ba_us"})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    tm = compute_debut_year_counts()
    bfs = load_bfs_annual()

    panel = tm.join(bfs, left_on="debut_year", right_on="year", how="outer_coalesce") \
              .sort("debut_year")
    panel.write_csv(OUT / "dynamism_bfs_validation.csv")

    overlap = panel.filter(pl.col("tm_debut_count").is_not_null()
                           & pl.col("bfs_ba_ba_us").is_not_null()).with_columns([
        pl.col("tm_debut_count").cast(pl.Float64),
        pl.col("bfs_ba_ba_us").cast(pl.Float64),
        pl.col("debut_year").cast(pl.Int64),
    ])
    o = overlap.to_pandas(use_pyarrow_extension_array=False)
    if len(o) >= 3:
        r_pearson = o[["tm_debut_count","bfs_ba_ba_us"]].corr(method="pearson").iloc[0,1]
        r_spearman = o[["tm_debut_count","bfs_ba_ba_us"]].corr(method="spearman").iloc[0,1]
        # Year-over-year diff correlation (more conservative; removes the slow trend)
        o_sorted = o.sort_values("debut_year").reset_index(drop=True)
        o_sorted["tm_yoy"] = o_sorted["tm_debut_count"].diff()
        o_sorted["bfs_yoy"] = o_sorted["bfs_ba_ba_us"].diff()
        yoy = o_sorted[["tm_yoy","bfs_yoy"]].dropna()
        r_yoy = yoy.corr(method="pearson").iloc[0,1] if len(yoy) >= 3 else float("nan")
    else:
        r_pearson = r_spearman = r_yoy = float("nan")

    summary = [
        "BFS validation of USPTO trademark debut counts as a business-formation proxy",
        "=" * 76,
        f"USPTO debut years available: {int(tm['debut_year'].min())}-{int(tm['debut_year'].max())}",
        f"BFS years available (complete months): {int(bfs['year'].min())}-{int(bfs['year'].max())}",
        f"Overlap years for correlation: {len(o)} (debut years intersected with complete BFS years)",
        "",
        f"Pearson  r (levels)         : {r_pearson:+.3f}",
        f"Spearman r (levels)         : {r_spearman:+.3f}",
        f"Pearson  r (year-over-year) : {r_yoy:+.3f}",
        "",
        "Interpretation:",
        "  - Level correlation captures the joint trend (both grew over 2004-2025).",
        "  - YoY correlation is a stricter test of co-movement once the trend is removed.",
        "  - YoY r > 0.5 supports the trademark debut count as a higher-resolution",
        "    complement to BFS. YoY r near 0 means the two series only share a trend,",
        "    not turning points.",
    ]
    text = "\n".join(summary)
    (OUT / "dynamism_bfs_validation.txt").write_text(text)
    print("\n" + text)

    # Two-axis plot: tm debuts (left), BFS BA_BA (right)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(tm["debut_year"], tm["tm_debut_count"], "-o", color="#2b6cb0",
             lw=2, ms=4, label="USPTO trademark debut count (per owner, first ever)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("USPTO debut count", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(bfs["year"], bfs["bfs_ba_ba_us"], "-s", color="#cc4444",
             lw=2, ms=4, label="BFS US business applications (annual, BA_BA U)")
    ax2.set_ylabel("BFS BA_BA business applications", color="#cc4444")
    ax2.tick_params(axis="y", labelcolor="#cc4444")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title(
        f"USPTO trademark debut count vs Census BFS business applications\n"
        f"Pearson r (levels) = {r_pearson:+.3f};  Pearson r (YoY) = {r_yoy:+.3f}",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "dynamism_bfs_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {OUT/'dynamism_bfs_validation.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
