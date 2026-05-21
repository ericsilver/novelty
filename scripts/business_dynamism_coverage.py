"""USPTO trademark debuts vs Census BFS business applications, 2004-2024.

For each year:
  - USPTO_DEBUTS:        unique owners whose first-ever filing was in that year
                         (across all NICE classes, on the clean H=2y panel)
  - BFS_BA:              Census Business Applications total (all SS-4 filings)
  - BFS_HBA:             Census High-Propensity Business Applications
                         (subset likely to become employer businesses)

Reports headline ratios and writes:
  paper/results/dynamism_coverage_panel.csv
  paper/results/dynamism_coverage.png
  paper/results/dynamism_coverage.txt
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
# BFS data is a small public reference dataset; we keep a copy in the
# repo under reference_data/ so that the script is reproducible without
# requiring the user to re-download from census.gov.
BFS = REPO / "reference_data" / "bfs"
OUT = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def uspto_debuts_by_year() -> pl.DataFrame:
    """Each owner's earliest filing date across all NICE classes.

    Counts are 'unique owners debuting in year Y' for Y in 1990-2024.
    """
    parts = []
    for cls in class_list():
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["owner_name", "filing_date"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        )
        parts.append(tm.group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date")))
        del tm
        gc.collect()
    debut = (
        pl.concat(parts)
        .group_by("owner_name")
        .agg(pl.col("debut_date").min().alias("debut_date"))
        .with_columns(pl.col("debut_date").str.slice(0, 4).cast(pl.Int32).alias("year"))
    )
    by_year = debut.group_by("year").agg(pl.len().alias("uspto_debuts")).sort("year")
    return by_year


def bfs_national_annual() -> pl.DataFrame:
    """Sum BFS monthly US-total to annual for series BA_BA and BA_HBA."""
    m = pl.read_csv(BFS / "bfs_monthly.csv", infer_schema_length=20000)
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    # National TOTAL only, both BA and HBA, all years
    sub = m.filter(
        (pl.col("geo") == "US")
        & (pl.col("naics_sector") == "TOTAL")
        & (pl.col("series").is_in(["BA_BA", "BA_HBA"]))
        & (pl.col("sa") == "U")  # unadjusted; we sum 12 months so seasonality cancels
    )
    sub = sub.with_columns(
        pl.sum_horizontal([pl.col(mo).cast(pl.Float64) for mo in months]).alias("annual_count"),
        pl.sum_horizontal([pl.col(mo).is_not_null().cast(pl.Int32) for mo in months]).alias(
            "months_observed"
        ),
    ).filter(pl.col("months_observed") == 12)  # require complete years only
    pivot = sub.select(["year", "series", "annual_count"]).pivot(
        values="annual_count", index="year", on="series", aggregate_function="first"
    ).rename({"BA_BA": "bfs_ba", "BA_HBA": "bfs_hba"}).sort("year")
    return pivot


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/3] uspto debut counts by year …", flush=True)
    uspto = uspto_debuts_by_year()
    print(uspto.filter(pl.col("year") >= 2000).to_pandas().to_string(index=False))

    print("\n[2/3] bfs national annual totals …", flush=True)
    bfs = bfs_national_annual()
    print(bfs.to_pandas().to_string(index=False))

    print("\n[3/3] joined panel + headline ratios", flush=True)
    panel = (
        bfs.join(uspto, on="year", how="left")
        .with_columns(
            (pl.col("uspto_debuts") / pl.col("bfs_ba")).alias("ratio_to_ba"),
            (pl.col("uspto_debuts") / pl.col("bfs_hba")).alias("ratio_to_hba"),
        )
        .sort("year")
    )
    panel.write_csv(OUT / "dynamism_coverage_panel.csv")
    pdf = panel.to_pandas()
    print(pdf.to_string(index=False))

    # Headline ratios
    lines = []
    lines.append("USPTO trademark debuts vs Census Business Formation Statistics")
    lines.append("=" * 70)
    lines.append("")
    for yr_band, lo, hi in [("2004-2008", 2004, 2008), ("2014-2018", 2014, 2018), ("2019-2024", 2019, 2024)]:
        sub = pdf[(pdf.year >= lo) & (pdf.year <= hi)].dropna(subset=["bfs_hba", "uspto_debuts"])
        if not len(sub):
            continue
        ba_total = sub.bfs_ba.sum()
        hba_total = sub.bfs_hba.sum()
        u_total = sub.uspto_debuts.sum()
        lines.append(f"{yr_band}:  USPTO debuts {u_total:>12,.0f}")
        lines.append(f"            BFS BA       {ba_total:>12,.0f}    ratio {u_total/ba_total:>6.3f}")
        lines.append(f"            BFS HBA      {hba_total:>12,.0f}    ratio {u_total/hba_total:>6.3f}")
        lines.append("")
    summary = "\n".join(lines)
    (OUT / "dynamism_coverage.txt").write_text(summary)
    print("\n" + summary)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    sub = pdf.dropna(subset=["bfs_hba"])

    ax = axes[0]
    ax.plot(sub.year, sub.bfs_ba / 1e6, "o-", color="#2b6cb0", lw=2, ms=6, label="BFS Business Applications (all SS-4)")
    ax.plot(sub.year, sub.bfs_hba / 1e6, "s-", color="#cc4444", lw=2, ms=6, label="BFS High-Propensity Apps (employer-likely)")
    ax.plot(sub.year, sub.uspto_debuts / 1e6, "^-", color="#229922", lw=2, ms=6, label="USPTO first-ever trademark filer (owner debuts)")
    ax.set_ylabel("Annual count (millions)")
    ax.set_title("Annual business-formation indicators, 2004-2024")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    ax = axes[1]
    ax.plot(sub.year, 100 * sub.ratio_to_hba, "s-", color="#cc4444", lw=2, ms=6,
            label="USPTO debuts / BFS High-Propensity Apps")
    ax.plot(sub.year, 100 * sub.ratio_to_ba, "o-", color="#2b6cb0", lw=2, ms=6,
            label="USPTO debuts / BFS Business Applications")
    ax.set_ylabel("Coverage ratio (%)")
    ax.set_xlabel("Year")
    ax.set_title("USPTO trademark debuts as a share of Census business applications")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.axhline(10, color="#666", ls=":", lw=1)
    ax.axhline(15, color="#666", ls=":", lw=1)
    ax.text(2005, 13, "Dinlersoz, Goldschlag, Myers, Zolas (2018):\n~10-15% of employer businesses ever trademark",
            fontsize=8, color="#666", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "dynamism_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] {OUT/'dynamism_coverage.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
