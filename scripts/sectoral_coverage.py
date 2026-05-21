"""Sectoral decomposition of USPTO debut coverage using NICE-NAICS
crosswalk. For each year and NAICS sector, compute:
  - BFS HBA in that sector
  - USPTO debuts in that sector (via NICE classes mapped to that NAICS)
  - coverage ratio

Coarse crosswalk in reference_data/nice_naics_crosswalk.csv; primary
NAICS sector only (Hall, Helmers, Rogers, & Sena, 2014, _JEL_, document
the imperfection). Many NICE classes have secondary mappings; this
crosswalk takes primary only and flags multi-sector NICE classes in
the notes column.

Outputs:
  paper/results/sectoral_coverage_panel.csv
  paper/results/sectoral_coverage.png
"""

from __future__ import annotations

import gc
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
REF = REPO / "reference_data"
OUT = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class", "") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


# BFS NAICS-sector codes are 2-digit; map our crosswalk's primary_naics_sector
# (e.g. "31-33") to BFS naics_sector code (e.g. "NAICSMNF") used in the
# bfs_monthly.csv file.
NAICS_RANGE_TO_BFS = {
    "11": "NAICS11", "21": "NAICS21", "22": "NAICS22", "23": "NAICS23",
    "31-33": "NAICSMNF", "42": "NAICS42", "44-45": "NAICSRET",
    "48-49": "NAICSTW", "51": "NAICS51", "52": "NAICS52", "53": "NAICS53",
    "54": "NAICS54", "55": "NAICS55", "56": "NAICS56", "61": "NAICS61",
    "62": "NAICS62", "71": "NAICS71", "72": "NAICS72", "81": "NAICS81",
}


def uspto_debuts_by_year_class() -> pl.DataFrame:
    """For each owner, find earliest filing across all classes, with class.
    Then count owner debuts per (year, class) pair."""
    parts = []
    for cls in class_list():
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["owner_name", "filing_date"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(pl.lit(cls).alias("class_id"))
        parts.append(tm)
        del tm
        gc.collect()
    allf = pl.concat(parts)
    debut = (
        allf.sort(["owner_name", "filing_date", "class_id"])
            .group_by("owner_name", maintain_order=False)
            .agg(pl.col("filing_date").first().alias("debut_date"),
                 pl.col("class_id").first().alias("class_id"))
            .with_columns(pl.col("debut_date").str.slice(0, 4).cast(pl.Int32).alias("year"))
    )
    return debut.group_by(["year", "class_id"]).agg(pl.len().alias("uspto_debuts")).sort(["year", "class_id"])


def bfs_sectoral_annual() -> pl.DataFrame:
    """Annual BFS HBA totals by naics_sector at the US national level."""
    m = pl.read_csv(REF / "bfs/bfs_monthly.csv", infer_schema_length=20000)
    months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    sub = m.filter(
        (pl.col("geo") == "US")
        & (pl.col("series") == "BA_HBA")
        & (pl.col("sa") == "U")
    )
    sub = sub.with_columns(
        pl.sum_horizontal([pl.col(mo).cast(pl.Float64) for mo in months]).alias("annual"),
        pl.sum_horizontal([pl.col(mo).is_not_null().cast(pl.Int32) for mo in months]).alias("nobs"),
    ).filter(pl.col("nobs") == 12)
    return sub.select(["year", "naics_sector", "annual"]).rename({"annual": "bfs_hba"})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    xw = pl.read_csv(REF / "nice_naics_crosswalk.csv", truncate_ragged_lines=True).with_columns(
        pl.col("nice_class").cast(pl.Utf8).str.zfill(3).alias("class_id")
    )
    print("[1/3] computing USPTO debuts by (year, class) …", flush=True)
    debuts = uspto_debuts_by_year_class()
    # Pad class_id to 3-char in case of single-digit
    debuts = debuts.with_columns(pl.col("class_id").cast(pl.Utf8).str.zfill(3))

    # Map class_id to BFS naics_sector
    xw2 = xw.with_columns(pl.col("primary_naics_sector").replace(NAICS_RANGE_TO_BFS).alias("bfs_naics"))
    debuts_with_naics = debuts.join(xw2.select(["class_id", "bfs_naics"]), on="class_id", how="left")
    debuts_by_sector = debuts_with_naics.group_by(["year", "bfs_naics"]).agg(
        pl.col("uspto_debuts").sum().alias("uspto_debuts")
    ).rename({"bfs_naics": "naics_sector"})

    print("[2/3] loading BFS sectoral …", flush=True)
    bfs = bfs_sectoral_annual()

    print("[3/3] joining + computing ratios …", flush=True)
    panel = (
        debuts_by_sector.join(bfs, on=["year", "naics_sector"], how="inner")
            .with_columns((pl.col("uspto_debuts") / pl.col("bfs_hba")).alias("ratio"))
            .filter(pl.col("year").is_between(2004, 2024))
            .sort(["year", "naics_sector"])
    )
    panel.write_csv(OUT / "sectoral_coverage_panel.csv")

    # Plot: ratio over time, by sector
    pivot = panel.pivot(values="ratio", index="year", on="naics_sector", aggregate_function="first").sort("year")
    pdf = pivot.to_pandas().set_index("year")
    # Sort sectors by their 2019-2024 average for color stability
    avg = pdf.loc[2019:2024].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.cm.tab20
    for i, sector in enumerate(avg.index):
        ax.plot(pdf.index, 100 * pdf[sector], "-o", ms=3, lw=1.5,
                color=cmap(i % 20), label=sector)
    ax.set_xlabel("Year")
    ax.set_ylabel("USPTO debuts / BFS HBA (%) for sector")
    ax.set_title("Sectoral coverage of USPTO trademark debuts as share of Census BFS\nHigh-Propensity Business Applications, by NAICS sector, 2004-2024")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT / "sectoral_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print("\nSectoral coverage by NAICS sector, 2019-2024 average (USPTO debuts / BFS HBA, %):")
    print((100 * avg).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
