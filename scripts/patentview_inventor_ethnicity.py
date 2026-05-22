"""Apply Census 2010 surname race/ethnicity probabilities to PatentsView
inventor names. Compare inventor-side ethnicity composition over time
to the trademark-side composition we already computed.

For each disambiguated inventor:
  - Take their FIRST patent (earliest patent_date) as their "debut year"
  - Apply Census surname classifier to their last name
  - Aggregate fractional counts by (year, ethnic bucket)

Data:
  data/raw/patentsview/g_inventor_disambiguated.tsv   (~24M rows)
  data/raw/patentsview/g_patent.tsv                   (patent_date)
  reference_data/census_surnames/Names_2010Census.csv

Output:
  paper/results/patent_inventor_ethnicity_panel.csv
  paper/results/patent_inventor_ethnicity.png
"""

from __future__ import annotations

import re
import gc
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PV = REPO / "data" / "raw" / "patentsview"
REF = REPO / "reference_data"
OUT = REPO / "paper" / "results"


def normalize_surname(s: str | None) -> str | None:
    """Normalize inventor last name to ASCII upper to match Census file."""
    if s is None or s == "":
        return None
    # Unicode-decompose then drop diacritics, upper, strip
    import unicodedata
    nrm = unicodedata.normalize("NFD", s)
    nrm = "".join(c for c in nrm if unicodedata.category(c) != "Mn")
    nrm = nrm.upper().strip()
    nrm = re.sub(r"[^A-Z'\- ]", "", nrm)
    nrm = nrm.strip()
    # Take last token if multi-token (some "first last" got swapped, or
    # compound surnames)
    parts = nrm.split()
    if not parts:
        return None
    return parts[-1] if len(parts[-1]) >= 2 else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading inventor table …", flush=True)
    inv = pl.read_csv(
        PV / "g_inventor_disambiguated.tsv", separator="\t",
        schema_overrides={"inventor_sequence": pl.Int64},
        null_values=[""],
    ).select(
        "patent_id", "inventor_id", "disambig_inventor_name_last", "gender_code"
    )
    print(f"      {inv.height:,} patent-inventor rows")

    print("[2/4] loading patent_date and joining …", flush=True)
    pat = pl.read_csv(
        PV / "g_patent.tsv", separator="\t", null_values=[""],
        schema_overrides={"patent_id": pl.Utf8},
    ).select("patent_id", "patent_date")
    inv = inv.join(pat, on="patent_id", how="left")
    inv = inv.with_columns(
        pl.col("patent_date").str.slice(0, 4).cast(pl.Int32).alias("year")
    ).drop("patent_date")

    print("[3/4] one row per inventor (earliest patent year) …", flush=True)
    inv_debut = (
        inv.filter(pl.col("year").is_not_null() & pl.col("disambig_inventor_name_last").is_not_null())
           .group_by("inventor_id")
           .agg(pl.col("year").min().alias("debut_year"),
                pl.col("disambig_inventor_name_last").first().alias("last_name"),
                pl.col("gender_code").first().alias("gender_code"))
    )
    print(f"      {inv_debut.height:,} unique inventors")

    print("      normalizing surnames …", flush=True)
    surnames = [normalize_surname(s) for s in inv_debut["last_name"].to_list()]
    inv_debut = inv_debut.with_columns(pl.Series("surname", surnames))
    inv_debut = inv_debut.filter(pl.col("surname").is_not_null())
    print(f"      {inv_debut.height:,} after surname normalization")

    print("[4/4] joining Census surname → ethnicity …", flush=True)
    surn = pl.read_csv(REF/"census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    joined = inv_debut.join(
        surn.select(["name_u"] + pct_cols),
        left_on="surname", right_on="name_u", how="inner"
    ).with_columns([(pl.col(c).fill_null(0.0)/100.0).alias(f"w_{c[3:]}") for c in pct_cols])
    print(f"      {joined.height:,} inventors matched Census surname  "
          f"({100*joined.height/inv_debut.height:.1f}%)")

    # Aggregate by year
    weight_cols = [f"w_{c[3:]}" for c in pct_cols]
    by_year = (
        joined.filter(pl.col("debut_year").is_between(1990, 2024))
              .group_by("debut_year")
              .agg([pl.len().alias("n_matched"),
                    *[pl.col(w).sum().alias(w) for w in weight_cols]])
              .sort("debut_year")
              .with_columns([(pl.col(w)/pl.col("n_matched")).alias(f"share_{w[2:]}") for w in weight_cols])
    )
    by_year.write_csv(OUT / "patent_inventor_ethnicity_panel.csv")

    # ---- Plot: side-by-side trademark vs patent ethnicity composition
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    tm_panel = pl.read_csv(OUT / "surname_ethnicity_panel.csv")
    tm = tm_panel.filter(pl.col("year").is_between(1990, 2024)).to_pandas()
    pv = by_year.to_pandas()
    eth_labels = {
        "share_white": ("Non-Hispanic White", "#2b6cb0"),
        "share_api":   ("Asian / Pacific Islander", "#229922"),
        "share_hispanic": ("Hispanic (any race)", "#cc7a00"),
        "share_black":    ("Non-Hispanic Black", "#cc4444"),
        "share_2prace":   ("Two or More Races", "#999999"),
        "share_aian":     ("AIAN", "#7a3eb2"),
    }
    for col, (label, color) in eth_labels.items():
        axes[0].plot(tm["year"], 100*tm[col], "-o", color=color, ms=3, lw=1.5, label=label)
        axes[1].plot(pv["debut_year"], 100*pv[col], "-o", color=color, ms=3, lw=1.5, label=label)
    axes[0].set_title(f"USPTO trademark debut filers\n(individual-name only, n_total={tm_panel['n_matched'].sum():,})")
    axes[1].set_title(f"PatentsView inventor debuts\n(unique inventors, n_total={joined.height:,})")
    for ax in axes:
        ax.set_xlabel("Debut year")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Surname-imputed ethnic share (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.suptitle("Surname-imputed ethnic composition: trademark debut filers vs patent inventor debuts, 1990-2024",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "patent_inventor_ethnicity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    print()
    print("Patent-side composition by decade:")
    for yr_band in [(1990,1999),(2000,2009),(2010,2019),(2020,2024)]:
        lo, hi = yr_band
        sub = pv[(pv["debut_year"] >= lo) & (pv["debut_year"] <= hi)]
        if not len(sub): continue
        n = sub["n_matched"].sum()
        print(f"  {lo}-{hi} (n_inventors={n:>9,})  "
              + "  ".join(f"{l[:5]}={100*(sub[col]*sub['n_matched']).sum()/n:5.1f}%"
                          for col,(l,_) in eth_labels.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
