"""Black trademark-rise vs patent-fall comparison.

US-only data on both sides:
* Trademark: Black surname-imputed share of US-domiciled debut filers,
  rising from 11.4% in 1990s to 15.7% in 2020-24.
* Patent: Black surname-imputed share of US-resident inventor debuts,
  falling from ~6.6% in 1990s to ~4.5% in 2020-24.

Test whether the opposite-direction trend persists once we restrict
the patent side to US-resident inventors only (consistent with the
US-only trademark sample).

If both are US-only and trademark-Black rises while patent-Black
falls, the divergence is a real phenomenon in formal innovation
activity by Black Americans:
  - Trademark formation rising (small-business / brand activity)
  - Patent invention falling (relative share)

Output:
  paper/results/black_tm_vs_patent_divergence.csv
  paper/results/black_tm_vs_patent_divergence.png
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PV = REPO / "data" / "raw" / "patentsview"
REF = REPO / "reference_data"
OUT = REPO / "paper" / "results"


def normalize_surname(s: str | None) -> str | None:
    if s is None or s == "":
        return None
    nrm = unicodedata.normalize("NFD", s)
    nrm = "".join(c for c in nrm if unicodedata.category(c) != "Mn")
    nrm = nrm.upper().strip()
    nrm = re.sub(r"[^A-Z'\- ]", "", nrm)
    parts = nrm.split()
    if not parts:
        return None
    return parts[-1] if len(parts[-1]) >= 2 else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ----- Trademark side: read existing US-only panel
    print("[1/3] trademark US-only panel …", flush=True)
    tm = pl.read_csv(OUT / "surname_ethnicity_us_panel.csv").filter(
        pl.col("year").is_between(1990, 2024)
    ).to_pandas()
    # Compute per-year shares (these are already in the panel as share_*)

    # ----- Patent side: US-resident inventor debuts, by year
    print("[2/3] patent US-resident inventor debuts → Census ethnicity by year …",
          flush=True)
    loc = pl.read_csv(PV / "g_location_disambiguated.tsv", separator="\t",
                       null_values=[""]).select("location_id", "disambig_country")
    inv = pl.read_csv(
        PV / "g_inventor_disambiguated.tsv", separator="\t", null_values=[""],
    ).select("patent_id", "inventor_id", "disambig_inventor_name_last", "location_id")
    inv = inv.join(loc, on="location_id", how="left").rename({"disambig_country": "country"})
    pat = pl.read_csv(
        PV / "g_patent.tsv", separator="\t", null_values=[""],
        schema_overrides={"patent_id": pl.Utf8},
    ).select("patent_id", "patent_date")
    inv = inv.join(pat, on="patent_id", how="left").with_columns(
        pl.col("patent_date").str.slice(0, 4).cast(pl.Int32).alias("year")
    ).filter(pl.col("country") == "US")

    # First-patent year per inventor
    inv_debut = (
        inv.filter(pl.col("year").is_not_null() & pl.col("disambig_inventor_name_last").is_not_null())
           .group_by("inventor_id")
           .agg(pl.col("year").min().alias("debut_year"),
                pl.col("disambig_inventor_name_last").first().alias("last_name"))
    )
    surnames = [normalize_surname(s) for s in inv_debut["last_name"].to_list()]
    inv_debut = inv_debut.with_columns(pl.Series("surname", surnames)).filter(pl.col("surname").is_not_null())
    print(f"      {inv_debut.height:,} unique US-resident inventors")

    surn = pl.read_csv(REF / "census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    joined = inv_debut.join(
        surn.select(["name_u"] + pct_cols), left_on="surname", right_on="name_u", how="inner",
    ).with_columns([(pl.col(c).fill_null(0.0)/100.0).alias(f"w_{c[3:]}") for c in pct_cols])
    print(f"      {joined.height:,} matched Census surname")

    weight_cols = [f"w_{c[3:]}" for c in pct_cols]
    pv = joined.filter(pl.col("debut_year").is_between(1990, 2024)).group_by("debut_year").agg(
        [pl.len().alias("n_matched"),
         *[pl.col(w).sum().alias(w) for w in weight_cols]]
    ).sort("debut_year").with_columns(
        [(pl.col(w)/pl.col("n_matched")).alias(f"share_{w[2:]}") for w in weight_cols]
    ).to_pandas()

    # ----- Build merged comparison panel
    print("[3/3] merging + plotting …", flush=True)
    merged = pd_merged = tm[["year","share_white","share_black","share_api","share_hispanic","n_matched"]].rename(
        columns={"n_matched":"n_tm",
                 "share_white":"tm_white","share_black":"tm_black",
                 "share_api":"tm_api","share_hispanic":"tm_hispanic"}
    ).merge(
        pv[["debut_year","share_white","share_black","share_api","share_hispanic","n_matched"]].rename(
            columns={"debut_year":"year","n_matched":"n_pv",
                     "share_white":"pv_white","share_black":"pv_black",
                     "share_api":"pv_api","share_hispanic":"pv_hispanic"}
        ), on="year", how="inner"
    )
    merged.to_csv(OUT / "black_tm_vs_patent_divergence.csv", index=False)

    # Plot: 4-panel (one per ethnic group), TM vs Patent side by side
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    eths = [("Non-Hispanic White", "white", axes[0,0]),
            ("Non-Hispanic Black", "black", axes[0,1]),
            ("Asian / Pacific Islander", "api", axes[1,0]),
            ("Hispanic (any race)", "hispanic", axes[1,1])]
    for label, key, ax in eths:
        ax.plot(merged["year"], 100*merged[f"tm_{key}"], "-o", color="#2b6cb0", ms=4, lw=2,
                label="Trademark debuts (US-only)")
        ax.plot(merged["year"], 100*merged[f"pv_{key}"], "-s", color="#cc4444", ms=4, lw=2,
                label="Patent inventor debuts (US-only)")
        ax.set_title(label)
        ax.set_ylabel("Share of US-domiciled individuals (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    axes[1,0].set_xlabel("Debut year")
    axes[1,1].set_xlabel("Debut year")
    fig.suptitle("US-only: surname-imputed ethnic share of debut filers, trademark vs patent",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "black_tm_vs_patent_divergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- Summary table
    print()
    print("Decade-by-decade comparison (US-only on both sides):")
    print(f"{'Period':<10}  {'TM White':>10}  {'PV White':>10}  {'TM Black':>10}  {'PV Black':>10}  "
          f"{'TM API':>10}  {'PV API':>10}  {'TM Hisp':>10}  {'PV Hisp':>10}")
    for lo, hi in [(1990,1999),(2000,2009),(2010,2019),(2020,2024)]:
        sub = merged[(merged["year"]>=lo)&(merged["year"]<=hi)]
        if not len(sub): continue
        tm_n = sub["n_tm"].sum(); pv_n = sub["n_pv"].sum()
        row = [f"{lo}-{hi}"]
        for key in ["white","black","api","hispanic"]:
            tm_w = (sub[f"tm_{key}"]*sub["n_tm"]).sum() / tm_n
            pv_w = (sub[f"pv_{key}"]*sub["n_pv"]).sum() / pv_n
            row.append(f"{100*tm_w:>9.2f}%")
            row.append(f"{100*pv_w:>9.2f}%")
        print("  ".join(row))

    # ----- Specifically check Black divergence over 1990-2024
    print()
    print("Black share trajectory comparison (US-only):")
    for yr in [1990, 2000, 2010, 2015, 2020, 2024]:
        sub = merged[merged["year"]==yr]
        if not len(sub): continue
        r = sub.iloc[0]
        print(f"  {yr}: TM Black = {100*r['tm_black']:.2f}%   PV Black = {100*r['pv_black']:.2f}%  diff = {100*(r['tm_black']-r['pv_black']):+.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
