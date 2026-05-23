"""US-only re-run of the patent guild HHI signal.

Joins PatentsView inventors to g_location_disambiguated.tsv to
attach inventor country. Restricts to US-resident inventors and
re-computes the surname HHI per (CPC class, decade).

Hypothesis: several of the rising-HHI CPC classes from the
all-inventors run (G09, H10, D06, G11, G02, H01 — semiconductor /
display / optics / hardware) will flatten or fall once
Chinese-resident inventors are stripped, because the rise was
driven by ethnic-group concentration in Chinese-resident filing.

Outputs:
  paper/results/patent_guild_hhi_us_panel.csv
  paper/results/patent_guild_hhi_us.png
  paper/results/patent_guild_hhi_compare.png  -- side-by-side all vs US
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

    print("[1/5] loading locations …", flush=True)
    loc = pl.read_csv(PV / "g_location_disambiguated.tsv", separator="\t",
                       null_values=[""]).select("location_id", "disambig_country")
    print(f"      {loc.height:,} locations")

    print("[2/5] loading inventors + patent dates + locations …", flush=True)
    inv = pl.read_csv(
        PV / "g_inventor_disambiguated.tsv", separator="\t", null_values=[""],
    ).select("patent_id", "inventor_id", "disambig_inventor_name_last", "location_id")
    pat = pl.read_csv(
        PV / "g_patent.tsv", separator="\t", null_values=[""],
        schema_overrides={"patent_id": pl.Utf8},
    ).select("patent_id", "patent_date")
    # Join location → inventor
    inv = inv.join(loc, on="location_id", how="left").rename({"disambig_country": "country"})
    inv = inv.join(pat, on="patent_id", how="left").with_columns(
        pl.col("patent_date").str.slice(0, 4).cast(pl.Int32).alias("year")
    )

    print("[3/5] loading primary CPC …", flush=True)
    cpc = pl.read_csv(
        PV / "g_cpc_current.tsv", separator="\t", null_values=[""],
        schema_overrides={"patent_id": pl.Utf8, "cpc_sequence": pl.Int64},
    ).filter(pl.col("cpc_sequence") == 0).select("patent_id", "cpc_class")
    inv = inv.join(cpc, on="patent_id", how="inner").filter(
        pl.col("year").is_not_null() & pl.col("disambig_inventor_name_last").is_not_null()
        & (pl.col("year") >= 1980) & (pl.col("year") <= 2024)
    )

    # US-only filter
    inv_us = inv.filter(pl.col("country") == "US")
    print(f"      total inventor-patent rows: {inv.height:,}")
    print(f"      US-resident rows:           {inv_us.height:,} ({100*inv_us.height/inv.height:.1f}%)")

    print("[4/5] normalizing surnames …", flush=True)
    surnames = [normalize_surname(s) for s in inv_us["disambig_inventor_name_last"].to_list()]
    inv_us = inv_us.with_columns(pl.Series("surname", surnames)).filter(pl.col("surname").is_not_null())

    # decade
    inv_us = inv_us.with_columns(((pl.col("year") // 10) * 10).alias("decade"))
    inv_uu = inv_us.unique(["inventor_id", "cpc_class", "decade"])
    print(f"      unique (US inventor × class × decade): {inv_uu.height:,}")

    print("[5/5] computing HHI per (cpc_class, decade) …", flush=True)
    sn_counts = inv_uu.group_by(["cpc_class","decade","surname"]).agg(pl.len().alias("n")).sort("n", descending=True)
    cell = sn_counts.group_by(["cpc_class","decade"]).agg(
        pl.col("n").sum().alias("total_inventors"),
        pl.col("surname").n_unique().alias("n_distinct_surnames"),
        (pl.col("n") / pl.col("n").sum()).pow(2).sum().alias("hhi"),
        (pl.col("n").head(5).sum() / pl.col("n").sum()).alias("top5_share"),
    ).filter(pl.col("total_inventors") >= 300).sort(["cpc_class","decade"])
    cell.write_csv(OUT / "patent_guild_hhi_us_panel.csv")

    pdf_us = cell.to_pandas()
    pdf_all = pl.read_csv(OUT / "patent_guild_hhi_panel.csv").to_pandas()

    # For comparison: build the rank tables on US-only
    pivot_us = pdf_us.pivot_table(index="cpc_class", columns="decade", values="hhi")
    pivot_us["delta"] = pivot_us.get(2020) - pivot_us.get(1990)
    rank_us = pivot_us.dropna(subset=["delta"]).sort_values("delta", ascending=False)
    print("\nTop 12 US-only CPC classes by HHI RISE 1990s→2020s:")
    print(rank_us.head(12)[[1990, 2000, 2010, 2020, "delta"]].to_string(float_format=lambda x: f"{x:.5f}"))
    print("\nBottom 8 US-only CPC classes (most diversifying):")
    print(rank_us.tail(8)[[1990, 2000, 2010, 2020, "delta"]].to_string(float_format=lambda x: f"{x:.5f}"))

    # Comparison plot: For the previously-top rising CPC classes,
    # plot HHI trajectory both all-inventors and US-only
    pivot_all = pdf_all.pivot_table(index="cpc_class", columns="decade", values="hhi")
    pivot_all["delta"] = pivot_all.get(2020) - pivot_all.get(1990)
    rank_all = pivot_all.dropna(subset=["delta"]).sort_values("delta", ascending=False)
    top_all_rising = rank_all.head(8).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(top_all_rising)))
    for i, cls in enumerate(top_all_rising):
        sub_all = pdf_all[pdf_all["cpc_class"] == cls].sort_values("decade")
        sub_us  = pdf_us[pdf_us["cpc_class"] == cls].sort_values("decade")
        axes[0].plot(sub_all["decade"], sub_all["hhi"], "-o", ms=4, color=cmap[i], lw=1.5, label=cls)
        axes[1].plot(sub_us["decade"],  sub_us["hhi"],  "-o", ms=4, color=cmap[i], lw=1.5, label=cls)
    for ax, ttl in zip(axes, ["All inventors (all-inventor HHI from previous run)",
                                "US-resident inventors only"]):
        ax.set_xlabel("Decade")
        ax.set_ylabel("Inventor-surname HHI")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, ncol=2, title="CPC class")
    fig.suptitle("Top 8 rising-HHI CPC classes: all-inventor vs US-resident-inventor HHI",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "patent_guild_hhi_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Single-panel: US-only top-rising
    top_us = rank_us.head(8).index.tolist()
    bot_us = rank_us.tail(8).index.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    cmap_r = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_us)))
    cmap_f = plt.cm.Blues(np.linspace(0.4, 0.9, len(bot_us)))
    for i, cls in enumerate(top_us):
        sub = pdf_us[pdf_us["cpc_class"] == cls].sort_values("decade")
        axes[0].plot(sub["decade"], sub["hhi"], "-o", ms=4, color=cmap_r[i], lw=1.5, label=cls)
    for i, cls in enumerate(bot_us):
        sub = pdf_us[pdf_us["cpc_class"] == cls].sort_values("decade")
        axes[1].plot(sub["decade"], sub["hhi"], "-o", ms=4, color=cmap_f[i], lw=1.5, label=cls)
    axes[0].set_title("US-resident only: top 8 CPC classes by HHI RISE 1990s→2020s")
    axes[1].set_title("US-resident only: top 8 CPC classes by HHI FALL (diversifying)")
    for ax in axes:
        ax.set_ylabel("Surname HHI")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
    axes[1].set_xlabel("Decade")
    fig.tight_layout()
    fig.savefig(OUT / "patent_guild_hhi_us.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Quick check: which previously-top-rising classes deflate
    print("\nFor the all-inventor top-8 RISING classes, how do US-only deltas compare?")
    print(f"{'CPC':>4}  {'ΔHHI all':>10}  {'ΔHHI US':>10}  {'shift':>10}")
    for cls in top_all_rising:
        d_all = rank_all.loc[cls, "delta"] if cls in rank_all.index else None
        d_us  = rank_us.loc[cls,  "delta"] if cls in rank_us.index  else None
        if d_all is None or d_us is None:
            print(f"{cls:>4}  {'-':>10}  {'-':>10}")
        else:
            shift = d_us - d_all
            print(f"{cls:>4}  {d_all:>+.5f}  {d_us:>+.5f}  {shift:>+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
