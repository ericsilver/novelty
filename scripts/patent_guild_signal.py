"""Patent-side guild signal: surname concentration within CPC class
over decades. Rising HHI within a class = the same surnames are
inventing more of that class's patents, which is the inheritance /
ethnic-cluster signature.

For each (cpc_class, decade):
  - inventor-surname HHI (sum of squared shares)
  - top-5 surname share
  - n unique surnames
  - n unique inventors

Then identify CPC classes where HHI is rising fastest -- candidates
for "guild-like" inheritance dynamics.

Outputs:
  paper/results/patent_guild_hhi_panel.csv
  paper/results/patent_guild_hhi.png
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

    print("[1/4] loading inventors + patent dates …", flush=True)
    inv = pl.read_csv(
        PV / "g_inventor_disambiguated.tsv", separator="\t", null_values=[""],
    ).select("patent_id", "inventor_id", "disambig_inventor_name_last")
    pat = pl.read_csv(
        PV / "g_patent.tsv", separator="\t", null_values=[""],
        schema_overrides={"patent_id": pl.Utf8},
    ).select("patent_id", "patent_date")
    inv = inv.join(pat, on="patent_id", how="left").with_columns(
        pl.col("patent_date").str.slice(0, 4).cast(pl.Int32).alias("year")
    )

    print("[2/4] loading CPC classifications (primary class) …", flush=True)
    cpc = pl.read_csv(
        PV / "g_cpc_current.tsv", separator="\t", null_values=[""],
        schema_overrides={"patent_id": pl.Utf8, "cpc_sequence": pl.Int64},
    ).filter(pl.col("cpc_sequence") == 0).select("patent_id", "cpc_class")
    print(f"      {cpc.height:,} primary CPC rows")

    print("[3/4] joining + normalizing surnames …", flush=True)
    inv = inv.join(cpc, on="patent_id", how="inner").filter(
        pl.col("year").is_not_null() & pl.col("disambig_inventor_name_last").is_not_null()
        & (pl.col("year") >= 1980) & (pl.col("year") <= 2024)
    )
    surnames = [normalize_surname(s) for s in inv["disambig_inventor_name_last"].to_list()]
    inv = inv.with_columns(pl.Series("surname", surnames)).filter(pl.col("surname").is_not_null())

    # Decade
    inv = inv.with_columns(((pl.col("year") // 10) * 10).alias("decade"))
    # Unique (inventor, cpc_class, decade) — one row per inventor per class per decade
    inv_u = inv.unique(["inventor_id", "cpc_class", "decade"])
    print(f"      {inv_u.height:,} unique (inventor × class × decade) rows")

    print("[4/4] computing surname concentration per (cpc_class, decade) …", flush=True)
    # For each (cpc_class, decade): compute surname HHI and top-5 share
    sn_counts = inv_u.group_by(["cpc_class","decade","surname"]).agg(pl.len().alias("n")).sort("n", descending=True)
    cell = sn_counts.group_by(["cpc_class","decade"]).agg(
        pl.col("n").sum().alias("total_inventors"),
        pl.col("surname").n_unique().alias("n_distinct_surnames"),
        (pl.col("n") / pl.col("n").sum()).pow(2).sum().alias("hhi"),
        (pl.col("n").head(5).sum() / pl.col("n").sum()).alias("top5_share"),
    ).filter(pl.col("total_inventors") >= 500).sort(["cpc_class","decade"])
    cell.write_csv(OUT / "patent_guild_hhi_panel.csv")
    print(f"      {cell.height:,} (class, decade) cells with ≥500 inventors")

    # Compute change in HHI 1980s → 2020s for each class
    pdf = cell.to_pandas()
    pivot = pdf.pivot_table(index="cpc_class", columns="decade", values="hhi")
    pivot["delta_2020s_vs_1990s"] = pivot.get(2020) - pivot.get(1990)
    rank = pivot.dropna(subset=["delta_2020s_vs_1990s"]).sort_values("delta_2020s_vs_1990s", ascending=False)
    print("\nTop 15 CPC classes with biggest HHI rise (1990s → 2020s):")
    print(rank.head(15)[[1990, 2000, 2010, 2020, "delta_2020s_vs_1990s"]].to_string(float_format=lambda x: f"{x:.5f}"))
    print("\nBottom 15 CPC classes (biggest HHI fall = most diversification):")
    print(rank.tail(15)[[1990, 2000, 2010, 2020, "delta_2020s_vs_1990s"]].to_string(float_format=lambda x: f"{x:.5f}"))

    # Plot: time series for top-rising vs top-falling classes
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    decades = sorted(pdf["decade"].unique())
    top_rising = rank.head(8).index.tolist()
    top_falling = rank.tail(8).index.tolist()

    cmap_r = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_rising)))
    cmap_f = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_falling)))

    ax = axes[0]
    for i, cls in enumerate(top_rising):
        sub = pdf[pdf["cpc_class"] == cls].sort_values("decade")
        ax.plot(sub["decade"], sub["hhi"], "-o", ms=4, color=cmap_r[i], lw=1.5, label=cls)
    ax.set_ylabel("Inventor-surname HHI")
    ax.set_title("Top 8 CPC classes by HHI RISE 1990s → 2020s (guild-like concentration)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[1]
    for i, cls in enumerate(top_falling):
        sub = pdf[pdf["cpc_class"] == cls].sort_values("decade")
        ax.plot(sub["decade"], sub["hhi"], "-o", ms=4, color=cmap_f[i], lw=1.5, label=cls)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Inventor-surname HHI")
    ax.set_title("Top 8 CPC classes by HHI FALL (most diversification of inventor surnames)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle("Patent-inventor surname concentration over time by CPC primary class",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "patent_guild_hhi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] {OUT/'patent_guild_hhi.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
