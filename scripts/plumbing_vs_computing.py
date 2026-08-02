"""Plumbing vs computing innovation gap, per the user's "sexy sector
swarm" hypothesis. Compare mean ΔKL by year across:

  NICE 037: Construction & Repair (includes plumbing services)
  NICE 009: Electronics & Software
  NICE 042: Scientific & Tech Services (R&D, software dev, design)
  NICE 044: Medical/Hygiene
  NICE 035: Advertising & Retail business services
  NICE 043: Hotel/Restaurant/Food

Is the "less-sexy" class (037) running with low vocabulary churn
(low |ΔKL|) -- a stagnation signal -- or with high harvest-tail
(negative ΔKL) -- a "tradition without innovation" signal?

Outputs:
  paper/results/plumbing_vs_computing.csv
  paper/results/plumbing_vs_computing.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

CLASSES = {
    "009": ("Electronics & Software",  "#2b6cb0"),
    "037": ("Construction & Repair",   "#cc4444"),  # "plumbing" lives here
    "042": ("Scientific & Tech Svc",   "#229922"),
    "044": ("Medical & Veterinary",    "#7a3eb2"),
    "035": ("Advertising & Retail",    "#cc7a00"),
    "043": ("Hotel / Restaurant",      "#999999"),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, _ in CLASSES.items():
        sp = PROC / f"surprise_class{cls}.parquet"
        if not sp.exists(): continue
        s = pl.read_parquet(sp, columns=["serial_number","year",
            "kl_vs_past","kl_vs_future",
            "n_ref_past","n_ref_future","n_terms"]).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
            & pl.col("year").is_between(1990, 2024)
        ).with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
            pl.col("dkl").abs().alias("abs_dkl") if False else (pl.col("kl_vs_past") - pl.col("kl_vs_future")).abs().alias("abs_dkl"),
        )
        by_year = s.group_by("year").agg(
            pl.len().alias("n"),
            pl.col("dkl").mean().alias("mean_dkl"),
            pl.col("dkl").std().alias("std_dkl"),
            pl.col("abs_dkl").mean().alias("mean_abs_dkl"),
            (pl.col("dkl") > 0).cast(pl.Float64).mean().alias("share_pos"),
        ).with_columns(pl.lit(cls).alias("class_id")).sort("year")
        rows.append(by_year)
    panel = pl.concat(rows)
    panel.write_csv(OUT / "plumbing_vs_computing.csv")

    pdf = panel.to_pandas()
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for cls, (name, color) in CLASSES.items():
        sub = pdf[pdf["class_id"] == cls]
        axes[0].plot(sub["year"], sub["mean_dkl"], "-o", ms=3, color=color, lw=1.7, label=name)
        axes[1].plot(sub["year"], sub["mean_abs_dkl"], "-o", ms=3, color=color, lw=1.7, label=name)
    axes[0].axhline(0, color="#444", lw=0.7)
    axes[0].set_ylabel(r"Mean $\Delta KL$ (signed)")
    axes[0].set_title("Signed $\\Delta KL$: positive = innovation tail, negative = harvest tail")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9, ncol=2)
    axes[1].set_ylabel(r"Mean $|\Delta KL|$")
    axes[1].set_xlabel("Year")
    axes[1].set_title("Magnitude of $|\\Delta KL|$: vocabulary churn rate (regardless of direction)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Plumbing vs computing: does the \"less-sexy\" sector stagnate or harvest?",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "plumbing_vs_computing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Mean ΔKL by class, selected periods:")
    for period, lo, hi in [("1995-1999",1995,1999),("2010-2014",2010,2014),("2020-2024",2020,2024)]:
        sub = pdf[(pdf["year"] >= lo) & (pdf["year"] <= hi)]
        print(f"\n{period}:")
        for cls, (name, _) in CLASSES.items():
            cs = sub[sub["class_id"] == cls]
            if not len(cs): continue
            wmean = (cs["mean_dkl"] * cs["n"]).sum() / cs["n"].sum()
            wabs  = (cs["mean_abs_dkl"] * cs["n"]).sum() / cs["n"].sum()
            print(f"  {cls} {name:30s}  signed ΔKL={wmean:+.4f}  |ΔKL|={wabs:.4f}  n={cs['n'].sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
