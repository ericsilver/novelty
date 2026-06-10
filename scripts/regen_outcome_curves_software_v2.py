"""Regenerate paper/results/outcome_curves_software.png as the conditional
5-year survival curve (denominator = filings that reached registration),
single line per panel, no legend, ylabel relabelled as the renewal rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(r"C:\shared-secure\research\novelty")
PROC = REPO / "data" / "processed"
OUT = REPO / "paper" / "results" / "outcome_curves_software.png"
TODAY_YEAR = 2026


def derive(records: pl.DataFrame, surprise: pl.DataFrame) -> pl.DataFrame:
    base = records.select(
        "serial_number", "filing_date", "status_code"
    ).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"),
        pl.col("status_code").str.slice(0, 1).alias("status_bucket"),
    ).with_columns(
        (
            (pl.col("status_bucket") == "6") | (pl.col("status_bucket") == "8")
        ).alias("reached_registration"),
    ).with_columns(
        # Conditional on reached_registration, "renewed" = still alive
        # (status bucket 6: live registered, maintained). Bucket 8 here
        # means the mark died post-registration for any reason (Section 8
        # failure, Section 9 failure, or other cancellation).
        (pl.col("status_bucket") == "6").alias("survived_5y"),
    )

    sup = surprise.select(
        "serial_number", "n_terms",
        "prospective_kl", "retrospective_kl",
        "n_ref_prospective", "n_ref_retrospective",
    ).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl")
    )
    return base.join(sup, on="serial_number", how="left")


def wilson(p, n, z=1.96):
    if n <= 0:
        return p, p
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def main() -> int:
    print("loading class 009 records + surprise", file=sys.stderr)
    rec = pl.read_parquet(PROC / "tm_class009.parquet")
    sup = pl.read_parquet(PROC / "surprise_class009.parquet")
    df = derive(rec, sup)

    clean = (
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= 1990) & (pl.col("year") <= 2020)
        & pl.col("reached_registration")  # CONDITIONAL on registration
    )
    df = df.filter(clean)
    n_total = df.height
    mean_rate = float(df["survived_5y"].mean())
    print(f"n filings (conditional on reg, 1990--2020, clean): {n_total:,}", file=sys.stderr)
    print(f"5y renewal rate among registered: {mean_rate*100:.1f}%", file=sys.stderr)

    kl_edges = np.array([0.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0])
    kl_centres = 0.5 * (kl_edges[:-1] + kl_edges[1:])
    d_edges = np.array([-2.0, -1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    d_centres = 0.5 * (d_edges[:-1] + d_edges[1:])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    panel_specs = [
        ("prospective_kl", "Prospective KL  (novel vs. past)", kl_edges, kl_centres),
        ("retrospective_kl", "Retrospective KL  (novel vs. future)", kl_edges, kl_centres),
        ("dkl", r"$\Delta KL$ (pros $-$ retr)  (innovator $\rightarrow$ right)", d_edges, d_centres),
    ]
    for ax, (xcol, xlabel, edges, centres) in zip(axes, panel_specs):
        d_pd = df.select(pl.col(xcol), pl.col("survived_5y").cast(pl.Int8)).drop_nulls().to_pandas()
        d_pd["bin"] = np.digitize(d_pd[xcol], edges) - 1
        d_pd = d_pd[(d_pd["bin"] >= 0) & (d_pd["bin"] < len(centres))]
        grp = d_pd.groupby("bin").agg(rate=("survived_5y", "mean"), n=("survived_5y", "size"))
        grp = grp[grp["n"] >= 50]
        xs = centres[grp.index]
        ys = grp["rate"].values
        lo_arr = np.array([wilson(y, n)[0] for y, n in zip(ys, grp["n"].values)])
        hi_arr = np.array([wilson(y, n)[1] for y, n in zip(ys, grp["n"].values)])
        ax.fill_between(xs, lo_arr, hi_arr, color="#d62728", alpha=0.22, linewidth=0)
        ax.plot(xs, ys, "o-", color="#d62728", linewidth=1.6, markersize=5)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("5-year renewal rate (survival)")
    axes[2].axvline(0, color="grey", linewidth=0.7, linestyle="--")
    axes[1].set_title("Software & Electronics — conditional on completed registration")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
