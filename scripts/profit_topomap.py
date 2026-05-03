"""Topographic heatmaps of firm financial outcomes by 2D position
on the (prospective KL, retrospective KL) plane.

For each firm in the matched SEC panel, compute (mean_pros, mean_retr)
across all the firm's filings; then bin into a 2D grid and average four
financial outcomes within each cell:

  (a) mean gross margin
  (b) within-firm gross margin standard deviation (variance proxy)
  (c) mean 1-year excess return over S&P 500
  (d) mean 4-year excess return over S&P 500

Cells with too few firms are blanked. The diagonal pros = retr is
overlaid as a reference; below-diagonal cells correspond to imitated-
innovator filings (high pros, low retr); above-diagonal to filings the
future never resembled.

Output: paper/results/profit_topomap.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

CLEAN = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1990) & (pl.col("year") <= 2020)
)


def main() -> int:
    print("[load] firm-level KL means + financials + returns…", flush=True)
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name", "cik")

    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit(): continue
        parts.append(pl.read_parquet(path).filter(CLEAN & pl.col("owner_name").is_not_null())
                     .select("owner_name", "prospective_kl", "retrospective_kl"))
    universe = pl.concat(parts).join(cw, on="owner_name", how="inner")

    # Per-firm mean prospective and retrospective KL
    firm_kl = universe.group_by("cik").agg(
        pl.col("prospective_kl").mean().alias("mean_pros"),
        pl.col("retrospective_kl").mean().alias("mean_retr"),
        pl.len().alias("n_filings"),
    ).filter(pl.col("n_filings") >= 3)
    print(f"  firm-level KL means: {firm_kl.height:,} firms with >=3 filings")

    # Per-firm financial summaries from SEC
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet").filter(
        (pl.col("revenue") > 0)
        & pl.col("gross_margin").is_not_null()
        & (pl.col("gross_margin") > -1) & (pl.col("gross_margin") < 1.5)
    )
    firm_fin = sec.group_by("cik").agg(
        pl.col("gross_margin").mean().alias("mean_gm"),
        pl.col("gross_margin").std().alias("std_gm"),
        pl.col("fy").n_unique().alias("n_years_fin"),
    ).filter(pl.col("n_years_fin") >= 3)

    # Per-firm return summaries
    rets = pl.read_parquet(PROC / "stock_returns.parquet")
    rets_pd = rets.to_pandas().sort_values(["cik", "base_year"])
    for k in (1, 4):
        rets_pd[f"excess_{k}y"] = rets_pd[f"ret_{k}y"] - rets_pd[f"sp500_ret_{k}y"]
    # Winsorize each horizon at 1/99
    for k in (1, 4):
        col = f"excess_{k}y"
        lo, hi = rets_pd[col].quantile([0.01, 0.99])
        rets_pd[col] = rets_pd[col].clip(lo, hi)
    firm_rets = rets_pd.groupby("cik").agg(
        mean_excess_1y=("excess_1y", "mean"),
        mean_excess_4y=("excess_4y", "mean"),
        n_yrs_ret=("excess_1y", "count"),
    ).reset_index()
    firm_rets = firm_rets[firm_rets["n_yrs_ret"] >= 3]

    # Join all
    panel = (
        firm_kl.to_pandas()
        .merge(firm_fin.to_pandas(), on="cik", how="left")
        .merge(firm_rets, on="cik", how="left")
    )
    print(f"  joined firm panel: {len(panel):,} firms")
    print(f"    with gross margin: {panel['mean_gm'].notna().sum():,}")
    print(f"    with excess 1y return: {panel['mean_excess_1y'].notna().sum():,}")
    print(f"    with excess 4y return: {panel['mean_excess_4y'].notna().sum():,}")

    # 2D bins on (mean_pros, mean_retr)
    # Use the central range where most firms live
    p_lo, p_hi = panel["mean_pros"].quantile([0.02, 0.98])
    r_lo, r_hi = panel["mean_retr"].quantile([0.02, 0.98])
    n_bins = 12
    p_edges = np.linspace(p_lo, p_hi, n_bins + 1)
    r_edges = np.linspace(r_lo, r_hi, n_bins + 1)
    p_centres = 0.5 * (p_edges[:-1] + p_edges[1:])
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])

    panel["p_bin"] = np.digitize(panel["mean_pros"], p_edges) - 1
    panel["r_bin"] = np.digitize(panel["mean_retr"], r_edges) - 1
    panel = panel[(panel["p_bin"] >= 0) & (panel["p_bin"] < n_bins)
                  & (panel["r_bin"] >= 0) & (panel["r_bin"] < n_bins)]

    def heatmap(metric: str, min_n: int = 5):
        H = np.full((n_bins, n_bins), np.nan)
        N = np.zeros((n_bins, n_bins), dtype=int)
        sub = panel.dropna(subset=[metric])
        grouped = sub.groupby(["r_bin", "p_bin"])[metric].agg(["mean", "size"]).reset_index()
        for _, row in grouped.iterrows():
            if row["size"] >= min_n:
                H[int(row["r_bin"]), int(row["p_bin"])] = row["mean"]
            N[int(row["r_bin"]), int(row["p_bin"])] = int(row["size"])
        return H, N

    panels = [
        ("mean_gm",          "Mean gross margin",            "viridis", "fraction"),
        ("std_gm",           "Within-firm gross-margin std.\\ deviation", "magma", "fraction"),
        ("mean_excess_1y",   "Mean 1-year excess return vs.\\ S\\&P 500", "RdYlGn", "fraction"),
        ("mean_excess_4y",   "Mean 4-year excess return vs.\\ S\\&P 500", "RdYlGn", "fraction"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (metric, title, cmap_name, _) in zip(axes.flat, panels):
        H, N = heatmap(metric)
        if metric in ("mean_excess_1y", "mean_excess_4y"):
            absmax = np.nanmax(np.abs(H)) if np.any(~np.isnan(H)) else 0.5
            absmax = max(absmax, 0.05)
            im = ax.pcolormesh(
                p_edges, r_edges, H,
                cmap=cmap_name, vmin=-absmax, vmax=absmax, shading="flat",
            )
        else:
            im = ax.pcolormesh(
                p_edges, r_edges, H,
                cmap=cmap_name, shading="flat",
            )
        # Diagonal pros = retr
        diag_lo = max(p_lo, r_lo)
        diag_hi = min(p_hi, r_hi)
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="black",
                linewidth=0.7, linestyle="--", alpha=0.7)
        # n labels in each cell where data is shown
        for i in range(n_bins):
            for j in range(n_bins):
                if not np.isnan(H[i, j]) and N[i, j] >= 5:
                    ax.text(p_centres[j], r_centres[i], f"{N[i,j]}",
                            ha="center", va="center", fontsize=6,
                            color="white", alpha=0.7)
        ax.set_xlabel(r"Mean prospective KL  (firm)")
        ax.set_ylabel(r"Mean retrospective KL  (firm)")
        ax.set_title(title)
        ax.set_xlim(p_lo, p_hi)
        ax.set_ylim(r_lo, r_hi)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        # Quadrant labels
        ax.text(0.99, 0.02, "innovator\n(below diag.)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#117a3a",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#117a3a", alpha=0.7))
        ax.text(0.02, 0.98, "tired\n(above diag.)",
                transform=ax.transAxes, va="top",
                fontsize=8, color="#7a1111",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#7a1111", alpha=0.7))

    fig.suptitle(
        f"Firm financial outcomes on the (prospective KL, retrospective KL) plane\n"
        f"each cell averaged over firms with $\\geq 3$ filings ($\\geq 3$ firm-years for outcome); "
        f"min 5 firms per cell to colour; small numbers in cells = N firms",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "profit_topomap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {RESULTS / 'profit_topomap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
