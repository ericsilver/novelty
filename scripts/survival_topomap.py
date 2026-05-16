"""Topographic heatmaps of trademark-survival outcomes by 2D position
on the (prospective KL, retrospective KL) plane. Parallel structure to
profit_topomap.py but with within-corpus survival outcomes (every
filing is observed; no SEC join needed).

For each owner_name with >=3 clean filings, compute (mean_pros,
mean_retr) across the firm's filings and the firm's mean survival
outcome. Bin into a 36x36 grid; min 1 firm per cell.

Output: paper/results/survival_topomap.png
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

# Cohort cap moved to filing_year ≤ 2018 so that for nearly every
# registration the 5y Section-8 window has already been adjudicated by
# 2026 (typical filing → registration lag ~1–2y; Section 8 due 5y after
# that ⇒ ~filing_year + 7).
CLEAN = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1990) & (pl.col("year") <= 2018)
)


def main() -> int:
    print("[load] firm-level KL means + survival outcomes…", flush=True)
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit(): continue
        parts.append(pl.read_parquet(path).filter(
            CLEAN & pl.col("owner_name").is_not_null()
        ).select("owner_name", "prospective_kl", "retrospective_kl",
                 "reached_registration", "currently_live")
        .with_columns(
            (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
        ).select("owner_name", "prospective_kl", "retrospective_kl",
                 "reached_registration", "passed_5y"))
    universe = pl.concat(parts)
    print(f"  {universe.height:,} clean filings, all classes", flush=True)

    firm = universe.group_by("owner_name").agg(
        pl.col("prospective_kl").mean().alias("mean_pros"),
        pl.col("retrospective_kl").mean().alias("mean_retr"),
        pl.col("reached_registration").mean().alias("mean_reach"),
        pl.col("passed_5y").mean().alias("mean_passed5"),
        pl.len().alias("n_filings"),
    ).filter(pl.col("n_filings") >= 3)
    print(f"  {firm.height:,} owners with >=3 clean filings", flush=True)

    panel = firm.to_pandas()

    # 36x36 bins on the central 2nd-98th-percentile range
    p_lo, p_hi = panel["mean_pros"].quantile([0.02, 0.98])
    r_lo, r_hi = panel["mean_retr"].quantile([0.02, 0.98])
    n_bins = 36
    p_edges = np.linspace(p_lo, p_hi, n_bins + 1)
    r_edges = np.linspace(r_lo, r_hi, n_bins + 1)

    panel["p_bin"] = np.digitize(panel["mean_pros"], p_edges) - 1
    panel["r_bin"] = np.digitize(panel["mean_retr"], r_edges) - 1
    panel = panel[(panel["p_bin"] >= 0) & (panel["p_bin"] < n_bins)
                  & (panel["r_bin"] >= 0) & (panel["r_bin"] < n_bins)]

    def heatmap(metric: str, agg: str = "mean", min_n: int = 1):
        H = np.full((n_bins, n_bins), np.nan)
        sub = panel.dropna(subset=[metric])
        grouped = sub.groupby(["r_bin", "p_bin"])[metric].agg([agg, "size"]).reset_index()
        grouped.columns = ["r_bin", "p_bin", "value", "size"]
        for _, row in grouped.iterrows():
            if row["size"] >= min_n:
                H[int(row["r_bin"]), int(row["p_bin"])] = row["value"]
        return H

    panels = [
        ("mean_reach", "mean", "viridis", None,
         "Mean Completed-Registration rate per firm"),
        ("mean_passed5", "mean", "viridis", None,
         "Mean Still-Live rate per firm  (registered \\& not cancelled)"),
        # Firm-level "consistency": std of passed_5y across the firm's filings.
        ("mean_passed5", "std", "magma", None,
         "Within-firm std.\\ of still-live outcomes"),
        # Density: how many firms in each cell (log scale)
        ("n_filings", "count", "Greys", "log",
         "Density: number of firms in each cell (log)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (metric, agg, cmap_name, scale, title) in zip(axes.flat, panels):
        H = heatmap(metric, agg=agg)
        if scale == "log":
            from matplotlib.colors import LogNorm
            im = ax.pcolormesh(p_edges, r_edges, H, cmap=cmap_name,
                               shading="flat", norm=LogNorm(vmin=1, vmax=np.nanmax(H)))
        else:
            im = ax.pcolormesh(p_edges, r_edges, H, cmap=cmap_name, shading="flat")
        diag_lo = max(p_lo, r_lo)
        diag_hi = min(p_hi, r_hi)
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="black",
                linewidth=0.7, linestyle="--", alpha=0.7)
        ax.set_xlabel(r"Mean prospective KL  (firm)")
        ax.set_ylabel(r"Mean retrospective KL  (firm)")
        ax.set_title(title)
        ax.set_xlim(p_lo, p_hi)
        ax.set_ylim(r_lo, r_hi)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        ax.text(0.99, 0.02, "innovator\n(below diag.)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#117a3a",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#117a3a", alpha=0.7))
        ax.text(0.02, 0.98, "tired\n(above diag.)",
                transform=ax.transAxes, va="top",
                fontsize=8, color="#7a1111",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#7a1111", alpha=0.7))
        ax.text(0.99, 0.98, "novel\n(top-right)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#1f3a7a",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f3a7a", alpha=0.7))
        ax.text(0.02, 0.02, "commonplace\n(lower-left)",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#555555", alpha=0.7))

    fig.suptitle(
        "Trademark post-registration durability on the (prospective KL, retrospective KL) plane\n"
        "still-live $=$ reached registration AND not cancelled-post-reg "
        "(filing years 1990--2018, so the 5y Section-8 window has been adjudicated)\n"
        f"each cell averaged over firms with $\\geq 3$ clean filings; "
        f"36$\\times$36 grid; min 1 firm per cell to colour; n={panel.shape[0]:,} firms total",
        y=1.04, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "survival_topomap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {RESULTS / 'survival_topomap.png'}")

    # ---------- Zoom + contour variants ----------
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _topomap_zoom import draw_zoom_panel, draw_contour_panel

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (metric, agg, cmap_name, scale, title) in zip(axes.flat, panels):
        H = heatmap(metric, agg=agg)
        im = draw_zoom_panel(ax, p_edges, r_edges, H, cmap=cmap_name,
                             title=title, log_scale=(scale == "log"))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Zoom into the lower-left ``commonplace'' quadrant — same data, "
        "axes cropped, colour scale clipped to the central 5--95\\% of cell values\n"
        "and Gaussian-smoothed isoscore contours overlaid (white) "
        f"to highlight the diagonal valley.  n={panel.shape[0]:,} firms.",
        y=1.04, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "survival_topomap_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {RESULTS / 'survival_topomap_zoom.png'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (metric, agg, cmap_name, scale, title) in zip(axes.flat, panels):
        if scale == "log":
            continue  # contour-fill on a log density isn't useful
        H = heatmap(metric, agg=agg)
        cs = draw_contour_panel(ax, p_edges, r_edges, H, cmap=cmap_name,
                                title=title, sigma_cells=1.5)
        cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    # The density panel doesn't smooth meaningfully — leave its axis blank.
    axes[1, 1].axis("off")
    fig.suptitle(
        "Topographic (smoothed-contour) view: full plane, "
        "Gaussian-smoothed cell values rendered as filled contours with "
        "isoscore lines.  Colour scale clipped to the central 5--95\\% of "
        f"cell values to keep the diagonal valley legible.  n={panel.shape[0]:,} firms.",
        y=1.04, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "survival_topomap_contour.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {RESULTS / 'survival_topomap_contour.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
