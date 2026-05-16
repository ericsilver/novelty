"""Per-industry deck of survival topomaps.
Page 1: all-classes summary (parallel to scripts/survival_topomap.py).
Subsequent pages: one per NICE class with at least 5,000 clean filings,
sorted by size desc.

For each owner_name (within the page's universe), compute (mean_pros,
mean_retr) and survival outcomes, bin into 36x36, color by mean.

Output: paper/results/survival_topomap_all_industries.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

# Cohort cap moved to filing_year ≤ 2018 so the 5y Section-8 window has
# been adjudicated for nearly every registration (filing_year + ~7 ≤ 2026).
CLEAN = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1990) & (pl.col("year") <= 2018)
)

N_BINS = 36
MIN_FIRMS_PER_FIRM = 3
MIN_FILINGS_PER_PAGE = 5000
MIN_FIRMS_FOR_PAGE = 200  # don't render a page if firm panel is tiny


def _load_class(cls: str) -> pl.DataFrame:
    return (pl.read_parquet(PROC / f"outcomes_class{cls}.parquet")
            .filter(CLEAN & pl.col("owner_name").is_not_null())
            .with_columns(
                (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
            )
            .select("owner_name", "prospective_kl", "retrospective_kl",
                    "reached_registration", "passed_5y"))


def _firm_panel(universe: pl.DataFrame) -> pd.DataFrame:
    return universe.group_by("owner_name").agg(
        pl.col("prospective_kl").mean().alias("mean_pros"),
        pl.col("retrospective_kl").mean().alias("mean_retr"),
        pl.col("reached_registration").mean().alias("mean_reach"),
        pl.col("passed_5y").mean().alias("mean_passed5"),
        pl.len().alias("n_filings"),
    ).filter(pl.col("n_filings") >= MIN_FIRMS_PER_FIRM).to_pandas()


def render_page(panel: pd.DataFrame, title: str, pdf: PdfPages) -> None:
    if len(panel) < MIN_FIRMS_FOR_PAGE:
        print(f"  SKIP (only {len(panel):,} firms): {title}", flush=True)
        return

    p_lo, p_hi = panel["mean_pros"].quantile([0.02, 0.98])
    r_lo, r_hi = panel["mean_retr"].quantile([0.02, 0.98])
    p_edges = np.linspace(p_lo, p_hi, N_BINS + 1)
    r_edges = np.linspace(r_lo, r_hi, N_BINS + 1)

    panel = panel.copy()
    panel["p_bin"] = np.digitize(panel["mean_pros"], p_edges) - 1
    panel["r_bin"] = np.digitize(panel["mean_retr"], r_edges) - 1
    panel = panel[(panel["p_bin"] >= 0) & (panel["p_bin"] < N_BINS)
                  & (panel["r_bin"] >= 0) & (panel["r_bin"] < N_BINS)]

    def heatmap(metric: str, agg: str = "mean", min_n: int = 1):
        H = np.full((N_BINS, N_BINS), np.nan)
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
        ("mean_passed5", "std", "magma", None,
         "Within-firm std.\\ of still-live outcomes"),
        ("n_filings", "count", "Greys", "log",
         "Density: number of firms in each cell (log)"),
    ]

    # Cache per-metric H for re-use across the three pages
    H_cache = {(metric, agg): heatmap(metric, agg=agg)
               for metric, agg, *_ in panels}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (metric, agg, cmap_name, scale, sub_title) in zip(axes.flat, panels):
        H = H_cache[(metric, agg)]
        if scale == "log":
            vmax = np.nanmax(H) if np.any(~np.isnan(H)) else 1
            im = ax.pcolormesh(p_edges, r_edges, H, cmap=cmap_name,
                               shading="flat", norm=LogNorm(vmin=1, vmax=max(vmax, 2)))
        else:
            im = ax.pcolormesh(p_edges, r_edges, H, cmap=cmap_name, shading="flat")
        diag_lo = max(p_lo, r_lo)
        diag_hi = min(p_hi, r_hi)
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="black",
                linewidth=0.7, linestyle="--", alpha=0.7)
        ax.set_xlabel(r"Mean prospective KL  (firm)")
        ax.set_ylabel(r"Mean retrospective KL  (firm)")
        ax.set_title(sub_title)
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
        f"{title}\n"
        f"each cell averaged over firms with $\\geq {MIN_FIRMS_PER_FIRM}$ clean filings; "
        f"{N_BINS}$\\times${N_BINS} grid; min 1 firm per cell to colour; "
        f"n={len(panel):,} firms",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---------- Page 2: zoom into lower-left, percentile-clipped colormap ----------
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _topomap_zoom import draw_zoom_panel, draw_contour_panel

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (metric, agg, cmap_name, scale, sub_title) in zip(axes.flat, panels):
        H = H_cache[(metric, agg)]
        im = draw_zoom_panel(ax, p_edges, r_edges, H, cmap=cmap_name,
                             title=sub_title, log_scale=(scale == "log"))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"{title}  —  ZOOM (lower-left commonplace quadrant; "
        f"colour scale clipped to central 5--95\\% of cell values; "
        f"isoscore contours overlaid)\nn={len(panel):,} firms",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---------- Page 3: smoothed-contour topographic view ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (metric, agg, cmap_name, scale, sub_title) in zip(axes.flat, panels):
        if scale == "log":
            ax.axis("off")
            continue
        H = H_cache[(metric, agg)]
        cs = draw_contour_panel(ax, p_edges, r_edges, H, cmap=cmap_name,
                                title=sub_title, sigma_cells=1.5)
        cbar = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"{title}  —  TOPOGRAPHIC (Gaussian-smoothed; isoscore lines)  "
        f"n={len(panel):,} firms",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    print("[load] all clean filings (per-class)…", flush=True)
    classes = sorted(p.stem.replace("outcomes_class", "")
                     for p in PROC.glob("outcomes_class*.parquet")
                     if p.stem.replace("outcomes_class", "").isdigit())

    per_class: dict[str, pl.DataFrame] = {}
    for cls in classes:
        per_class[cls] = _load_class(cls)
        print(f"  class {cls}: {per_class[cls].height:,} clean filings", flush=True)

    universe = pl.concat(list(per_class.values()))
    print(f"[summary] {universe.height:,} clean filings across all classes", flush=True)

    out = RESULTS / "survival_topomap_all_industries.pdf"
    with PdfPages(out) as pdf:
        # Page 1: summary
        all_panel = _firm_panel(universe)
        print(f"  summary panel: {len(all_panel):,} firms with >=3 filings", flush=True)
        render_page(
            all_panel,
            "Trademark survival outcomes on the (prospective KL, retrospective KL) plane "
            "— ALL CLASSES",
            pdf,
        )

        # Subsequent pages: one per industry, sorted by clean-filing count desc
        sizes = [(cls, df.height) for cls, df in per_class.items()
                 if df.height >= MIN_FILINGS_PER_PAGE]
        sizes.sort(key=lambda t: -t[1])

        for cls, n in sizes:
            label = industry_name(cls)
            print(f"[page] class {cls} {label} ({n:,} clean)", flush=True)
            page_panel = _firm_panel(per_class[cls])
            render_page(
                page_panel,
                f"Class {cls} — {label} — {n:,} clean filings",
                pdf,
            )

    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
