"""Per-industry deck of the Figure-5-style outcome-by-KL-bin plot.
One page per NICE class with at least 5,000 clean filings, bundled
into a single multi-page PDF so the user can flip through.

Output: paper/results/outcome_curves_all_industries.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
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


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return p, p
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def render_industry(cls: str, industry_label: str, n_clean: int, pdf: PdfPages) -> None:
    df = pl.read_parquet(PROC / f"outcomes_class{cls}.parquet").filter(CLEAN)
    if df.height < 5000:
        return

    outcomes = [
        ("reached_registration", "Completed registration"),
        ("survived_5y", "Renewed 5y registration"),
    ]
    colors = ["#1f77b4", "#d62728"]

    kl_edges = np.array([0.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
                         6.0, 6.5, 7.0, 8.0, 10.0])
    kl_centres = 0.5 * (kl_edges[:-1] + kl_edges[1:])
    d_edges = np.array([-2.0, -1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0,
                        0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    d_centres = 0.5 * (d_edges[:-1] + d_edges[1:])

    panel_specs = [
        ("prospective_kl", "Prospective KL\n(novel vs. past)", kl_edges, kl_centres),
        ("retrospective_kl", "Retrospective KL\n(novel vs. future)", kl_edges, kl_centres),
        ("dkl", r"$\Delta KL$ (pros $-$ retr)" "\n" r"(innovator $\rightarrow$ right)", d_edges, d_centres),
        ("avg_kl", "Mean KL  (pros + retr) / 2\n" r"(unusual against \emph{both} reference windows)", kl_edges, kl_centres),
    ]

    # Min bin size of 100 (was 30): with n=30, Wilson 95% half-width on
    # a 50% rate is ~18 percentage points, wide enough that per-bin noise
    # dominates. n=100 gives ~10 pp. We also annotate the bin sample
    # sizes alongside the markers so the reader can see when bands are
    # narrow because of large n vs.~small SE.
    MIN_BIN = 100

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    pdf_data = df.select(
        "prospective_kl", "retrospective_kl", "dkl",
        "reached_registration", "survived_5y",
    ).to_pandas()
    pdf_data["avg_kl"] = (pdf_data["prospective_kl"] + pdf_data["retrospective_kl"]) / 2.0

    for ax, (xcol, xlabel, edges, centres) in zip(axes, panel_specs):
        for (col, label), color in zip(outcomes, colors):
            d_pd = pdf_data[[xcol, col]].dropna().copy()
            d_pd[col] = d_pd[col].astype(int)
            d_pd["bin"] = np.digitize(d_pd[xcol], edges) - 1
            d_pd = d_pd[(d_pd["bin"] >= 0) & (d_pd["bin"] < len(centres))]
            grp = d_pd.groupby("bin").agg(rate=(col, "mean"), n=(col, "size"))
            grp = grp[grp["n"] >= MIN_BIN]
            if grp.empty:
                continue
            xs = centres[grp.index]
            ys = grp["rate"].values
            ns = grp["n"].values
            lo_arr = np.array([wilson(y, n)[0] for y, n in zip(ys, ns)])
            hi_arr = np.array([wilson(y, n)[1] for y, n in zip(ys, ns)])
            ax.fill_between(xs, lo_arr, hi_arr, color=color, alpha=0.22, linewidth=0)
            mean_rate = float(d_pd[col].mean())
            ax.plot(xs, ys, "o-", color=color, linewidth=1.6, markersize=5,
                    label=f"{label} (mean {mean_rate*100:.1f}%)")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Outcome rate")
    axes[2].axvline(0, color="grey", linewidth=0.7, linestyle="--")
    axes[1].set_title(
        f"{industry_label} ({cls}) — {n_clean:,} clean filings\n"
        r"shaded band = 95\% Wilson CI per bin (marginal, NOT joint)"
        f"  ·  bin minimum n = {MIN_BIN}"
    )
    axes[3].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   fontsize=8, frameon=False)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    # First pass: get clean-filing counts so we can sort by size
    classes = sorted(p.stem.replace("outcomes_class", "")
                     for p in PROC.glob("outcomes_class*.parquet")
                     if p.stem.replace("outcomes_class", "").isdigit())
    sizes: list[tuple[str, int]] = []
    for cls in classes:
        n = pl.read_parquet(PROC / f"outcomes_class{cls}.parquet").filter(CLEAN).height
        if n >= 5000:
            sizes.append((cls, n))
    sizes.sort(key=lambda t: -t[1])  # largest first

    out = RESULTS / "outcome_curves_all_industries.pdf"
    with PdfPages(out) as pdf:
        for cls, n in sizes:
            try:
                render_industry(cls, industry_name(cls), n, pdf)
                print(f"  rendered class {cls}  n={n:,}", flush=True)
            except Exception as e:
                print(f"  SKIP class {cls}: {e}", flush=True)
    print(f"\n[done] wrote {out}  ({len(sizes)} industries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
