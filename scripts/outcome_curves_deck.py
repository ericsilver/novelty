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
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
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


def build_debut_quintiles() -> pl.DataFrame:
    """Per owner_name (across all classes), the firm's first-ever filing's
    prospective KL and retrospective KL, each quintile-split 1..5
    (1 = lowest, 5 = highest)."""
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit(): continue
        parts.append(pl.read_parquet(path).filter(CLEAN & pl.col("owner_name").is_not_null())
                     .select("owner_name", "filing_date", "kl_vs_past", "kl_vs_future"))
    universe = pl.concat(parts).sort("filing_date")
    debut = universe.unique(subset=["owner_name"], keep="first").select(
        "owner_name", "kl_vs_past", "kl_vs_future"
    ).rename({"kl_vs_past": "debut_pros_kl", "kl_vs_future": "debut_retr_kl"})
    pdf = debut.to_pandas()
    pdf["debut_quintile"] = pd.qcut(pdf["debut_pros_kl"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    pdf["debut_retro_quintile"] = pd.qcut(pdf["debut_retr_kl"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return pl.from_pandas(pdf)


def render_industry(cls: str, industry_label: str, n_clean: int, pdf: PdfPages,
                    debut_q: pl.DataFrame) -> None:
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

    MIN_BIN = 100  # n=100 → Wilson 95% half-width ~10 pp at 50% rate

    pdf_data = df.join(debut_q.select("owner_name", "debut_quintile",
                                       "debut_retro_quintile"),
                       on="owner_name", how="left").select(
        "owner_name", "debut_quintile", "debut_retro_quintile",
        "kl_vs_past", "kl_vs_future", "dkl",
        "reached_registration", "survived_5y",
    ).to_pandas()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.0), sharey=True)

    # ----- Panel A: same outcome compared across pros vs retr KL axis -----
    # Red lines: outcome plotted against prospective KL.
    # Blue lines: outcome plotted against retrospective KL.
    # Solid: 5y survival.  Dotted: completed registration.
    # Bands at alpha 0.30 (70% transparent) and explicit zorder so they
    # do not over-paint each other.
    ax = axes[0]
    PROS_COLOR, RETRO_COLOR = "#d62728", "#1f77b4"
    series = [
        ("kl_vs_past",  "survived_5y",           PROS_COLOR,  "-",  2.0, 1, 2, "5y survival vs. prospective KL (vs. past)"),
        ("kl_vs_past",  "reached_registration",  PROS_COLOR,  ":",  2.0, 1, 2, "Registration vs. prospective KL (vs. past)"),
        ("kl_vs_future","survived_5y",           RETRO_COLOR, "-",  2.0, 1, 2, "5y survival vs. retrospective KL (vs. future)"),
        ("kl_vs_future","reached_registration",  RETRO_COLOR, ":",  2.0, 1, 2, "Registration vs. retrospective KL (vs. future)"),
    ]
    bands_to_draw = []
    for xcol, col, color, ls, lw, _z_band, _z_line, lbl in series:
        d_pd = pdf_data[[xcol, col]].dropna().copy()
        d_pd[col] = d_pd[col].astype(int)
        d_pd["bin"] = np.digitize(d_pd[xcol], kl_edges) - 1
        d_pd = d_pd[(d_pd["bin"] >= 0) & (d_pd["bin"] < len(kl_centres))]
        grp = d_pd.groupby("bin").agg(rate=(col, "mean"), n=(col, "size"))
        grp = grp[grp["n"] >= MIN_BIN]
        if grp.empty:
            continue
        xs = kl_centres[grp.index]
        ys = grp["rate"].values
        ns = grp["n"].values
        lo_arr = np.array([wilson(y, n)[0] for y, n in zip(ys, ns)])
        hi_arr = np.array([wilson(y, n)[1] for y, n in zip(ys, ns)])
        bands_to_draw.append((xs, lo_arr, hi_arr, color, ls, lw, lbl, ys))

    # Draw all bands first (zorder 1), then all lines (zorder 3) so the
    # lines always sit on top regardless of plot order.
    for xs, lo_arr, hi_arr, color, _, _, _, _ in bands_to_draw:
        ax.fill_between(xs, lo_arr, hi_arr, color=color, alpha=0.30, linewidth=0, zorder=1)
    for xs, _, _, color, ls, lw, lbl, ys in bands_to_draw:
        ax.plot(xs, ys, marker="o", linestyle=ls, color=color,
                linewidth=lw, markersize=4, label=lbl, zorder=3)
    ax.set_xlabel("KL value (prospective/vs.-past in red, retrospective/vs.-future in blue)")
    ax.set_ylabel("Outcome rate")
    ax.grid(alpha=0.3, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.0), fontsize=7, frameon=False)
    ax.set_title("Outcomes by prospective (vs. past) and retrospective (vs. future) KL")

    # ----- Panel B: 5y survival by dKL, split by debut PROSPECTIVE KL quintile -----
    quintile_colors = ["#7a1111", "#bf6b3a", "#9a9a9a", "#3a8bbf", "#117a3a"]
    pros_q_labels = {1: "Q1 unsurprising debut (low pros)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 surprising debut (high pros)"}
    _draw_quintile_panel(axes[1], pdf_data, "debut_quintile", pros_q_labels,
                         quintile_colors, d_edges, d_centres, MIN_BIN,
                         "5y survival by $\\Delta KL$, split by firm's\n"
                         "debut-filing PROSPECTIVE-KL quintile")

    # ----- Panel C: 5y survival by dKL, split by debut RETROSPECTIVE KL quintile -----
    retro_q_labels = {1: "Q1 future-resembled debut (low retr)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 future-unlike debut (high retr)"}
    _draw_quintile_panel(axes[2], pdf_data, "debut_retro_quintile", retro_q_labels,
                         quintile_colors, d_edges, d_centres, MIN_BIN,
                         "5y survival by $\\Delta KL$, split by firm's\n"
                         "debut-filing RETROSPECTIVE-KL quintile")

    fig.suptitle(
        f"{industry_label} ({cls}) — {n_clean:,} clean filings  ·  "
        f"shaded band = 95\\% Wilson CI per bin (marginal)  ·  bin minimum n = {MIN_BIN}",
        y=1.04, fontsize=10,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_quintile_panel(ax, pdf_data, qcol, qlabels, qcolors, edges, centres, min_bin, title):
    col = "survived_5y"
    for q in (1, 2, 3, 4, 5):
        d_pd = pdf_data[pdf_data[qcol] == q][["dkl", col]].dropna().copy()
        if len(d_pd) < min_bin: continue
        d_pd[col] = d_pd[col].astype(int)
        d_pd["bin"] = np.digitize(d_pd["dkl"], edges) - 1
        d_pd = d_pd[(d_pd["bin"] >= 0) & (d_pd["bin"] < len(centres))]
        grp = d_pd.groupby("bin").agg(rate=(col, "mean"), n=(col, "size"))
        grp = grp[grp["n"] >= min_bin]
        if grp.empty: continue
        xs = centres[grp.index]
        ys = grp["rate"].values
        ns = grp["n"].values
        lo_arr = np.array([wilson(y, n)[0] for y, n in zip(ys, ns)])
        hi_arr = np.array([wilson(y, n)[1] for y, n in zip(ys, ns)])
        ax.fill_between(xs, lo_arr, hi_arr, color=qcolors[q-1], alpha=0.25, linewidth=0)
        ax.plot(xs, ys, "o-", color=qcolors[q-1], linewidth=1.4, markersize=4,
                label=qlabels[q])
    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"$\Delta KL$ (pros $-$ retr)  (innovator $\rightarrow$ right)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.0), fontsize=7, frameon=False)
    ax.set_title(title)


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

    print("[debut] computing per-firm debut prospective-KL quintiles…", flush=True)
    debut_q = build_debut_quintiles()
    print(f"  {debut_q.height:,} firms quintile-split on debut prospective KL", flush=True)
    print(f"  per-quintile size: {debut_q.group_by('debut_quintile').len().sort('debut_quintile')['len'].to_list()}")

    out = RESULTS / "outcome_curves_all_industries.pdf"
    with PdfPages(out) as pdf:
        for cls, n in sizes:
            try:
                render_industry(cls, industry_name(cls), n, pdf, debut_q)
                print(f"  rendered class {cls}  n={n:,}", flush=True)
            except Exception as e:
                print(f"  SKIP class {cls}: {e}", flush=True)
    print(f"\n[done] wrote {out}  ({len(sizes)} industries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
