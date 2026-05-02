"""Three-panel scatter: per-industry mean prospective KL, retrospective
KL, and dKL versus per-industry mean gross margin (from matched
publicly-traded firms in that industry).

Each industry is one dot, labelled with its NICE class number.

Output: paper/results/industry_vs_finance.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    CLEAN = (
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= 1995) & (pl.col("year") <= 2020)
    )

    cw = pl.read_parquet(REPO_ROOT / "data/processed/uspto_sec_crosswalk.parquet").select("owner_name", "cik")
    sec = pl.read_parquet(REPO_ROOT / "data/processed/sec_firm_year.parquet").filter(
        (pl.col("gross_margin").is_not_null())
        & (pl.col("gross_margin") > 0) & (pl.col("gross_margin") < 1.0)
        & (pl.col("revenue") > 0)
    ).select("cik", "fy", "gross_margin")

    rows = []
    for path in sorted((REPO_ROOT / "data/processed").glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        df = pl.read_parquet(path).filter(CLEAN)
        if df.is_empty():
            continue
        ind_pros = float(df["prospective_kl"].mean())
        ind_retr = float(df["retrospective_kl"].mean())
        ind_dkl = float(df["dkl"].mean())

        # firms in this industry that match SEC
        matched_owners = df.join(cw, on="owner_name", how="inner")["cik"].unique()
        if matched_owners.len() < 5:
            continue
        firm_panel = sec.filter(pl.col("cik").is_in(matched_owners))
        if firm_panel.height < 30:
            continue
        gm = float(firm_panel["gross_margin"].mean())
        rows.append({
            "cls": cls,
            "industry": industry_name(cls),
            "pros": ind_pros,
            "retr": ind_retr,
            "dkl": ind_dkl,
            "gross_margin": gm,
            "n_firms": int(matched_owners.len()),
            "n_firm_years": int(firm_panel.height),
        })
    df = pl.DataFrame(rows)
    print(f"[industry-vs-finance] {df.height} industries with >=5 matched firms")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    panels = [
        ("pros", "Prospective KL\n(industry mean — surprise at filing time)"),
        ("retr", "Retrospective KL\n(industry mean — unusual in hindsight = unimitated)"),
        ("dkl",  r"$\Delta KL$" + " (industry mean — innovator score)"),
    ]
    for ax, (col, xlabel) in zip(axes, panels):
        x = df[col].to_numpy()
        y = df["gross_margin"].to_numpy()
        sizes = (df["n_firms"].to_numpy() * 4).clip(20, 250)
        ax.scatter(x, y, s=sizes, alpha=0.55, color="#1f77b4", edgecolor="black", linewidth=0.4)
        # Pearson correlation
        rho = float(np.corrcoef(x, y)[0, 1])
        ax.set_title(f"{xlabel.splitlines()[0]}\n($\\rho = {rho:+.2f}$)")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        # Always label the user-requested classes; then top/bottom of x and y axes
        FORCE_LABEL = {"034", "026", "044", "038", "019", "032", "009", "005", "036"}
        labelled = set()
        rows_idx = {df.row(i, named=True)["cls"]: i for i in range(df.height)}
        for cls in FORCE_LABEL:
            i = rows_idx.get(cls)
            if i is None or i in labelled: continue
            labelled.add(i)
            row = df.row(i, named=True)
            short = row["industry"].split(" (")[0]
            ax.annotate(f"{short} ({cls})", (x[i], y[i]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color="#222",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        for i in list(np.argsort(-y)[:2]) + list(np.argsort(y)[:2]) + list(np.argsort(-x)[:2]) + list(np.argsort(x)[:2]):
            if i in labelled: continue
            labelled.add(i)
            row = df.row(int(i), named=True)
            short = row["industry"].split(" (")[0]
            ax.annotate(f"{short} ({row['cls']})", (x[i], y[i]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color="#222",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        # Linear fit line
        if len(x) >= 3:
            b, a = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, a + b * xx, color="#d62728", linewidth=1.0, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Industry mean gross margin\n(matched public firms)")
    fig.suptitle("Per-industry novelty vs.\\ financial outcome", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "industry_vs_finance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    df.write_parquet(REPO_ROOT / "data/processed/industry_vs_finance.parquet")
    print(f"[done] wrote industry_vs_finance.png + parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
