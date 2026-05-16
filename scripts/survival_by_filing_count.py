"""Simple precursor: 5y survival rate by firm filing count.

Before any KL machinery, we want to know whether more-prolific firms have
higher per-filing survival.  This is the simplest possible firm-level
correlate of survival and belongs early in the paper.

Output:
  paper/results/survival_by_filing_count.png
  paper/results/survival_by_filing_count.txt
  paper/results/survival_by_filing_count.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

CLEAN_BASE = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
    & pl.col("prospective_kl").is_finite()
    & pl.col("retrospective_kl").is_finite()
    & pl.col("owner_name").is_not_null()
)


def load_filings(year_min: int, year_max: int) -> pl.DataFrame:
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        df = pl.read_parquet(path).filter(
            CLEAN_BASE
            & (pl.col("year") >= year_min)
            & (pl.col("year") <= year_max)
        ).with_columns(
            (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y"),
            pl.lit(cls).alias("nice_class"),
        ).select("owner_name", "nice_class", "year", "passed_5y", "reached_registration")
        parts.append(df)
    return pl.concat(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2023)
    args = ap.parse_args()

    print(f"[load] years {args.year_min}-{args.year_max}…", flush=True)
    df = load_filings(args.year_min, args.year_max)
    print(f"  {df.height:,} clean filings", flush=True)

    # Per-firm filing count (across the full window)
    firm = df.group_by("owner_name").agg(
        pl.len().alias("n_filings"),
        pl.col("passed_5y").mean().alias("firm_survival_rate"),
        pl.col("reached_registration").mean().alias("firm_reach_rate"),
    )
    print(f"  {firm.height:,} firms with >=1 clean filing", flush=True)

    # Firm-level: bucketize by n_filings and compute share of FIRMS with high survival
    # (We compute two views:
    #  view A — per-firm: distribution of firm_survival_rate by n_filings bucket
    #  view B — per-filing: pooled passed_5y by n_filings bucket of the owning firm)

    df = df.join(firm.select("owner_name", "n_filings"), on="owner_name", how="left")

    # Bucket boundaries — log-spaced, cover the long tail
    edges = [1, 2, 3, 5, 10, 25, 100, 1_000, 10_000, 1_000_000]
    edge_labels = ["1", "2", "3-4", "5-9", "10-24", "25-99", "100-999",
                   "1k-9.9k", "10k-999k", ">=1M"]

    def bucket_id(n):
        for i in range(len(edges) - 1):
            if edges[i] <= n < edges[i + 1]:
                return i
        return len(edges) - 2

    df_pd = df.select("n_filings", "passed_5y", "reached_registration").to_pandas()
    df_pd["bucket"] = df_pd["n_filings"].map(bucket_id)

    rows = []
    print("\n[per-filing] passed_5y by n_filings bucket:")
    print(f"  {'bucket':<12s}{'n_filings':>14s}{'n_firms':>11s}"
          f"{'pass_5y':>10s}{'reach_reg':>11s}{'95% CI':>20s}")
    for b, lbl in enumerate(edge_labels):
        sub = df_pd[df_pd["bucket"] == b]
        if sub.empty:
            continue
        n_filings = len(sub)
        n_firms = sub["n_filings"].drop_duplicates().count() if n_filings else 0
        # Better: distinct firms in this bucket
        n_firms_actual = (df.filter((pl.col("n_filings") >= edges[b]) &
                                    (pl.col("n_filings") < edges[b + 1]))
                          ["owner_name"].n_unique())
        p5 = sub["passed_5y"].mean()
        rr = sub["reached_registration"].mean()
        # Wilson interval for proportion (per-filing)
        from math import sqrt
        z = 1.96
        n = n_filings
        p = p5
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        ci = (max(0, center - half), min(1, center + half))
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        print(f"  {lbl:<12s}{n_filings:>14,}{n_firms_actual:>11,}"
              f"{p5:>10.3f}{rr:>11.3f}{ci_str:>20s}")
        rows.append((lbl, n_filings, n_firms_actual, p5, rr, ci[0], ci[1]))

    # Plot: bar chart of pass_5y by bucket
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [r[0] for r in rows]
    p5_vals = [r[3] for r in rows]
    reach_vals = [r[4] for r in rows]
    n_filings = [r[1] for r in rows]
    cis = [(r[5], r[6]) for r in rows]
    yerr = np.array([[v - lo for v, (lo, _) in zip(p5_vals, cis)],
                     [hi - v for v, (_, hi) in zip(p5_vals, cis)]])

    ax = axes[0]
    bars = ax.bar(range(len(rows)), p5_vals, yerr=yerr, capsize=3,
                  color="#2b6cb0", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Firm filing-count bucket (across clean window)")
    ax.set_ylabel("Per-filing pass_5y rate (registered & still live)")
    ax.set_ylim(0, max(p5_vals) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("(a) Survival by firm filing count")
    for i, (v, n) in enumerate(zip(p5_vals, n_filings)):
        ax.text(i, v + (yerr[1, i] if i < yerr.shape[1] else 0) + 0.005,
                f"n={n:,}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax2 = axes[1]
    ax2.bar(range(len(rows)), reach_vals, color="#888", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_xlabel("Firm filing-count bucket")
    ax2.set_ylabel("Per-filing reached_registration rate")
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title("(b) Reached registration by firm filing count (decomposes (a))")

    fig.suptitle(
        f"Survival rate as a function of how many trademarks the firm files "
        f"({df.height:,} filings; years {args.year_min}-{args.year_max})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = RESULTS / "survival_by_filing_count.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {out_png}", flush=True)

    # Text summary
    out_txt = RESULTS / "survival_by_filing_count.txt"
    with out_txt.open("w") as f:
        f.write("Survival by firm filing count\n")
        f.write(f"Years: {args.year_min}-{args.year_max}\n")
        f.write(f"Total clean filings: {df.height:,}\n")
        f.write(f"Total firms: {firm.height:,}\n\n")
        f.write(f"{'bucket':<12s}{'n_filings':>14s}{'n_firms':>11s}"
                f"{'pass_5y':>10s}{'reach_reg':>11s}{'95% CI lo':>12s}{'95% CI hi':>12s}\n")
        for r in rows:
            f.write(f"{r[0]:<12s}{r[1]:>14,}{r[2]:>11,}"
                    f"{r[3]:>10.3f}{r[4]:>11.3f}{r[5]:>12.3f}{r[6]:>12.3f}\n")
    print(f"[done] wrote {out_txt}", flush=True)

    # CSV
    out_csv = RESULTS / "survival_by_filing_count.csv"
    pl.DataFrame(rows, schema=["bucket", "n_filings", "n_firms", "pass_5y",
                               "reach_reg", "ci95_lo", "ci95_hi"],
                 orient="row").write_csv(out_csv)
    print(f"[done] wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
