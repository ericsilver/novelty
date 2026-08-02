"""Templated G/S strings: how often the SAME normalised goods/services
text appears across the corpus, and whether multiplicity predicts
survival.

Memory-conservative re-implementation of Part 2 of
diagonal_and_templated.py:

* keep everything in polars (no big pandas frames)
* compute per-string counts in one pass, write to parquet
* join the count back into per-filing rows (drop the long text)
* aggregate by count-multiplicity bin in polars
* convert only the small bin/top tables to pandas at the end

Output:
  paper/results/templated_survival.json  (tables + top-50)
  paper/results/templated_survival.png   (bar chart by multiplicity)
  data/processed/gs_template_counts.parquet  (gs_norm, count)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

# filing_year cap of 2018 so the 5y Section-8 window has been adjudicated
# for nearly every registration (~filing_year + 7 ≤ 2026).
CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1990) & (pl.col("year") <= 2018)
)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return p, p
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def _normalise(col: str = "goods_services") -> pl.Expr:
    return (pl.col(col).str.to_lowercase()
            .str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars())


def main() -> int:
    print("[load] outcomes (clean) across all classes…", flush=True)
    out_parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        out_parts.append(pl.read_parquet(path).filter(CLEAN)
            .with_columns(
                (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
            ).select("serial_number", "reached_registration", "passed_5y"))
    out_all = pl.concat(out_parts).unique(subset=["serial_number"])
    n_clean = out_all.height
    print(f"  clean filings: {n_clean:,}", flush=True)

    print("[load+norm] goods_services per class, write per-class parquet…", flush=True)
    norm_parts = []
    for path in sorted(PROC.glob("tm_class*.parquet")):
        cls = path.stem.replace("tm_class", "")
        if not cls.isdigit():
            continue
        df = (pl.read_parquet(path).select("serial_number", "goods_services")
              .filter(pl.col("goods_services").is_not_null())
              .with_columns(_normalise().alias("gs_norm"))
              .filter(pl.col("gs_norm").str.len_chars() >= 5)
              .select("serial_number", "gs_norm"))
        norm_parts.append(df)
        print(f"  class {cls}: {df.height:,} filings normalised", flush=True)
    gs_all = pl.concat(norm_parts).unique(subset=["serial_number"])
    print(f"  total filings with G/S: {gs_all.height:,}", flush=True)
    del norm_parts

    print("[count] per-string multiplicity…", flush=True)
    counts = (gs_all.group_by("gs_norm")
              .agg(pl.len().alias("count"))
              .sort("count", descending=True))
    counts.write_parquet(PROC / "gs_template_counts.parquet")
    print(f"  unique normalised G/S strings: {counts.height:,}", flush=True)
    print(f"  median count: {counts['count'].median()}, p95: {counts['count'].quantile(0.95)}, max: {counts['count'].max()}")

    # Top-50 templates with their outcomes
    print("[top50] joining top-50 templates with outcomes for printout…", flush=True)
    top50 = counts.head(50).join(
        gs_all.join(out_all, on="serial_number", how="inner"),
        on="gs_norm", how="inner"
    ).group_by("gs_norm").agg(
        pl.len().alias("n_clean"),
        pl.col("count").first().alias("count_total"),
        pl.col("reached_registration").mean().alias("reach"),
        pl.col("passed_5y").mean().alias("surv5"),
    ).sort("count_total", descending=True).to_pandas()

    print("\nTop-50 most-templated normalised G/S strings:")
    print(f"{'count':>8} {'reg%':>5} {'surv5%':>6}  | text")
    for _, row in top50.iterrows():
        snippet = row["gs_norm"][:120].replace("\n", " ")
        print(f"  {int(row['count_total']):>7} {row['reach']*100:5.1f} {row['surv5']*100:5.1f}  | {snippet}")

    # Filing-level aggregation by count-bin, all polars
    print("\n[bin] filing-level outcomes by template-multiplicity bin…", flush=True)
    filings = (gs_all.join(out_all, on="serial_number", how="inner")
               .join(counts, on="gs_norm", how="left")
               .select("count", "reached_registration", "passed_5y"))
    print(f"  filings tagged with count: {filings.height:,}")

    # Bin labels
    breaks = [(1, 1, "1 (unique)"),
              (2, 2, "2"),
              (3, 5, "3-5"),
              (6, 10, "6-10"),
              (11, 50, "11-50"),
              (51, 200, "51-200"),
              (201, 1000, "201-1000"),
              (1001, 5000, "1001-5000"),
              (5001, 10**9, "5000+")]

    rows = []
    for lo, hi, lbl in breaks:
        sub = filings.filter((pl.col("count") >= lo) & (pl.col("count") <= hi))
        n = sub.height
        if n == 0:
            continue
        reach = sub["reached_registration"].mean()
        surv5 = sub["passed_5y"].mean()
        reach_lo, reach_hi = wilson(reach, n)
        surv5_lo, surv5_hi = wilson(surv5, n)
        rows.append(dict(
            label=lbl, n=n, reach=reach, surv5=surv5,
            reach_lo=reach_lo, reach_hi=reach_hi,
            surv5_lo=surv5_lo, surv5_hi=surv5_hi,
        ))
        print(f"  {lbl:<14} n={n:>10,}  reg={reach*100:5.1f}%  surv5={surv5*100:5.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = np.arange(len(rows))
    surv = np.array([r["surv5"] for r in rows]) * 100
    reach = np.array([r["reach"] for r in rows]) * 100
    surv_err_lo = surv - np.array([r["surv5_lo"] for r in rows]) * 100
    surv_err_hi = np.array([r["surv5_hi"] for r in rows]) * 100 - surv
    reach_err_lo = reach - np.array([r["reach_lo"] for r in rows]) * 100
    reach_err_hi = np.array([r["reach_hi"] for r in rows]) * 100 - reach
    width = 0.4
    ax.bar(xs - width/2, reach, width, yerr=[reach_err_lo, reach_err_hi],
           color="#d62728", alpha=0.85, label="Completed registration", capsize=3)
    ax.bar(xs + width/2, surv, width, yerr=[surv_err_lo, surv_err_hi],
           color="#1f77b4", alpha=0.85, label="Renewed 5y", capsize=3)
    for i, r in enumerate(rows):
        ax.text(i, max(r["reach"], r["surv5"]) * 100 + 1.5,
                f"n={r['n']:,}", ha="center", va="bottom", fontsize=8, rotation=20)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("Outcome rate (%)")
    ax.set_xlabel("Filings sharing the exact (normalised) G/S text")
    ax.set_title("Templated trademark descriptions: registration & still-live rate\n"
                 "(still-live = registered \\& not cancelled-post-reg, "
                 f"filing years 1990--2018; n={n_clean:,} clean filings)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png = RESULTS / "templated_survival.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {out_png}")

    out_json = RESULTS / "templated_survival.json"
    out_json.write_text(json.dumps({
        "n_clean_filings": n_clean,
        "n_unique_norm_strings": int(counts.height),
        "by_count_bin": rows,
        "top50": top50.to_dict(orient="records"),
    }, indent=2, default=str))
    print(f"[done] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
