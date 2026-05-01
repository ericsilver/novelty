"""Aggregate per-filing surprise scores into a (registrant × year) table.

For each (owner_name, filing_year):
    n_filings           int     count of filings
    n_filings_clean     int     count with both 5y windows populated and n_terms>=3
    mean_dkl            float   mean prospective_kl - retrospective_kl  (clean)
    max_dkl             float   max  dkl                                (clean)
    p90_dkl             float   90th-percentile dkl                     (clean)
    n_high_dkl          int     count of clean filings with dkl >= threshold (default 0.5)
    share_high_dkl      float   n_high_dkl / n_filings_clean
    mean_pros_kl        float   mean prospective KL                     (clean)
    mean_retr_kl        float   mean retrospective KL                   (clean)

Owner names are taken verbatim from USPTO XML — entity-resolution to financial
identifiers (Compustat / CRSP) is a downstream concern.

Usage:
    python -m novelty.firm_year \\
        --surprise data/processed/surprise_class032.parquet \\
        --out      data/processed/firm_year_class032.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]


def aggregate(surprise: pl.DataFrame, *, high_dkl: float = 0.5) -> pl.DataFrame:
    df = surprise.with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
        (
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
        ).alias("clean"),
    ).filter(pl.col("owner_name").is_not_null() & (pl.col("year").is_not_null()))

    return (
        df.group_by(["owner_name", "year"])
        .agg(
            pl.len().alias("n_filings"),
            pl.col("clean").sum().alias("n_filings_clean"),
            pl.col("dkl").filter(pl.col("clean")).mean().alias("mean_dkl"),
            pl.col("dkl").filter(pl.col("clean")).max().alias("max_dkl"),
            pl.col("dkl").filter(pl.col("clean")).quantile(0.9).alias("p90_dkl"),
            (pl.col("clean") & (pl.col("dkl") >= high_dkl)).sum().alias("n_high_dkl"),
            pl.col("prospective_kl").filter(pl.col("clean")).mean().alias("mean_pros_kl"),
            pl.col("retrospective_kl").filter(pl.col("clean")).mean().alias("mean_retr_kl"),
            pl.col("n_terms").filter(pl.col("clean")).sum().alias("n_terms_total"),
        )
        .with_columns(
            (pl.col("n_high_dkl") / pl.col("n_filings_clean").cast(pl.Float64)).alias("share_high_dkl"),
        )
        .sort(["year", "owner_name"])
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surprise", type=Path, required=True, help="per-filing surprise parquet")
    ap.add_argument("--out", type=Path, required=True, help="output firm-year parquet")
    ap.add_argument("--high-dkl", type=float, default=0.5, help="threshold for 'high innovation' filings")
    args = ap.parse_args()

    df = pl.read_parquet(args.surprise)
    print(f"[in ] {args.surprise}: {df.height:,} filings", file=sys.stderr)
    fy = aggregate(df, high_dkl=args.high_dkl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fy.write_parquet(args.out)
    print(
        f"[out] {fy.height:,} firm-years -> {args.out}  "
        f"(unique firms: {fy['owner_name'].n_unique():,})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
