"""Derive trademark-survival outcomes from USPTO status codes and join with
per-filing surprise scores. Output: data/processed/outcomes_class<NNN>.parquet.

USPTO status code reference (collapsed buckets):
    6xx  — live (registered, in opposition, in maintenance)
    7xx  — dead during examination (never registered)
    8xx  — dead post-registration (was registered, later cancelled/expired)

We rely on the status code rather than the XML registration_date field,
because many pre-2003 records have a 6xx/8xx status code (and were therefore
registered at some point) but a missing registration_date.

Outcomes derived per filing:
    reached_registration   bool   status code starts with '6' or '8'
    currently_live         bool   status_code starts with '6'
    abandoned_at_exam      bool   status_code starts with '7'
    cancelled_postreg      bool   status_code starts with '8'
    survived_5y            bool   currently_live AND filed >= 5 years ago
    survived_10y           bool   currently_live AND filed >= 10 years ago
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]


def derive(records: pl.DataFrame, surprise: pl.DataFrame, *, today_year: int) -> pl.DataFrame:
    base = records.select(
        "serial_number",
        "filing_date",
        "registration_date",
        "status_code",
        "owner_name",
        "mark_identification",
    ).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"),
        pl.col("status_code").str.slice(0, 1).alias("status_bucket"),
    ).with_columns(
        (pl.col("status_bucket") == "6").alias("currently_live"),
        (pl.col("status_bucket") == "7").alias("abandoned_at_exam"),
        (pl.col("status_bucket") == "8").alias("cancelled_postreg"),
    ).with_columns(
        (
            (pl.col("status_bucket") == "6")
            | (pl.col("status_bucket") == "8")
        ).alias("reached_registration"),
    ).with_columns(
        (
            pl.col("currently_live")
            & ((today_year - pl.col("year")) >= 5)
        ).alias("survived_5y"),
        (
            pl.col("currently_live")
            & ((today_year - pl.col("year")) >= 10)
        ).alias("survived_10y"),
    )

    surprise_slim = surprise.select(
        "serial_number",
        "n_terms",
        "prospective_kl",
        "n_ref_prospective",
        "retrospective_kl",
        "n_ref_retrospective",
    ).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl")
    )

    return base.join(surprise_slim, on="serial_number", how="left")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--surprise", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--today-year", type=int, default=2026)
    args = ap.parse_args()

    rec = pl.read_parquet(args.records)
    sup = pl.read_parquet(args.surprise)
    out = derive(rec, sup, today_year=args.today_year)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    print(f"[done] {out.height:,} rows -> {args.out}", file=sys.stderr)
    eligible = out.filter(out["year"] <= args.today_year - 5).height
    surv = out.filter(out["year"] <= args.today_year - 5)["survived_5y"].mean()
    print(f"  5y-eligible: {eligible:,}; 5y survival rate: {surv*100:.1f}%", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
