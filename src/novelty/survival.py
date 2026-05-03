"""Derive trademark-survival outcomes from USPTO status codes and join with
per-filing surprise scores. Output: data/processed/outcomes_class<NNN>.parquet.

USPTO status code reference (collapsed buckets and key sub-codes):
    6xx  — live (registered, in opposition, in maintenance)
    7xx  — dead during examination (never registered)
    8xx  — dead post-registration (was registered, later cancelled/expired)
    686  — dead, cancelled because Section 8 (5-year maintenance) not filed
    688  — dead, cancelled because Section 9 (10-year renewal) not filed

We rely on the status code rather than the XML registration_date field,
because many pre-2003 records have a 6xx/8xx status code (and were therefore
registered at some point) but a missing registration_date.

Outcomes derived per filing:
    reached_registration   bool   status starts with '6' or '8' (ever registered)
    currently_live         bool   status starts with '6'
    abandoned_at_exam      bool   status starts with '7'
    cancelled_postreg      bool   status starts with '8'
    cancelled_section8     bool   status == '686' (failed 5y gate)
    cancelled_section9     bool   status == '688' (failed 10y gate)
    survived_5y            bool   reached_registration AND age >= 5 AND
                                  NOT cancelled_section8.  ``Made it past
                                  the 5-year Section-8 maintenance gate.''
    survived_10y           bool   survived_5y AND age >= 10 AND
                                  NOT cancelled_section9.  Strict subset of
                                  survived_5y within any individual filing.
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
        (pl.col("status_code") == "686").alias("cancelled_section8"),
        (pl.col("status_code") == "688").alias("cancelled_section9"),
    ).with_columns(
        (
            (pl.col("status_bucket") == "6")
            | (pl.col("status_bucket") == "8")
        ).alias("reached_registration"),
    ).with_columns(
        # Made it past the 5-year Section-8 gate: reached registration, is
        # at least 5 years old, AND was not cancelled specifically for
        # failing Section 8 (status 686).
        (
            (
                (pl.col("status_bucket") == "6")
                | (pl.col("status_bucket") == "8")
            )
            & ((today_year - pl.col("year")) >= 5)
            & (pl.col("status_code") != "686")
        ).alias("survived_5y"),
    ).with_columns(
        # Made it past the 10-year Section-9 gate: survived_5y AND at
        # least 10 years old AND not cancelled for Section 9 (status 688).
        # Strict subset of survived_5y within a single filing.
        (
            pl.col("survived_5y")
            & ((today_year - pl.col("year")) >= 10)
            & (pl.col("status_code") != "688")
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
