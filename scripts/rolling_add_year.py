"""Backfill the `year` column onto the rolling surprise files.

The production topic files carry serial_number, year, and the three scores;
the rolling rescorer wrote everything except `year`, which topic_debut.py and
topic_outcomes_all.py select on. Rather than re-run a 40-minute rescore, take
the filing year from the class file and join it on.

Idempotent: files that already carry `year` are left alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> int:
    files = sorted(PROC.glob("rolling_surprise_class*.parquet"))
    if not files:
        print("no rolling files found", file=sys.stderr)
        return 1
    done = skipped = 0
    for f in files:
        d = pl.read_parquet(f)
        if "year" in d.columns:
            skipped += 1
            continue
        cls = f.stem.replace("rolling_surprise_class", "")[:3]
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                             columns=["serial_number", "filing_date"]).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
        ).select("serial_number", "year").unique(subset="serial_number", keep="first")
        out = d.join(tm, on="serial_number", how="left")
        out = out.select(["serial_number", "year"]
                         + [c for c in out.columns if c not in ("serial_number", "year")])
        out.write_parquet(f)
        done += 1
        print(f"  [{f.name}] year added ({out['year'].null_count():,} null)",
              file=sys.stderr, flush=True)
    print(f"[done] {done} updated, {skipped} already had year", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
