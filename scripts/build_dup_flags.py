"""Cache exact-duplicate-text flags for every filing, per class.

Same construction as lowA_forensics.py (normalized description text hashed;
per-class cluster size, distinct owners, earliest filer) but kept for every
filing 1995-2018 and persisted slim, so downstream analyses can join a
boilerplate/copy indicator without re-hashing the corpus.

Output: paper/results/dup_flags/{NNN}.parquet
        (serial_number, dupN, dup_owners, prior_other_owner)
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "paper" / "results" / "dup_flags"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
FY_LO, FY_HI = 1995, 2018


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        tp = PROC / f"tm_class{c}.parquet"
        if not tp.exists():
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                          "owner_name", "goods_services"]).filter(
            pl.col("filing_date").fill_null("").str.len_chars() >= 8).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
            pl.col("goods_services").fill_null("").str.to_lowercase()
              .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ")
              .str.strip_chars().alias("norm"),
            pl.col("owner_name").fill_null("").str.to_uppercase()
              .str.replace_all(r"[^A-Z0-9 ]", "").str.replace_all(r"\s+", " ")
              .str.strip_chars().alias("own"))
        tm = tm.filter(pl.col("fy").is_between(FY_LO, FY_HI)
                       & (pl.col("norm").str.len_chars() > 0)).with_columns(
            pl.col("norm").hash(seed=7).alias("h")).drop("norm")
        cl = tm.group_by("h").agg(pl.len().alias("dupN"),
                                  pl.col("own").n_unique().alias("dup_owners"),
                                  pl.col("filing_date").min().alias("first_fd"),
                                  pl.col("own").sort_by("filing_date").first().alias("first_own"))
        tm = tm.join(cl, on="h", how="left").with_columns(
            ((pl.col("dupN") > 1) & (pl.col("filing_date") > pl.col("first_fd"))
             & (pl.col("own") != pl.col("first_own"))).alias("prior_other_owner"))
        tm.select("serial_number", "dupN", "dup_owners", "prior_other_owner").write_parquet(
            OUT / f"{c}.parquet")
        log(f"[{c}] {tm.height:,} rows")
        del tm, cl
        gc.collect()
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
