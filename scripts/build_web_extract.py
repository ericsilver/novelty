"""Build the query extract the browser tool reads: Parquet, partitioned and sorted.

The explorer pages are static because they answer questions decided in advance.
Searching by company, filtering by year, or crossing the two is not decidable in
advance, so it needs a query engine -- and a browser can be one, if the data is
laid out so that a query touches a small part of it.

Two layout choices do all the work, and both are about letting the reader's
browser skip bytes rather than download them:

  partition by filing year   A year filter then reads only the files it needs.
                             Thirty small files instead of one large one.
  sort by normalized owner   Parquet stores per-row-group min/max for every
                             column. Sorted by owner, a name lookup compares
                             against those statistics and skips almost every
                             row group without fetching it. Unsorted, the same
                             query reads the file end to end. This single
                             decision is the difference between a fast tool and
                             an unusable one.

Rows are every filing in the corpus, not only the scored ones, because a tool
that silently omits two-thirds of the record is worse than no tool. Score
columns are null where the reference windows could not be filled.

Usage:  python scripts/build_web_extract.py [OUTDIR]
Output: docs/data/filings/year=YYYY/part.parquet  plus a manifest
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "data" / "filings"

CLASSES = [f"{i:03d}" for i in range(1, 46)]
YEAR_LO, YEAR_HI = 1985, 2025
GATE_LO, GATE_HI = 4.0, 8.5
SUFFIXES = (r"\b(INC|INCORPORATED|LLC|L\.L\.C|LTD|LIMITED|CORP|CORPORATION|"
            r"COMPANY|CO|LP|LLP|PLC|GMBH|SA|NV|BV|AB|AG|AS|OY|SPA|SRL|PTY|"
            r"KK|KABUSHIKI KAISHA)\b")


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def norm_owner(col: str) -> pl.Expr:
    return (pl.col(col).str.to_uppercase()
            .str.replace_all(r"[^A-Z0-9 ]", " ")
            .str.replace_all(SUFFIXES, "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars())


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("gd")
    ).drop_nulls("gd").group_by("serial_number").agg(pl.col("gd").min())
    log(f"[setup] {ev.height:,} gate cancellations")

    parts = []
    for c in CLASSES:
        tp = PROC / f"tm_class{c}.parquet"
        if not tp.exists():
            continue
        d = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                         "registration_date", "owner_name",
                                         "mark_identification"]).filter(
            pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        sp = PROC / f"rolling_surprise_class{c}.parquet"
        if sp.exists():
            s = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                             "topic_kl_vs_future"])
            d = d.join(s, on="serial_number", how="left")
            del s
        else:
            d = d.with_columns(pl.lit(None, pl.Float64).alias("topic_kl_vs_past"),
                               pl.lit(None, pl.Float64).alias("topic_kl_vs_future"))
        parts.append(d.with_columns(pl.lit(c).alias("nice_class")))
        del d
        gc.collect()
    df = pl.concat(parts)
    del parts
    gc.collect()
    log(f"[rows] {df.height:,} filings across {df['nice_class'].n_unique()} classes")

    df = df.with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("filing_year"),
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        norm_owner("owner_name").alias("owner_norm"),
    ).filter(pl.col("filing_year").is_between(YEAR_LO, YEAR_HI))

    df = df.join(ev, on="serial_number", how="left").with_columns(
        pl.col("rd").dt.year().cast(pl.Int32).alias("registration_year"),
        pl.col("rd").is_not_null().alias("registered"),
        (((pl.col("gd") - pl.col("rd")).dt.total_days() / 365.25)
         .is_between(GATE_LO, GATE_HI, closed="left")).alias("gate_failed"),
        (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("lead"),
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("atypicality"),
    ).select(
        "serial_number", "owner_name", "owner_norm", "mark_identification",
        "nice_class", "filing_year", "filing_date", "registration_year",
        "registered", "gate_failed",
        pl.col("topic_kl_vs_past").alias("kl_past"),
        pl.col("topic_kl_vs_future").alias("kl_future"),
        "lead", "atypicality")
    del ev
    gc.collect()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"rows": int(df.height), "partitions": [],
                "columns": df.columns,
                "sorted_by": "owner_norm within each filing_year",
                "note": "Score columns are null where the reference windows "
                        "could not be filled (corpus edges and thin references)."}
    for y in range(YEAR_LO, YEAR_HI + 1):
        part = df.filter(pl.col("filing_year") == y).sort("owner_norm")
        if part.height == 0:
            continue
        d = OUT / f"year={y}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "part.parquet"
        part.write_parquet(p, compression="zstd", compression_level=9,
                           row_group_size=20_000, statistics=True)
        mb = p.stat().st_size / 1e6
        manifest["partitions"].append(
            {"year": y, "rows": int(part.height), "mb": round(mb, 2),
             "path": f"year={y}/part.parquet"})
        log(f"  {y}: {part.height:>9,} rows  {mb:>7.2f} MB")
        del part
        gc.collect()

    total = sum(p["mb"] for p in manifest["partitions"])
    manifest["total_mb"] = round(total, 1)
    manifest["max_partition_mb"] = round(
        max(p["mb"] for p in manifest["partitions"]), 2)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"\n[done] {manifest['rows']:,} rows, {total:.0f} MB across "
        f"{len(manifest['partitions'])} partitions, "
        f"largest {manifest['max_partition_mb']:.1f} MB "
        f"(GitHub's per-file limit is 100 MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
