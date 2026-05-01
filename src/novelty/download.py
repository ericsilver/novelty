"""Stream the USPTO Trademark Annual Applications backfile (product TRTYRAP),
filter to one NICE class (default Class 32 = beers/non-alcoholic beverages),
and write a slim Parquet of matching records.

The annual product is published as ~91 ZIP parts of the full 1884–latest
backfile (~12 GB total). We process one part at a time: download → stream-parse
XML → filter → append → delete the ZIP. Peak disk use stays ~300 MB.

Usage:
    python -m novelty.download --nice 032 --max-parts 1   # smoke test, 1 part
    python -m novelty.download --nice 032                  # full sweep
    python -m novelty.download --nice 032 --year 2023      # also filter year
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from . import uspto
from .parse import iter_zip_xml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

PARQUET_SCHEMA = pa.schema(
    [
        ("serial_number", pa.string()),
        ("filing_date", pa.string()),
        ("registration_date", pa.string()),
        ("status_code", pa.string()),
        ("mark_identification", pa.string()),
        ("nice_classes", pa.list_(pa.string())),
        ("us_classes", pa.list_(pa.string())),
        ("owner_name", pa.string()),
        ("goods_services", pa.string()),
        ("source_zip", pa.string()),
    ]
)


def _process_part(
    part: uspto.FileEntry,
    nice: str,
    year_prefix: str | None,
    writer: pq.ParquetWriter,
) -> int:
    dest = DATA_RAW / part.file_name
    t0 = time.time()
    print(f"[get ] {part.file_name} ({part.size_bytes/1e6:.1f} MB)…", file=sys.stderr, flush=True)
    uspto.stream_download(part.download_url, dest)
    print(f"       downloaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    rows: list[dict] = []
    seen = 0
    for cf in iter_zip_xml(dest):
        seen += 1
        if nice not in cf.nice_classes:
            continue
        if year_prefix and (cf.filing_date or "")[:4] != year_prefix:
            continue
        rows.append(
            {
                "serial_number": cf.serial_number,
                "filing_date": cf.filing_date,
                "registration_date": cf.registration_date,
                "status_code": cf.status_code,
                "mark_identification": cf.mark_identification,
                "nice_classes": cf.nice_classes,
                "us_classes": cf.us_classes,
                "owner_name": cf.owner_name,
                "goods_services": cf.goods_services,
                "source_zip": part.file_name,
            }
        )

    if rows:
        tbl = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
        writer.write_table(tbl)
    print(
        f"       parsed {seen} records, kept {len(rows)} matching nice={nice}"
        + (f" year={year_prefix}" if year_prefix else "")
        + f"  ({time.time()-t0:.1f}s total)",
        file=sys.stderr,
        flush=True,
    )

    try:
        os.remove(dest)
    except OSError:
        pass
    return len(rows)


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nice", default="032", help="NICE class as 3-digit string (default 032)")
    ap.add_argument("--year", default=None, help="optional filing-year filter (e.g. 2023)")
    ap.add_argument("--max-parts", type=int, default=None, help="stop after N parts from the start (oldest)")
    ap.add_argument("--last-parts", type=int, default=None, help="only process the last N parts (most recent records)")
    args = ap.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    parts = uspto.latest_backfile_files(uspto.PRODUCT_TRADEMARK_ANNUAL_APPLICATIONS)
    if args.max_parts:
        parts = parts[: args.max_parts]
    if args.last_parts:
        parts = parts[-args.last_parts :]
    total_size = sum(p.size_bytes for p in parts)
    print(
        f"[plan] {len(parts)} parts, {total_size/1e9:.2f} GB total. "
        f"NICE={args.nice}  year={args.year or 'ALL'}",
        file=sys.stderr,
    )

    suffix = f"_{args.year}" if args.year else ""
    smoke = ""
    if args.max_parts:
        smoke = f"_first{args.max_parts}"
    elif args.last_parts:
        smoke = f"_last{args.last_parts}"
    out_path = DATA_PROCESSED / f"tm_class{args.nice}{suffix}{smoke}.parquet"

    kept = 0
    with pq.ParquetWriter(out_path, PARQUET_SCHEMA, compression="zstd") as writer:
        for i, part in enumerate(parts, 1):
            print(f"[part] {i}/{len(parts)}  {part.file_name}", file=sys.stderr)
            kept += _process_part(part, args.nice, args.year, writer)

    print(f"[done] {kept} rows -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
