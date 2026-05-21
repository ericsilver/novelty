"""Re-parse all USPTO Trademark annual archives to extract owner state
and country. Single pass across all 91 parts; one row per serial_number.

This is the missing piece for deconfounding the surname-imputed
ethnicity result: post-2018 the trademark corpus is contaminated with
foreign-applicant filings (especially from China during the subsidy
period; see USPTO 2021 Trademark Public Advisory Committee report),
which the surname classifier inflates into the API ethnicity bucket.

Output:
  data/processed/owner_address.parquet
  columns: serial_number, owner_state (US 2-char, '' if not US),
           owner_country (2-char ISO)

Runtime: ~30-60 minutes wall time at ~12 GB total download.

Usage:
    python scripts/extract_owner_addresses.py            # full backfile
    python scripts/extract_owner_addresses.py --last 5   # just last 5 parts (smoke)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from novelty import uspto
from novelty.parse import iter_zip_xml

REPO = Path(__file__).resolve().parents[1]
DATA_RAW = REPO / "data" / "raw"
DATA_PROC = REPO / "data" / "processed"

SCHEMA = pa.schema([
    ("serial_number", pa.string()),
    ("filing_date", pa.string()),
    ("owner_state", pa.string()),
    ("owner_country", pa.string()),
    ("source_zip", pa.string()),
])


def main() -> int:
    load_dotenv(REPO / ".env")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--last", type=int, default=None,
                    help="only process the last N parts (most recent records)")
    ap.add_argument("--max-parts", type=int, default=None,
                    help="stop after N parts from the start (oldest first)")
    args = ap.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)

    parts = uspto.latest_backfile_files(uspto.PRODUCT_TRADEMARK_ANNUAL_APPLICATIONS)
    if args.last:
        parts = parts[-args.last:]
    if args.max_parts:
        parts = parts[: args.max_parts]
    total_size = sum(p.size_bytes for p in parts)
    print(f"[plan] {len(parts)} parts, {total_size/1e9:.2f} GB total",
          file=sys.stderr, flush=True)

    out_path = DATA_PROC / "owner_address.parquet"
    kept = 0
    t_total = time.time()
    with pq.ParquetWriter(out_path, SCHEMA, compression="zstd") as writer:
        for i, part in enumerate(parts, 1):
            dest = DATA_RAW / part.file_name
            print(f"[part {i:>3}/{len(parts)}] {part.file_name} "
                  f"({part.size_bytes/1e6:.0f} MB) downloading…",
                  file=sys.stderr, flush=True)
            t0 = time.time()
            uspto.stream_download(part.download_url, dest)
            rows = []
            seen = 0
            for cf in iter_zip_xml(dest):
                seen += 1
                if not cf.serial_number:
                    continue
                rows.append({
                    "serial_number": cf.serial_number,
                    "filing_date": cf.filing_date,
                    "owner_state": cf.owner_state or "",
                    "owner_country": cf.owner_country or "",
                    "source_zip": part.file_name,
                })
                # Flush every 200k rows so memory stays bounded
                if len(rows) >= 200_000:
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                    kept += len(rows)
                    rows.clear()
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                kept += len(rows)
            try:
                os.remove(dest)
            except OSError:
                pass
            print(f"           seen={seen:>9,}  kept={kept:>9,}  "
                  f"part={time.time()-t0:>5.1f}s  total={time.time()-t_total:>6.1f}s",
                  file=sys.stderr, flush=True)
            gc.collect()
    print(f"[done] {kept:,} records -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
