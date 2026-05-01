"""Download the USPTO trademark backfile in one streaming pass and write a
slim Parquet for EVERY NICE class (001..045) in parallel. A filing that
declares multiple NICE classes is written once into each of its classes'
parquets.

Output: data/processed/tm_class<NNN>.parquet for each class with at least
one matching record.

Usage:
    python scripts/download_all_classes.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty import uspto
    from novelty.parse import iter_zip_xml
    from novelty.industries import all_classes

    target = set(all_classes())

    schema = pa.schema(
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

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Open writers lazily per class
    writers: dict[str, pq.ParquetWriter] = {}
    counts: dict[str, int] = defaultdict(int)

    parts = uspto.latest_backfile_files(uspto.PRODUCT_TRADEMARK_ANNUAL_APPLICATIONS)
    print(f"[plan] {len(parts)} parts, "
          f"{sum(p.size_bytes for p in parts)/1e9:.2f} GB total, "
          f"writing {len(target)} per-class parquets",
          file=sys.stderr)

    for i, part in enumerate(parts, 1):
        zp = DATA_RAW / part.file_name
        t0 = time.time()
        print(f"[part {i:>2}/{len(parts)}] {part.file_name} ({part.size_bytes/1e6:.1f} MB) downloading…", file=sys.stderr, flush=True)
        uspto.stream_download(part.download_url, zp)

        per_class_rows: dict[str, list[dict]] = defaultdict(list)
        seen = 0
        for cf in iter_zip_xml(zp):
            seen += 1
            classes_in_filing = set(cf.nice_classes) & target
            if not classes_in_filing:
                continue
            row = {
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
            for c in classes_in_filing:
                per_class_rows[c].append(row)

        # Flush each class's batch to its writer
        for cls, rows in per_class_rows.items():
            if not rows:
                continue
            if cls not in writers:
                path = DATA_PROCESSED / f"tm_class{cls}.parquet"
                writers[cls] = pq.ParquetWriter(path, schema, compression="zstd")
            writers[cls].write_table(pa.Table.from_pylist(rows, schema=schema))
            counts[cls] += len(rows)

        try:
            os.remove(zp)
        except OSError:
            pass
        elapsed = time.time() - t0
        total_kept = sum(len(r) for r in per_class_rows.values())
        print(f"            parsed {seen} records, kept {total_kept} class-rows ({elapsed:.1f}s)", file=sys.stderr, flush=True)

    for w in writers.values():
        w.close()

    print(f"\n[done] wrote {len(writers)} per-class parquets:", file=sys.stderr)
    for cls in sorted(counts):
        print(f"  class {cls}: {counts[cls]:>9,} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
