"""Step 2: full TRTYRAP re-parse for prosecution events, attorney,
filing basis, and death dates.

Phase 1: download every backfile zip to data/raw/ (skips existing).
Phase 2: stream-parse every zip into:
  data/processed/case_events.parquet   serial_number, code, type, date, seq
  data/processed/case_extras.parquet   serial_number + attorney_name,
      abandonment_date, status_date, basis flags, published-for-opposition,
      section-8 flags, renewal flag
  data/processed/event_code_dict.parquet  code -> modal description

Restartable: zips already downloaded are kept; parse phase runs over all
zips in one pass and writes chunked parquet.
"""
from __future__ import annotations

import collections
import gc
import io
import sys
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"

EV_SCHEMA = pa.schema([
    ("serial_number", pa.string()),
    ("code", pa.string()),
    ("etype", pa.string()),
    ("date", pa.int32()),
    ("seq", pa.int32()),
])
EX_SCHEMA = pa.schema([
    ("serial_number", pa.string()),
    ("attorney_name", pa.string()),
    ("abandonment_date", pa.string()),
    ("status_date", pa.string()),
    ("published_for_opposition_date", pa.string()),
    ("intent_to_use_in", pa.string()),
    ("use_application_currently_in", pa.string()),
    ("filed_as_use_application_in", pa.string()),
    ("filing_basis_current_44d_in", pa.string()),
    ("filing_basis_current_44e_in", pa.string()),
    ("filing_basis_current_66a_in", pa.string()),
    ("without_basis_currently_in", pa.string()),
    ("section_8_filed_in", pa.string()),
    ("section_8_accepted_in", pa.string()),
    ("renewal_filed_in", pa.string()),
])

HEADER_FIELDS = {
    "attorney-name": "attorney_name",
    "abandonment-date": "abandonment_date",
    "status-date": "status_date",
    "published-for-opposition-date": "published_for_opposition_date",
    "intent-to-use-in": "intent_to_use_in",
    "use-application-currently-in": "use_application_currently_in",
    "filed-as-use-application-in": "filed_as_use_application_in",
    "filing-basis-current-44d-in": "filing_basis_current_44d_in",
    "filing-basis-current-44e-in": "filing_basis_current_44e_in",
    "filing-basis-current-66a-in": "filing_basis_current_66a_in",
    "without-basis-currently-in": "without_basis_currently_in",
    "section-8-filed-in": "section_8_filed_in",
    "section-8-accepted-in": "section_8_accepted_in",
    "renewal-filed-in": "renewal_filed_in",
}


def download_all() -> list[Path]:
    sys.path.insert(0, str(REPO / "src"))
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from novelty import uspto

    files = uspto.latest_backfile_files(uspto.PRODUCT_TRADEMARK_ANNUAL_APPLICATIONS)
    paths = []
    total = len(files)
    for i, f in enumerate(sorted(files, key=lambda x: x.file_name), 1):
        dest = RAW / f.file_name
        paths.append(dest)
        if dest.exists() and dest.stat().st_size == (f.size_bytes or dest.stat().st_size):
            continue
        t0 = time.time()
        uspto.stream_download(f.download_url, dest)
        print(f"[dl {i}/{total}] {f.file_name} "
              f"({dest.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)",
              file=sys.stderr, flush=True)
    return paths


def parse_all(paths: list[Path]) -> None:
    ev_writer = pq.ParquetWriter(PROC / "case_events.parquet", EV_SCHEMA,
                                 compression="zstd")
    ex_writer = pq.ParquetWriter(PROC / "case_extras.parquet", EX_SCHEMA,
                                 compression="zstd")
    code_desc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ev_buf = {k: [] for k in EV_SCHEMA.names}
    ex_buf = {k: [] for k in EX_SCHEMA.names}
    n_cases = 0
    t0 = time.time()

    def flush():
        nonlocal ev_buf, ex_buf
        if ev_buf["serial_number"]:
            ev_writer.write_table(pa.table(ev_buf, schema=EV_SCHEMA))
            ev_buf = {k: [] for k in EV_SCHEMA.names}
        if ex_buf["serial_number"]:
            ex_writer.write_table(pa.table(ex_buf, schema=EX_SCHEMA))
            ex_buf = {k: [] for k in EX_SCHEMA.names}

    for zi, zp in enumerate(paths, 1):
        try:
            z = zipfile.ZipFile(zp)
            xml_names = [n for n in z.namelist() if n.lower().endswith(".xml")]
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {zp.name}: {e}", file=sys.stderr, flush=True)
            continue
        for xn in xml_names:
            with z.open(xn) as fh:
                serial = None
                header: dict[str, str | None] = {}
                events: list[tuple[str, str, int, int]] = []
                in_header = False
                path: list[str] = []
                for ev, el in ET.iterparse(io.BufferedReader(fh, 1 << 20),
                                           events=("start", "end")):
                    tag = el.tag.split("}")[-1]
                    if ev == "start":
                        path.append(tag)
                        if tag == "case-file-header":
                            in_header = True
                        continue
                    # end
                    path.pop()
                    if tag == "serial-number" and len(path) >= 1 and path[-1] == "case-file":
                        serial = (el.text or "").strip()
                    elif in_header and tag in HEADER_FIELDS:
                        header[HEADER_FIELDS[tag]] = (el.text or "").strip() or None
                    elif tag == "case-file-header":
                        in_header = False
                    elif tag == "case-file-event-statement":
                        code = typ = None
                        date = seq = None
                        desc = None
                        for c in el:
                            ct = c.tag.split("}")[-1]
                            if ct == "code":
                                code = (c.text or "").strip()
                            elif ct == "type":
                                typ = (c.text or "").strip()
                            elif ct == "date":
                                try:
                                    date = int((c.text or "0").strip() or 0)
                                except ValueError:
                                    date = 0
                            elif ct == "number":
                                try:
                                    seq = int((c.text or "0").strip() or 0)
                                except ValueError:
                                    seq = 0
                            elif ct == "description-text":
                                desc = (c.text or "").strip()
                        if code:
                            events.append((code, typ or "", date or 0, seq or 0))
                            if desc and len(code_desc[code]) < 1000:
                                code_desc[code][desc] += 1
                        el.clear()
                    elif tag == "case-file":
                        if serial:
                            for code, typ, date, seq in events:
                                ev_buf["serial_number"].append(serial)
                                ev_buf["code"].append(code)
                                ev_buf["etype"].append(typ)
                                ev_buf["date"].append(date)
                                ev_buf["seq"].append(seq)
                            ex_buf["serial_number"].append(serial)
                            for k in EX_SCHEMA.names[1:]:
                                ex_buf[k].append(header.get(k))
                            n_cases += 1
                            if n_cases % 100_000 == 0:
                                flush()
                                print(f"[parse] {n_cases:,} cases "
                                      f"({zi}/{len(paths)} zips, "
                                      f"{time.time()-t0:.0f}s)",
                                      file=sys.stderr, flush=True)
                        serial, header, events = None, {}, []
                        el.clear()
        z.close()
        gc.collect()
    flush()
    ev_writer.close()
    ex_writer.close()
    codes = pa.table({
        "code": list(code_desc.keys()),
        "description": [c.most_common(1)[0][0] for c in code_desc.values()],
    })
    pq.write_table(codes, PROC / "event_code_dict.parquet")
    print(f"[done] {n_cases:,} cases parsed ({time.time()-t0:.0f}s)",
          file=sys.stderr, flush=True)


def main() -> int:
    paths = download_all()
    print(f"[dl done] {len(paths)} zips", file=sys.stderr, flush=True)
    parse_all(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
