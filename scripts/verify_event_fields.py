"""Step 1 verification: download ONE TRTYRAP yearly file and check whether
the XML carries (a) prosecution-history events with dates, (b) attorney /
correspondent fields, (c) filing-basis fields.

Prints the full element-tag inventory of the first case files plus samples
of the target fields. Downloads to data/raw/ (kept for the full re-parse).
"""
from __future__ import annotations

import collections
import io
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    from novelty import uspto

    files = uspto.latest_backfile_files(uspto.PRODUCT_TRADEMARK_ANNUAL_APPLICATIONS)
    # pick a recent, moderate-size yearly file (2018-ish)
    cand = [f for f in files if "18" in f.file_name or "2018" in f.file_name] or files
    f = sorted(cand, key=lambda x: x.size_bytes or 0)[len(cand) // 2]
    print(f"[pick] {f.file_name} ({(f.size_bytes or 0)/1e6:.0f} MB)", file=sys.stderr, flush=True)

    dest = RAW / f.file_name
    if not dest.exists():
        uspto.stream_download(f.download_url, dest)
    print(f"[have] {dest} ({dest.stat().st_size/1e6:.0f} MB)", file=sys.stderr, flush=True)

    tag_counts: collections.Counter = collections.Counter()
    samples: dict[str, list[str]] = collections.defaultdict(list)
    INTEREST = ("event", "attorney", "correspondent", "basis", "transaction",
                "prosecution", "action", "office")

    with zipfile.ZipFile(dest) as z:
        xml_names = [n for n in z.namelist() if n.lower().endswith(".xml")]
        print(f"[zip ] {len(xml_names)} xml member(s): {xml_names[:3]}",
              file=sys.stderr, flush=True)
        with z.open(xml_names[0]) as fh:
            n_cases = 0
            path = []
            for ev, el in ET.iterparse(io.BufferedReader(fh), events=("start", "end")):
                tag = el.tag.split("}")[-1]
                if ev == "start":
                    path.append(tag)
                    continue
                # end event
                full = "/".join(path[-4:])
                tag_counts[full] += 1
                low = tag.lower()
                if any(k in low for k in INTEREST) and el.text and el.text.strip():
                    if len(samples[full]) < 3:
                        samples[full].append(el.text.strip()[:60])
                if tag == "case-file":
                    n_cases += 1
                    el.clear()
                    if n_cases >= 200:
                        break
                path.pop()
    print(f"\n[scan] {n_cases} case files parsed", file=sys.stderr, flush=True)
    print("\n=== tags matching interest keywords ===", file=sys.stderr, flush=True)
    for full, c in sorted(tag_counts.items()):
        low = full.lower()
        if any(k in low for k in INTEREST):
            s = samples.get(full, [])
            print(f"  {full}  x{c}   e.g. {s}", file=sys.stderr, flush=True)
    print("\n=== all distinct tag paths (first 120) ===", file=sys.stderr, flush=True)
    for full in sorted(set(tag_counts))[:120]:
        print(f"  {full}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
