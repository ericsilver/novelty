"""Download PatentsView bulk-data tables needed to compute per-assignee
annual patent counts.

Files (~few GB total):
  g_patent.tsv.zip               patent_id, patent_date, ...
  g_assignee_disambiguated.tsv.zip   patent_id, assignee_id, disambig name

Output: data/raw/patentsview/{g_patent,g_assignee_disambiguated}.tsv
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "raw" / "patentsview"
UA = "Eric Silver epsilver@gmail.com (novelty research)"

FILES = [
    "g_patent.tsv.zip",
    "g_assignee_disambiguated.tsv.zip",
]
BASE = "https://s3.amazonaws.com/data.patentsview.org/download"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA})

    for fname in FILES:
        zp = DEST / fname
        tsv = DEST / fname.replace(".zip", "")
        if tsv.exists() and tsv.stat().st_size > 0:
            print(f"[skip] {tsv.name} already extracted ({tsv.stat().st_size/1e6:.0f} MB)", file=sys.stderr)
            continue
        if not zp.exists():
            url = f"{BASE}/{fname}"
            t0 = time.time()
            print(f"[get ] {url}", file=sys.stderr, flush=True)
            with s.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = zp.with_suffix(".part")
                with tmp.open("wb") as fh:
                    for buf in r.iter_content(chunk_size=1 << 20):
                        if buf:
                            fh.write(buf)
                tmp.rename(zp)
            print(f"       {zp.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s", file=sys.stderr)
        print(f"[unzip] {zp.name}", file=sys.stderr, flush=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(DEST)
        try:
            zp.unlink()
        except OSError:
            pass

    for fname in FILES:
        tsv = DEST / fname.replace(".zip", "")
        if tsv.exists():
            print(f"  {tsv.name}: {tsv.stat().st_size/1e6:.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
