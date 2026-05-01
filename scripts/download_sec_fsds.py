"""Download all SEC Financial Statement Data Sets quarterly ZIPs.

Polite by SEC rules: User-Agent with contact email, ~1 req/sec, retries on 429/5xx.

Usage:
    python scripts/download_sec_fsds.py --start 2009q1 --end 2024q4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "raw" / "sec"
UA = "Eric Silver epsilver@gmail.com (novelty research)"
BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"


def quarters(start: str, end: str):
    sy, sq = int(start[:4]), int(start[-1])
    ey, eq = int(end[:4]), int(end[-1])
    y, q = sy, sq
    while (y, q) <= (ey, eq):
        yield f"{y}q{q}"
        q += 1
        if q > 4:
            q = 1
            y += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2009q1")
    ap.add_argument("--end", default="2024q4")
    ap.add_argument("--sleep", type=float, default=1.1, help="seconds between requests")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})

    qs = list(quarters(args.start, args.end))
    print(f"[plan] {len(qs)} quarterly ZIPs ({args.start} .. {args.end})", file=sys.stderr)

    for q in qs:
        dest = DEST / f"{q}.zip"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {q}.zip ({dest.stat().st_size/1e6:.1f} MB)", file=sys.stderr)
            continue
        url = f"{BASE}/{q}.zip"
        for attempt in range(4):
            try:
                t0 = time.time()
                with s.get(url, stream=True, timeout=180) as r:
                    if r.status_code == 404:
                        print(f"[404 ] {q} not yet published", file=sys.stderr)
                        break
                    r.raise_for_status()
                    tmp = dest.with_suffix(".part")
                    with tmp.open("wb") as fh:
                        for buf in r.iter_content(chunk_size=1 << 20):
                            if buf:
                                fh.write(buf)
                    tmp.rename(dest)
                size_mb = dest.stat().st_size / 1e6
                print(f"[get ] {q}.zip {size_mb:.1f} MB in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
                break
            except (requests.RequestException, OSError) as e:
                wait = 2 ** attempt
                print(f"[retry {attempt+1}] {q}: {e}; sleep {wait}s", file=sys.stderr)
                time.sleep(wait)
        time.sleep(args.sleep)

    print(f"[done] {len(list(DEST.glob('*.zip')))} ZIPs in {DEST}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
