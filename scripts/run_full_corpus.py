"""Build vocab + surprise for every NICE class that has a tm_class parquet,
skipping stages whose outputs already exist. Unlike process_all_classes.py,
a failing class is logged and skipped rather than aborting the batch.

Usage:
    python scripts/run_full_corpus.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"

# The novelty package lives under src/ and is not installed in this venv.
ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"))


def run(args: list[str]) -> int:
    print(">>> " + " ".join(args), file=sys.stderr, flush=True)
    t0 = time.time()
    rc = subprocess.run(args, env=ENV).returncode
    print(f"    ({time.time()-t0:.1f}s, rc={rc})", file=sys.stderr, flush=True)
    return rc


def main() -> int:
    classes = sorted(p.stem.replace("tm_class", "")
                     for p in DATA.glob("tm_class*.parquet")
                     if p.stem.replace("tm_class", "").isdigit())
    print(f"[plan] {len(classes)} classes", file=sys.stderr, flush=True)

    py = sys.executable
    failed: list[str] = []
    for i, cls in enumerate(classes, 1):
        rec = DATA / f"tm_class{cls}.parquet"
        vocab = DATA / f"vocab_class{cls}.parquet"
        sur = DATA / f"surprise_class{cls}.parquet"
        print(f"[{i}/{len(classes)}] class {cls}", file=sys.stderr, flush=True)

        try:
            if not vocab.exists():
                if run([py, "-m", "novelty.dictionary",
                        "--input", str(rec), "--out", str(vocab)]) != 0:
                    raise RuntimeError("dictionary failed")
            if not sur.exists():
                if run([py, "-m", "novelty.surprise",
                        "--records", str(rec), "--vocab", str(vocab),
                        "--out", str(sur)]) != 0:
                    raise RuntimeError("surprise failed")
        except Exception as e:  # noqa: BLE001
            print(f"    !! class {cls}: {e}; continuing", file=sys.stderr, flush=True)
            failed.append(cls)
            # remove partial outputs so a re-run retries cleanly
            for p in (vocab, sur):
                if p.exists() and p.stat().st_size == 0:
                    p.unlink()

    print(f"[done] {len(classes)-len(failed)} ok, {len(failed)} failed: {failed}",
          file=sys.stderr, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
