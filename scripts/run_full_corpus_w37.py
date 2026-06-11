"""Build W=3 and W=7 surprise scores for every class (vocab already built).
Skips existing outputs; continues past per-class failures.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
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
    py = sys.executable
    failed = []
    for i, cls in enumerate(classes, 1):
        rec = DATA / f"tm_class{cls}.parquet"
        vocab = DATA / f"vocab_class{cls}.parquet"
        if not vocab.exists():
            print(f"[{i}] class {cls}: no vocab, skipping", file=sys.stderr, flush=True)
            failed.append(cls)
            continue
        for w in (3, 7):
            out = DATA / f"surprise_class{cls}_W{w}.parquet"
            if out.exists():
                continue
            print(f"[{i}/{len(classes)}] class {cls} W={w}", file=sys.stderr, flush=True)
            if run([py, "-m", "novelty.surprise", "--records", str(rec),
                    "--vocab", str(vocab), "--out", str(out),
                    "--window", str(w)]) != 0:
                failed.append(f"{cls}_W{w}")
                if out.exists() and out.stat().st_size == 0:
                    out.unlink()
    print(f"[done] failed: {failed}", file=sys.stderr, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
