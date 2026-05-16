"""Recompute the full per-class pipeline with H=2y decay reference.

Per class:
  1. If surprise_decay_class{N}_h2.parquet missing: run surprise_decay H_pros=H_retr=2.0
  2. Convert decay output to surprise-schema (rename n_eff_* -> n_ref_*, drop
     halflife cols), overwrite surprise_class{N}.parquet
  3. Regenerate firm_year_class{N}.parquet via novelty.firm_year
  4. Regenerate outcomes_class{N}.parquet via novelty.survival

Class 9 already has H=2 decay output (surprise_decay_class009_h2.parquet);
that step is skipped.  Other classes run sequentially to bound peak memory.

Usage:
    python scripts/recompute_h2.py            # all 45 classes
    python scripts/recompute_h2.py --classes 009,032   # subset
    python scripts/recompute_h2.py --convert-only      # skip decay; only convert + regen
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "processed"
H = 2.0


def run(args: list[str]) -> int:
    print(">>> " + " ".join(args), file=sys.stderr, flush=True)
    t0 = time.time()
    rc = subprocess.run(args).returncode
    print(f"    ({time.time()-t0:.1f}s, rc={rc})", file=sys.stderr, flush=True)
    return rc


def convert_decay_to_surprise(decay_path: Path, out_path: Path) -> None:
    df = pl.read_parquet(decay_path)
    df = df.rename({"n_eff_prospective": "n_ref_prospective",
                    "n_eff_retrospective": "n_ref_retrospective"})
    df = df.with_columns(
        pl.col("n_ref_prospective").fill_null(0).cast(pl.Int32, strict=False).fill_null(0),
        pl.col("n_ref_retrospective").fill_null(0).cast(pl.Int32, strict=False).fill_null(0),
    ).drop(["halflife_pros", "halflife_retr"]).select(
        "serial_number", "filing_date", "year", "mark_identification",
        "owner_name", "n_terms", "prospective_kl", "n_ref_prospective",
        "retrospective_kl", "n_ref_retrospective",
    )
    df.write_parquet(out_path)
    print(f"    converted -> {out_path.name}  ({df.height:,} rows)", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="all")
    ap.add_argument("--convert-only", action="store_true",
                    help="skip surprise_decay, only convert existing decays + regen")
    args = ap.parse_args()

    if args.classes == "all":
        classes = sorted(p.stem.replace("tm_class", "")
                         for p in DATA.glob("tm_class*.parquet")
                         if p.stem.replace("tm_class", "").isdigit())
    else:
        classes = [c.strip().zfill(3) for c in args.classes.split(",")]

    py = sys.executable
    for cls in classes:
        print(f"\n=== class {cls} ===", file=sys.stderr, flush=True)
        rec = DATA / f"tm_class{cls}.parquet"
        vocab = DATA / f"vocab_class{cls}.parquet"
        decay = DATA / f"surprise_decay_class{cls}_h2.parquet"
        sur = DATA / f"surprise_class{cls}.parquet"
        fy = DATA / f"firm_year_class{cls}.parquet"
        out = DATA / f"outcomes_class{cls}.parquet"

        if not rec.exists():
            print(f"    SKIP: missing {rec.name}", file=sys.stderr); continue
        if not vocab.exists():
            print(f"    SKIP: missing {vocab.name}", file=sys.stderr); continue

        # Step 1: surprise_decay H=2
        if not decay.exists() and not args.convert_only:
            rc = run([py, "-m", "novelty.surprise_decay",
                      "--records", str(rec), "--vocab", str(vocab),
                      "--out", str(decay),
                      "--halflife-pros", str(H), "--halflife-retr", str(H)])
            if rc != 0 or not decay.exists():
                print(f"    DECAY FAILED for class {cls}", file=sys.stderr); continue
        elif not decay.exists():
            print(f"    SKIP class {cls}: no decay output and --convert-only", file=sys.stderr); continue

        # Step 2: convert to surprise-schema
        convert_decay_to_surprise(decay, sur)

        # Step 3: firm_year (force rebuild)
        if fy.exists(): fy.unlink()
        rc = run([py, "-m", "novelty.firm_year",
                  "--surprise", str(sur), "--out", str(fy)])
        if rc != 0:
            print(f"    FIRM_YEAR FAILED for class {cls}", file=sys.stderr); continue

        # Step 4: outcomes (force rebuild)
        if out.exists(): out.unlink()
        rc = run([py, "-m", "novelty.survival",
                  "--records", str(rec), "--surprise", str(sur),
                  "--out", str(out)])
        if rc != 0:
            print(f"    SURVIVAL FAILED for class {cls}", file=sys.stderr); continue

        print(f"    OK class {cls}", file=sys.stderr, flush=True)

    print("\n[done] recompute_h2 finished", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
