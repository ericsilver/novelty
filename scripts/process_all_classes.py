"""For every per-class parquet under data/processed/, run:
    1. dictionary  (vocab parquet)
    2. surprise    (per-filing KL parquet)
    3. firm_year   (registrant-year aggregation)
    4. survival    (outcomes parquet)

Skips a stage if its output already exists.

Usage:
    python scripts/process_all_classes.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "processed"


def run(args: list[str]) -> None:
    print(">>> " + " ".join(args), file=sys.stderr, flush=True)
    subprocess.run(args, check=True)


def main() -> int:
    classes = sorted(p.stem.replace("tm_class", "")
                     for p in DATA.glob("tm_class*.parquet")
                     if p.stem.replace("tm_class", "").isdigit())
    print(f"[plan] {len(classes)} classes to process", file=sys.stderr)

    py = sys.executable
    for cls in classes:
        rec = DATA / f"tm_class{cls}.parquet"
        vocab = DATA / f"vocab_class{cls}.parquet"
        sur = DATA / f"surprise_class{cls}.parquet"
        fy = DATA / f"firm_year_class{cls}.parquet"
        out = DATA / f"outcomes_class{cls}.parquet"

        # Vocabulary
        if not vocab.exists():
            run([py, "-m", "novelty.dictionary", "--input", str(rec), "--out", str(vocab)])

        # Surprise (default min_df 50)
        if not sur.exists():
            run([py, "-m", "novelty.surprise",
                 "--records", str(rec), "--vocab", str(vocab), "--out", str(sur)])

        # Firm-year
        if not fy.exists():
            run([py, "-m", "novelty.firm_year",
                 "--surprise", str(sur), "--out", str(fy)])

        # Survival outcomes
        if not out.exists():
            run([py, "-m", "novelty.survival",
                 "--records", str(rec), "--surprise", str(sur), "--out", str(out)])

    print("[done]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
