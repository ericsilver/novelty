"""Rename the KL columns to name their reference window explicitly.

Why. The scorers followed the Murdock/Barron vocabulary, in which
"prospective" surprise is measured against the PAST and "retrospective"
surprise against the FUTURE. That is correct in the source literature and
matches this paper's definitions, but the words invite the opposite reading:
a reader meeting `prospective_kl` in a parquet naturally takes it to mean
"looking forward," and would then invert every interpretation built on it.
The repository is public and the panels are meant to be reused, so the
columns are renamed to say which window they are measured against.

Verified against the code before renaming, in both scorers:
  src/novelty/surprise.py      log_pros aggregates years [y-W, y-1]  -> past
                               log_retr aggregates years [y+1, y+W]  -> future
  scripts/topic_p_scorer_all.py  q_p = past window, q_f = future window
                               pros = KL(P || q_p), retr = KL(P || q_f)

Mapping (the sign of dKL is unchanged: dKL = vs_past - vs_future):
  prospective_kl       -> kl_vs_past
  retrospective_kl     -> kl_vs_future
  n_ref_prospective    -> n_ref_past
  n_ref_retrospective  -> n_ref_future
  topic_pros           -> topic_kl_vs_past
  topic_retr           -> topic_kl_vs_future
  topic_dkl            unchanged (the paper's named construct)

Idempotent: a file already carrying the new names is skipped. Each file is
rewritten via a temporary sibling and then swapped, so an interrupted run
leaves every file readable under one naming scheme or the other.

Usage:  PYTHONPATH=src .venv-analysis/Scripts/python.exe scripts/migrate_kl_column_names.py [--dry-run]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"

RENAME = {
    "prospective_kl": "kl_vs_past",
    "retrospective_kl": "kl_vs_future",
    "n_ref_prospective": "n_ref_past",
    "n_ref_retrospective": "n_ref_future",
    "topic_pros": "topic_kl_vs_past",
    "topic_retr": "topic_kl_vs_future",
}


def main() -> int:
    dry = "--dry-run" in sys.argv
    targets = sorted(PROC.glob("*.parquet"))
    touched = skipped = failed = 0

    for p in targets:
        try:
            names = set(pl.scan_parquet(p).collect_schema().names())
        except Exception as exc:
            print(f"[skip:unreadable] {p.name}: {exc}", file=sys.stderr)
            failed += 1
            continue

        mapping = {old: new for old, new in RENAME.items() if old in names}
        if not mapping:
            skipped += 1
            continue

        collision = {new for new in mapping.values() if new in names}
        if collision:
            print(f"[skip:collision] {p.name}: {sorted(collision)} already present",
                  file=sys.stderr)
            failed += 1
            continue

        if dry:
            print(f"[dry] {p.name}: {mapping}")
            touched += 1
            continue

        tmp = p.with_suffix(".parquet.migrating")
        try:
            pl.read_parquet(p).rename(mapping).write_parquet(tmp)
            os.replace(tmp, p)
            touched += 1
            if touched % 25 == 0:
                print(f"  ... {touched} rewritten", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[fail] {p.name}: {exc}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            failed += 1

    print(f"[done] rewritten={touched} untouched={skipped} failed={failed}",
          file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
