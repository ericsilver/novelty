"""Per-clause audit of the CLEAN filter.

Reports, for every clause separately and for the cumulative pipeline:
    n_filings_kept,  n_filings_dropped_by_this_clause,  share_dropped

Output:
  stdout: human-readable table
  paper/results/clean_filter_audit.json: machine-readable summary
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"


def main() -> int:
    cols = ["n_ref_prospective", "n_ref_retrospective", "n_terms", "year"]
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        parts.append(pl.read_parquet(path).select(cols))
    df = pl.concat(parts)
    n_total = df.height
    print(f"raw filings (all classes, all years, all densities): {n_total:,}")

    clauses = [
        ("year >= 1990",                 pl.col("year") >= 1990),
        ("year <= 2018",                 pl.col("year") <= 2018),
        ("n_ref_prospective >= 1000",    pl.col("n_ref_prospective") >= 1000),
        ("n_ref_retrospective >= 1000",  pl.col("n_ref_retrospective") >= 1000),
        ("n_terms >= 3",                 pl.col("n_terms") >= 3),
    ]

    # --- (a) marginal pass rate of each clause individually ---
    print("\n--- marginal pass rate (each clause applied alone to the raw corpus) ---")
    marg_rows = []
    for name, expr in clauses:
        n = df.filter(expr).height
        marg_rows.append({"clause": name, "n_kept_alone": n,
                          "n_dropped_alone": n_total - n,
                          "pct_dropped_alone": (n_total - n) / n_total})
        print(f"  {name:<32} kept {n:>11,}  dropped {n_total - n:>11,}  "
              f"({(n_total - n)/n_total*100:5.1f}%)")

    # --- (b) cumulative drop applying clauses in order ---
    print("\n--- cumulative drop (clauses applied in order, AND'd together) ---")
    running = pl.lit(True)
    cum_rows = []
    last_n = n_total
    for name, expr in clauses:
        running = running & expr
        n = df.filter(running).height
        cum_rows.append({"clause_added": name, "n_after": n,
                         "n_lost_this_step": last_n - n,
                         "pct_lost_this_step": (last_n - n) / n_total})
        print(f"  + {name:<32} → kept {n:>11,}  "
              f"lost-this-step {last_n - n:>11,}  ({(last_n - n)/n_total*100:5.1f}% of raw)")
        last_n = n

    # --- (c) marginal-given-others: drop each clause from the full filter and count what comes back ---
    full = clauses[0][1]
    for _, e in clauses[1:]:
        full = full & e
    n_full = df.filter(full).height
    print(f"\n  full CLEAN filter → kept {n_full:,}  ({n_full/n_total*100:.1f}% of raw)")

    print("\n--- if we dropped just one clause from CLEAN, how many extra filings come back? ---")
    rescue_rows = []
    for name, expr in clauses:
        keep = pl.lit(True)
        for n2, e2 in clauses:
            if n2 == name:
                continue
            keep = keep & e2
        n = df.filter(keep).height
        gain = n - n_full
        rescue_rows.append({"clause_dropped": name, "n_kept": n,
                            "extra_vs_full_clean": gain,
                            "pct_extra_of_full_clean": gain / n_full})
        print(f"  drop {name:<32} → kept {n:>11,}  (+{gain:>9,} vs full CLEAN, "
              f"+{gain/n_full*100:5.1f}%)")

    out = RESULTS / "clean_filter_audit.json"
    out.write_text(json.dumps({
        "n_raw": n_total,
        "n_full_clean": n_full,
        "marginal_each_clause": marg_rows,
        "cumulative_in_order": cum_rows,
        "rescue_if_drop_one": rescue_rows,
    }, indent=2, default=str))
    print(f"\n[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
