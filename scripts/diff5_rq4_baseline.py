"""RQ4 population-debut-rate baseline contrast.

The raw 73.7% entrant share among theme-origin carriers (diff4) could partly
reflect the corpus's overall debut rate in those years. Here we compute, for
each year, the population-wide entrant share among ALL granted filings in the
4-class subset (owner's first grant within {y, y-1}). Then for each theme we
compute the YEAR-MATCHED expected entrant share -- the mean of population
entrant rates weighted by the year distribution of that theme's 50 earliest
carriers -- and report the theme's entrant share MINUS the year-matched
baseline as the Schumpeter-I excess.

Output: paper/results/diff5_rq4_baseline.json (+ stdout)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT = REPO / "paper" / "results" / "diff5_rq4_baseline.json"
CLASSES = ["009", "042", "039", "035"]
TAU = 0.20
N_EARLY = 50


def main() -> int:
    t0 = time.time()
    # owner -> first grant year across the 4 classes
    print("[1/3] owner -> first grant year ...", file=sys.stderr)
    own_year = []
    for c in CLASSES:
        d = pl.read_parquet(DATA / f"tm_class{c}.parquet",
                             columns=["serial_number", "owner_name", "filing_date",
                                      "registration_date"]).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
        ).filter(
            pl.col("year").is_not_null() & pl.col("owner_name").is_not_null()
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 4)
        ).select("serial_number", "owner_name", "year")
        own_year.append(d)
    all_grants = pl.concat(own_year)
    owner_first = dict(all_grants.group_by("owner_name").agg(pl.col("year").min())
                                   .iter_rows())
    print(f"     {len(owner_first):,} owners", file=sys.stderr)

    # tag each granted filing entrant=1 if owner_first[owner] >= year-1
    all_grants = all_grants.with_columns(
        pl.col("owner_name").map_elements(lambda o: owner_first.get(o, 99999),
                                          return_dtype=pl.Int32).alias("of_year")
    ).with_columns(((pl.col("of_year") >= pl.col("year") - 1)).alias("entrant"))

    # population entrant share by year (1990-2024)
    print("[2/3] population entrant share by year ...", file=sys.stderr)
    pop = (all_grants.filter((pl.col("year") >= 1990) & (pl.col("year") <= 2024))
                     .group_by("year").agg(pl.col("entrant").mean().alias("pop_share"),
                                            pl.len().alias("n")).sort("year"))
    pop_pd = pop.to_pandas()
    pop_share_by_year = dict(zip(pop_pd.year, pop_pd.pop_share))
    print(f"     population entrant share: mean {pop_pd.pop_share.mean():.1%}, "
          f"range {pop_pd.pop_share.min():.1%}-{pop_pd.pop_share.max():.1%}",
          file=sys.stderr)

    # per-theme year-matched baseline
    print("[3/3] per-theme contrast ...", file=sys.stderr)
    topics = pl.read_parquet(DATA / "filing_topics.parquet")
    # join owner_name into filing_topics via serial -> all_grants
    j = topics.join(all_grants.select("serial_number", "owner_name", "entrant"),
                    on="serial_number", how="inner")
    T = int(j["top_topic"].max()) + 1

    rows = []
    for k in range(T):
        cell = j.filter((pl.col("top_topic") == k) & (pl.col("top_weight") >= TAU))
        if cell.height < N_EARLY:
            continue
        early = cell.sort("year").head(N_EARLY)
        theme_share = float(early["entrant"].mean())
        # year-matched baseline: avg pop_share over the years of these 50
        years = early["year"].to_list()
        bases = [pop_share_by_year.get(y, np.nan) for y in years]
        baseline = float(np.nanmean(bases))
        rows.append({"theme": k, "n_early": early.height,
                     "theme_entrant_share": theme_share, "baseline_share": baseline,
                     "excess": theme_share - baseline,
                     "year_min": int(min(years)), "year_max": int(max(years))})

    es = np.array([r["excess"] for r in rows])
    ts = np.array([r["theme_entrant_share"] for r in rows])
    bs = np.array([r["baseline_share"] for r in rows])
    print(f"\n[RQ4 base] {len(rows)} themes")
    print(f"   theme entrant share:    mean {ts.mean():.1%}  median {np.median(ts):.1%}")
    print(f"   year-matched baseline:  mean {bs.mean():.1%}  median {np.median(bs):.1%}")
    print(f"   EXCESS (theme - base):  mean {es.mean():+.1%}  median {np.median(es):+.1%}")
    print(f"   themes with excess > 0: {(es>0).sum()}/{len(es)}")
    print(f"   themes with excess > 10pp: {(es>0.10).sum()}/{len(es)}")
    print(f"   themes with excess < 0 (incumbent-biased vs base): {(es<0).sum()}/{len(es)}")

    out = {
        "n_themes": len(rows),
        "pop_entrant_share_by_year": [{"year": int(r.year), "share": float(r.pop_share),
                                       "n": int(r.n)} for r in pop_pd.itertuples()],
        "summary": {
            "theme_entrant_mean": float(ts.mean()),
            "theme_entrant_median": float(np.median(ts)),
            "baseline_mean": float(bs.mean()),
            "baseline_median": float(np.median(bs)),
            "excess_mean": float(es.mean()),
            "excess_median": float(np.median(es)),
            "share_excess_positive": float((es > 0).mean()),
            "share_excess_above_10pp": float((es > 0.10).mean()),
            "share_excess_negative": float((es < 0).mean()),
        },
        "by_theme": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[done] -> {OUT}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
