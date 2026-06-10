"""Diffusion Phase 0 (#9/#10 demo): cross-class theme transit.

PROPOSAL_diffusion's canonical phenomenon: a business-model vocabulary that is
novel in one industry and later arrives in others ("Uber-for-X"). We track a
curated set of multi-token business-model phrases by regex on the raw g/s text
(bypassing the analyzer's stopword removal, which mangles phrases like
"on demand" / "as a service"), and compute per (class, year) prevalence among
GRANTED filings. The first year each class crosses a prevalence floor gives the
theme's arrival ordering across {009 software, 042 tech-svcs, 039 transport,
035 advertising/retail} -- a direct read of cross-industry diffusion.

This is descriptive (the robust register of the program); it does not by itself
prove "innovation". Output: paper/results/diff2_transit.json (+ stdout).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT = REPO / "paper" / "results" / "diff2_transit.json"

CLASSES = {"009": "software", "042": "tech-svcs", "039": "transport", "035": "advert-retail"}
FLOOR = 0.01   # prevalence floor to count a theme "present" in a class-year
YEAR_MIN = 1990

# label -> regex (case-insensitive, matched on raw goods_services)
THEMES = {
    "on-demand": r"on[\s-]?demand",
    "peer-to-peer": r"peer[\s-]?to[\s-]?peer|p2p",
    "blockchain": r"blockchain|distributed ledger",
    "cryptocurrency": r"cryptocurrenc|crypto[\s-]?currency|digital currenc",
    "as-a-service": r"as a service|\bsaas\b|\bpaas\b|\biaas\b",
    "artificial-intelligence": r"artificial intelligence|machine learning|\bA\.?I\.?\b",
    "subscription": r"subscription",
    "cloud": r"cloud[\s-]?(comput|based|storage|platform)|cloud\b",
    "mobile-app": r"mobile app|smartphone app|downloadable.{0,20}application",
    "streaming": r"stream(ing)?\b",
    "augmented-reality": r"augmented reality|virtual reality|\bAR\b|\bVR\b",
    "internet-of-things": r"internet of things|\bIoT\b|connected device",
    "real-time": r"real[\s-]?time",
    "marketplace": r"marketplace|online platform connecting",
}


def main() -> int:
    frames = []
    for c in CLASSES:
        p = DATA / f"tm_class{c}.parquet"
        if not p.exists():
            print(f"MISSING {p} -- run download_all_classes.py first", file=sys.stderr)
            raise SystemExit(2)
        df = pl.read_parquet(p, columns=["filing_date", "registration_date", "goods_services"])
        df = df.with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
            pl.lit(c).alias("nice"),
            pl.col("goods_services").str.to_lowercase().alias("gs"),
        ).filter(
            (pl.col("gs").str.len_chars() > 0) & pl.col("year").is_not_null()
            & (pl.col("year") >= YEAR_MIN)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 4)  # granted
        ).select("nice", "year", "gs")
        frames.append(df)
        print(f"  class {c} ({CLASSES[c]}): {df.height:,} granted", file=sys.stderr)
    rec = pl.concat(frames)

    # tag each filing with each theme's presence
    for label, pat in THEMES.items():
        rec = rec.with_columns(pl.col("gs").str.contains(f"(?i){pat}").alias(f"t_{label}"))

    res = {"classes": CLASSES, "floor": FLOOR, "themes": {}}
    print(f"\nTheme transit across classes (first year prevalence >= {FLOOR:.0%} among granted):")
    print(f"{'theme':>24} | " + " | ".join(f"{CLASSES[c]:>13}" for c in CLASSES))
    for label in THEMES:
        prev = (rec.group_by(["nice", "year"]).agg(pl.col(f"t_{label}").mean().alias("prev"),
                                                    pl.len().alias("n")))
        arrivals = {}
        for c in CLASSES:
            cc = prev.filter((pl.col("nice") == c) & (pl.col("prev") >= FLOOR)
                             & (pl.col("n") >= 50)).sort("year")
            arrivals[c] = int(cc["year"][0]) if cc.height else None
        # peak prevalence per class (for context)
        peaks = {}
        for c in CLASSES:
            cc = prev.filter((pl.col("nice") == c) & (pl.col("n") >= 50))
            peaks[c] = float(cc["prev"].max()) if cc.height else 0.0
        res["themes"][label] = {"arrival_year": arrivals, "peak_prev": peaks}
        row = " | ".join(f"{(str(arrivals[c]) if arrivals[c] else '--'):>13}" for c in CLASSES)
        print(f"{label:>24} | {row}")

    # transit summary: themes whose origin (earliest arrival) is software/tech (009/042)
    print("\nThemes that originate in software/tech (009/042) then reach 039/035:")
    transits = []
    for label, d in res["themes"].items():
        yrs = {c: y for c, y in d["arrival_year"].items() if y}
        if not yrs:
            continue
        origin = min(yrs, key=yrs.get)
        reached_phys = [c for c in ("039", "035") if c in yrs]
        if origin in ("009", "042") and reached_phys:
            lags = {c: yrs[c] - yrs[origin] for c in reached_phys}
            transits.append({"theme": label, "origin": origin, "origin_year": yrs[origin],
                             "reached": {c: yrs[c] for c in reached_phys}, "lag_years": lags})
            print(f"  {label}: born {CLASSES[origin]} {yrs[origin]} -> "
                  + ", ".join(f"{CLASSES[c]} +{lags[c]}y" for c in reached_phys))
    res["software_origin_transits"] = transits

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[done] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
