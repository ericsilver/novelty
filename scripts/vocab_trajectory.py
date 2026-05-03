"""Time-series of selected bigram usage in Class 9 (Software & Electronics).

For each bigram, plot the share of the year's clean filings whose
goods/services text contains the bigram as a substring (case-insensitive).
Selected to span four trajectory shapes:
  (a) rising and persistent (downloadable software)
  (b) entered, swelled, then declined (compact disc, world wide web)
  (c) recent rise (blockchain, smart contract, mobile app)
  (d) old terms that essentially vanished (cassette, dial-up, fax)

Output:
  paper/results/vocab_trajectory.png
  paper/results/vocab_trajectory_metrics.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"


TRAJECTORIES = [
    # (display name, regex pattern, color)
    ("downloadable software",   r"\bdownloadable\s+software\b",   "#1f77b4"),
    ("mobile app(s)",           r"\bmobile\s+app(?:lication)?s?\b","#2ca02c"),
    ("blockchain",              r"\bblockchain\b",                 "#117a3a"),
    ("smart contract(s)",       r"\bsmart\s+contract(?:s)?\b",     "#0a5a2e"),
    ("compact disc(s) / CD-ROM",r"\bcompact\s+disc(?:s)?\b|\bCD[- ]?ROM\b","#bf6b3a"),
    ("World Wide Web",          r"\bworld\s+wide\s+web\b",         "#7a1111"),
    ("cassette tape(s)",        r"\bcassette(?:\s+tape)?s?\b",     "#888888"),
    ("dial-up / dial up",       r"\bdial[- ]?up\b",                "#444444"),
    ("fax (machine)",           r"\bfax(?:\s+machine)?(?:es)?\b",  "#222222"),
    ("artificial intelligence", r"\bartificial\s+intelligence\b",  "#9467bd"),
    ("virtual reality",         r"\bvirtual\s+reality\b",          "#d62728"),
    ("non-fungible token / NFT",r"\bnon[- ]?fungible\s+token(?:s)?\b|\bnft\b", "#e377c2"),
]


def main() -> int:
    print("[load] Class 9 records…", flush=True)
    df = pl.read_parquet(PROC / "tm_class009.parquet").select(
        "filing_date", "goods_services"
    ).filter(
        pl.col("filing_date").is_not_null() & pl.col("goods_services").is_not_null()
    ).with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"))
    df = df.filter((pl.col("year") >= 1985) & (pl.col("year") <= 2025))
    print(f"  {df.height:,} filings, years {df['year'].min()}-{df['year'].max()}", flush=True)

    # Per-year filing counts (denominator)
    year_counts = (
        df.group_by("year").len().rename({"len": "n_filings"}).sort("year")
    )
    print("  per-year totals computed.")

    # For each bigram, compute per-year share
    pdf = df.to_pandas()
    pdf["gs_lc"] = pdf["goods_services"].str.lower().fillna("")

    series_data: dict[str, np.ndarray] = {}
    for label, pattern, color in TRAJECTORIES:
        rx = re.compile(pattern, re.IGNORECASE)
        pdf["hit"] = pdf["gs_lc"].str.contains(rx)
        per_year = pdf.groupby("year")["hit"].agg(["sum", "count"]).reset_index()
        per_year["share"] = per_year["sum"] / per_year["count"]
        series_data[label] = per_year[["year", "share", "count"]]
        peak = per_year.loc[per_year["share"].idxmax()]
        print(f"  {label:<32} peak share {peak['share']*100:.1f}% in {int(peak['year'])} ({int(peak['count']):,} filings/yr)")

    # ---- Two-panel figure: rising-and-persistent on top, peaked-and-vanished on bottom ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    rising = ["downloadable software", "mobile app(s)", "blockchain",
              "smart contract(s)", "artificial intelligence",
              "virtual reality", "non-fungible token / NFT"]
    declining = ["compact disc(s) / CD-ROM", "World Wide Web",
                 "cassette tape(s)", "dial-up / dial up", "fax (machine)"]

    for label, pattern, color in TRAJECTORIES:
        sd = series_data[label]
        sd = sd[sd["count"] >= 50]  # need at least 50 filings in year for stable share
        ax = axes[0] if label in rising else axes[1]
        ax.plot(sd["year"], sd["share"] * 100, "-", color=color, linewidth=1.6, label=label)

    for ax in axes:
        ax.set_ylabel("Share of year's Class-9 filings\nwhose G/S text contains the term (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    axes[0].set_title("Newer terms: rising or recent emergence")
    axes[1].set_title("Older terms: peaked then declined")
    axes[1].set_xlabel("Filing year")

    fig.tight_layout()
    fig.savefig(RESULTS / "vocab_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {RESULTS / 'vocab_trajectory.png'}")

    metrics = {label: sd.to_dict(orient="records") for label, sd in series_data.items()}
    (RESULTS / "vocab_trajectory_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
