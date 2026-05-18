"""For each token of interest, rank firms by their first-ever adoption
date of that token (across all classes), then measure how outcomes vary
with adoption rank.

For each token T we report:
  - top-N named pioneers (the actual first-to-adopt firms)
  - per-rank-bucket survival rate of the firm's T-adoption filing
  - per-rank-bucket P(ever public)  (firm appears in SEC crosswalk)
  - per-rank-bucket median total firm-filings (scale proxy)

Survival uses `passed_5y = reached_registration AND currently_live` on
the firm's earliest T-mentioning filing.  Cohorts beyond 2019 have a
truncated forward window and are flagged.

Outputs:
  paper/results/pioneer_ladder_{token}.csv
  paper/results/pioneer_ladder_{token}_pioneers.txt
  paper/results/pioneer_ladder_summary.png    (one panel per token)
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

DEFAULT_TOKENS = ["website", "wireless", "online retail", "social media", "blockchain"]


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def collect_token_filings(token: str) -> pl.DataFrame:
    """Stream all classes; return per-filing rows where g/s matches token."""
    body = token.replace(" ", r"\s+")
    pat = r"(?i)\b" + body + r"\b"
    parts = []
    for cls in class_list():
        tm_p  = PROC / f"tm_class{cls}.parquet"
        out_p = PROC / f"outcomes_class{cls}.parquit" if False else PROC / f"outcomes_class{cls}.parquet"
        if not out_p.exists():
            continue
        tm = pl.read_parquet(
            tm_p,
            columns=["serial_number", "filing_date", "owner_name", "goods_services"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
            & pl.col("goods_services").is_not_null()
            & pl.col("goods_services").str.contains(pat)
        ).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"),
        ).select("serial_number", "filing_date", "year", "owner_name")
        out = pl.read_parquet(
            out_p,
            columns=["serial_number", "reached_registration", "currently_live"],
        )
        j = tm.join(out, on="serial_number", how="inner").with_columns(
            (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
        )
        if j.height:
            parts.append(j.with_columns(pl.lit(cls).alias("nice_class")))
        del tm, out, j
        gc.collect()
    return pl.concat(parts) if parts else pl.DataFrame()


def all_owner_filing_counts() -> pl.DataFrame:
    """Total trademark filings per owner across all classes (firm scale)."""
    parts = []
    for cls in class_list():
        tm_p = PROC / f"tm_class{cls}.parquet"
        d = pl.read_parquet(tm_p, columns=["owner_name"]).filter(
            pl.col("owner_name").is_not_null()
        ).group_by("owner_name").agg(pl.len().alias("n"))
        parts.append(d)
    return (pl.concat(parts).group_by("owner_name")
              .agg(pl.col("n").sum().alias("total_filings")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default=",".join(DEFAULT_TOKENS),
                    help="comma-separated tokens (use literal phrase for bigrams)")
    ap.add_argument("--top-pioneers", type=int, default=30)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("[scale] counting total filings per owner …", flush=True)
    totals = all_owner_filing_counts()
    print(f"        {totals.height:,} unique owners", flush=True)

    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik", "first_year", "last_year"
    ).rename({"first_year": "sec_first_year", "last_year": "sec_last_year"})
    sec_owners = sec.select("owner_name").unique().with_columns(
        pl.lit(True).alias("in_sec")
    )
    print(f"[sec ] crosswalk: {sec.height:,} matched owners", flush=True)

    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    fig, axes = plt.subplots(len(tokens), 2, figsize=(13, 3.4 * len(tokens)))
    if len(tokens) == 1:
        axes = axes.reshape(1, 2)

    for ti, tok in enumerate(tokens):
        print(f"\n=== {tok} ===", flush=True)
        f = collect_token_filings(tok)
        if f.height == 0:
            print(f"  no filings"); continue
        print(f"  filings containing token: {f.height:,}", flush=True)

        # Firm-level T-adoption: earliest T-filing per owner
        firm = (f.sort(["owner_name", "filing_date"])
                 .group_by("owner_name")
                 .agg([pl.col("filing_date").min().alias("adopt_date"),
                       pl.col("year").min().alias("adopt_year"),
                       pl.len().alias("n_T_filings"),
                       pl.col("serial_number").first().alias("first_serial"),
                       pl.col("passed_5y").first().alias("first_passed_5y")])
                 .sort("adopt_date"))
        # Owner outcomes need to be joined back from the first T-filing's serial
        firm = firm.with_row_index("rank", offset=1)
        firm = firm.join(totals, on="owner_name", how="left")
        firm = firm.join(sec_owners, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False)
        )
        n_firms = firm.height
        n_obs   = firm.filter(pl.col("adopt_year") <= 2021).height
        print(f"  unique firms: {n_firms:,}   (5y-observable cohort: {n_obs:,})", flush=True)

        # Top-N pioneers (named)
        top = firm.head(args.top_pioneers).select(
            "rank", "adopt_year", "owner_name", "first_passed_5y",
            "in_sec", "n_T_filings", "total_filings"
        )
        lines = [f"Top {args.top_pioneers} pioneers of '{tok}'  (n={n_firms:,} firms total)\n",
                 f"{'rank':>4s}  {'yr':>4s}  {'sur':>3s}  {'sec':>3s}  "
                 f"{'nT':>4s}  {'tot':>5s}  owner"]
        for r in top.iter_rows(named=True):
            sur = "Y" if r["first_passed_5y"] else "·"
            sec = "Y" if r["in_sec"] else "·"
            lines.append(f"  {r['rank']:>3d}  {r['adopt_year']:>4d}  "
                         f"{sur:>3s}  {sec:>3s}  "
                         f"{r['n_T_filings']:>4d}  {(r['total_filings'] or 0):>5d}  "
                         f"{(r['owner_name'] or '')[:60]}")
        (OUT / f"pioneer_ladder_{tok.replace(' ','_')}_pioneers.txt").write_text("\n".join(lines))

        # Buckets: 1-10, 11-50, 51-200, 201-1000, 1001-5000, 5001+
        edges = [0, 10, 50, 200, 1000, 5000, n_firms + 1]
        labels = ["1-10", "11-50", "51-200", "201-1k", "1k-5k", "5k+"]
        firm = firm.with_columns(
            pl.col("rank").cut(breaks=edges[1:-1], labels=labels).alias("bucket")
        )

        # Per-bucket survival (only firms with 5y-observable adopt cohort)
        obs = firm.filter(pl.col("adopt_year") <= 2021)
        per_bucket = (obs.group_by("bucket")
                         .agg([pl.len().alias("n"),
                               pl.col("first_passed_5y").cast(pl.Float64).mean().alias("survival_rate"),
                               pl.col("in_sec").cast(pl.Float64).mean().alias("p_public"),
                               pl.col("total_filings").median().alias("median_total_filings"),
                               pl.col("n_T_filings").median().alias("median_T_filings"),
                              ])
                         .sort("bucket"))
        # Re-order buckets by labels
        order = {l: i for i, l in enumerate(labels)}
        per_bucket = per_bucket.with_columns(
            pl.col("bucket").cast(pl.String).map_elements(lambda b: order.get(b, 99), return_dtype=pl.Int32).alias("ord")
        ).sort("ord").drop("ord")
        print("  per-bucket (5y-observable cohort only):")
        for r in per_bucket.iter_rows(named=True):
            print(f"    {str(r['bucket']):>6s}  n={r['n']:>6,}  "
                  f"surv={r['survival_rate']:.3f}  pub={r['p_public']:.3f}  "
                  f"med_total_filings={r['median_total_filings']:.0f}  "
                  f"med_T_filings={r['median_T_filings']:.0f}")
        per_bucket.write_csv(OUT / f"pioneer_ladder_{tok.replace(' ','_')}.csv")

        # Plot for this token: bar of survival and P(public), shared bucket axis
        ax_s, ax_p = axes[ti]
        xs = list(range(len(labels)))
        pb = per_bucket.to_pandas().set_index("bucket").reindex(labels)
        pb["n"] = pb["n"].fillna(0)
        # Survival bars + overall baseline line
        baseline_s = obs["first_passed_5y"].cast(pl.Float64).mean()
        ax_s.bar(xs, pb["survival_rate"], color="#2b6cb0", alpha=0.85, edgecolor="black")
        ax_s.axhline(baseline_s, ls="--", color="#444", lw=1, label=f"all-{tok} mean={baseline_s:.2f}")
        for x, n in zip(xs, pb["n"]):
            n = int(n) if n == n else 0  # NaN guard
            ax_s.text(x, 0.02, f"n={n:,}" if n else "—",
                      ha="center", va="bottom", fontsize=8,
                      color="white" if n else "#666")
        ax_s.set_xticks(xs); ax_s.set_xticklabels(labels)
        ax_s.set_ylabel("5y survival of T-adoption filing")
        ax_s.set_ylim(0, max(0.8, pb["survival_rate"].max() * 1.15))
        ax_s.set_title(f"'{tok}'  ({n_firms:,} firms; 5y-obs: {n_obs:,})")
        ax_s.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_s.grid(axis="y", alpha=0.3)
        # P(public) bars
        baseline_p = obs["in_sec"].cast(pl.Float64).mean()
        ax_p.bar(xs, pb["p_public"], color="#cc4444", alpha=0.85, edgecolor="black")
        ax_p.axhline(baseline_p, ls="--", color="#444", lw=1, label=f"all-{tok} mean={baseline_p:.3f}")
        ax_p.set_xticks(xs); ax_p.set_xticklabels(labels)
        ax_p.set_ylabel("P(ever in SEC EDGAR)")
        ax_p.set_ylim(0, max(0.05, pb["p_public"].max() * 1.2))
        ax_p.set_title(f"Adoption-rank bucket")
        ax_p.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_p.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "pioneer_ladder_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote pioneer_ladder_summary.png and per-token CSVs/pioneer lists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
