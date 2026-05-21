"""C1 descriptive: per (NICE class, year) top-N owner concentration.

For each (class, year), compute:
  - top-10-owner share of filings  (concentration of flow)
  - HHI on owner filings within the year (more sensitive)
  - top-10 absolute filing count
  - number of unique owners (denominator for entry-rate diagnostics)

Cross-check: in NICE 043 (Restaurants & Hotels) does top-10 share rise on
the known timeline of major chain expansions? In NICE 042 (Sci & Tech) is
the concentration rise consistent with the within-class debut-share fall
flagged in B2?

Outputs:
  paper/results/dynamism_topN_share.csv
  paper/results/dynamism_topN_share_byclass.png
  paper/results/dynamism_topN_share_summary.txt
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from novelty.industries import name as industry_name

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

YEAR_MIN = 1985
YEAR_MAX = 2025
TOP_N = 10


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    top_owners_by_class = {}
    for cls in class_list():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                              columns=["owner_name","filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(
            pl.col("filing_date").str.slice(0,4).cast(pl.Int64).alias("year")
        ).filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        if tm.height == 0:
            continue
        # Per year stats
        for yr in range(YEAR_MIN, YEAR_MAX+1):
            sub = tm.filter(pl.col("year") == yr)
            if sub.height == 0:
                continue
            owner_counts = (sub.group_by("owner_name").agg(pl.len().alias("n"))
                                .sort("n", descending=True))
            n_total = int(owner_counts["n"].sum())
            n_owners = owner_counts.height
            top = owner_counts.head(TOP_N)
            top_n_count = int(top["n"].sum())
            top_n_share = top_n_count / n_total
            # HHI
            shares = owner_counts["n"].cast(pl.Float64).to_numpy() / n_total
            hhi = float((shares**2).sum())  # in [0,1]; multiply by 10000 for conventional units
            rows.append({
                "cls": cls, "class_name": industry_name(cls), "year": yr,
                "n_filings": n_total, "n_unique_owners": n_owners,
                "top10_count": top_n_count, "top10_share": top_n_share,
                "hhi": hhi * 10000,
            })
        # Track top-5 owners of all time for sanity check
        all_owner = (tm.group_by("owner_name").agg(pl.len().alias("n"))
                       .sort("n", descending=True).head(5).to_pandas())
        top_owners_by_class[cls] = all_owner["owner_name"].tolist()
        del tm
        gc.collect()

    df = pl.from_dicts(rows).sort(["cls","year"])
    df.write_csv(OUT / "dynamism_topN_share.csv")

    # Long-run slope per class
    summary_rows = []
    for cls in sorted(df["cls"].unique().to_list()):
        sub = df.filter(pl.col("cls") == cls).sort("year").to_pandas()
        if len(sub) < 5: continue
        x = sub["year"].to_numpy().astype(float)
        slope_top10 = np.polyfit(x, sub["top10_share"].to_numpy(), 1)[0]
        slope_hhi = np.polyfit(x, sub["hhi"].to_numpy(), 1)[0]
        summary_rows.append({
            "cls": cls, "class_name": industry_name(cls),
            "mean_top10_share": sub["top10_share"].mean(),
            "slope_top10_share_per_year": slope_top10,
            "slope_hhi_per_year": slope_hhi,
            "top10_first": sub["top10_share"].iloc[0],
            "top10_last": sub["top10_share"].iloc[-1],
            "hhi_first": sub["hhi"].iloc[0],
            "hhi_last": sub["hhi"].iloc[-1],
            "top5_alltime": "; ".join(top_owners_by_class.get(cls, [])[:5]),
        })
    summary = pl.from_dicts(summary_rows).sort("slope_top10_share_per_year", descending=True)
    summary.write_csv(OUT / "dynamism_topN_share_summary.csv")

    print("Top 10 classes by rising top-10-owner share:")
    print(summary.head(10).select(["cls","class_name","top10_first","top10_last",
                                     "slope_top10_share_per_year",
                                     "top5_alltime"]).to_pandas().to_string(index=False))
    print()
    print("Bottom 10 (top-10 share most decreasing):")
    print(summary.tail(10).select(["cls","class_name","top10_first","top10_last",
                                     "slope_top10_share_per_year",
                                     "top5_alltime"]).to_pandas().to_string(index=False))

    # Headline cross-check: classes where B2 said debut share fell
    print("\nCross-check vs B2 (within-class debut share falling): "
          "tech-info classes 009, 038, 042, 016")
    sub = summary.filter(pl.col("cls").is_in(["009","038","042","016","035"])).to_pandas()
    print(sub[["cls","class_name","top10_first","top10_last","slope_top10_share_per_year",
                "hhi_first","hhi_last","top5_alltime"]].to_string(index=False))

    txt_lines = [
        "C1: top-N owner concentration descriptives",
        "=" * 60,
        "",
        "Headline numbers (selected classes, 1985 vs 2025):",
        "",
        "Class | Top-10 share 1985 | Top-10 share 2025 | Slope pp/yr | Top-5 owners",
    ]
    for _, r in sub.iterrows():
        txt_lines.append(
            f"  {r['cls']} {r['class_name']:32s} | "
            f"{r['top10_first']*100:5.1f}% | {r['top10_last']*100:5.1f}% | "
            f"{r['slope_top10_share_per_year']*100:+5.3f} | "
            f"{r['top5_alltime'][:80]}"
        )
    (OUT / "dynamism_topN_share_summary.txt").write_text("\n".join(txt_lines))

    # Small-multiples plot: top-10 share over time, all classes
    clses = sorted(df["cls"].unique().to_list())
    n = len(clses); cols = 6; rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*1.9), sharex=True)
    pooled_top = df.group_by("year").agg([
        pl.col("top10_count").sum().alias("top10_count"),
        pl.col("n_filings").sum().alias("n_filings"),
    ]).with_columns(
        (pl.col("top10_count").cast(pl.Float64)/pl.col("n_filings").cast(pl.Float64)).alias("pooled_share")
    ).sort("year").to_pandas()
    for i, cls in enumerate(clses):
        ax = axes[i // cols, i % cols]
        s = df.filter(pl.col("cls") == cls).sort("year").to_pandas()
        ax.plot(s["year"], s["top10_share"], "-", color="#cc4444", lw=1.2)
        ax.plot(pooled_top["year"], pooled_top["pooled_share"], "-", color="#cccccc", lw=0.6, alpha=0.7)
        cname = industry_name(cls)
        ax.set_title(f"{cls} {cname[:24]}", fontsize=7)
        ax.set_ylim(0, max(0.4, s["top10_share"].max()*1.05))
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(alpha=0.2)
    for j in range(n, rows*cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle(
        f"Top-{TOP_N}-owner share of filings by NICE class, {YEAR_MIN}-{YEAR_MAX}.\n"
        f"Red = class-specific; grey = pooled.",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "dynamism_topN_share_byclass.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] wrote topN outputs to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
