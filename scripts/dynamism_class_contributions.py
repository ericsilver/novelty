"""B2: which NICE classes drive the pooled debut-share trend?

Reads paper/results/dynamism_firsttimer_share_byclass.csv (from B1) and
computes each class's contribution to the year-over-year change in the
pooled debut share, decomposed via:

  Δ share_pooled = Σ_c w_c * Δ share_c + Σ_c share_c_lag * Δ w_c

where w_c is class c's share of total filings (a within-class effect plus
a between-class composition effect). We report each class's total
contribution over 2000-2025 (the recovery period after the 1996-2003
trough), ranked by absolute size.

Also reports per-class long-run slopes (1985 -> 2025) for both share and
share_of_total to make the "growing classes" and "shrinking classes"
visible directly.

Outputs:
  paper/results/dynamism_class_contributions.csv
  paper/results/dynamism_class_contributions.tex
  paper/results/dynamism_class_contributions.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES  = REPO / "paper" / "results"


def main() -> int:
    df = pl.read_csv(RES / "dynamism_firsttimer_share_byclass.csv").sort(["cls","year"])

    # Per-class share of total filings, per year
    total_by_year = df.group_by("year").agg(pl.col("n_filings").sum().alias("total_filings"))
    df = df.join(total_by_year, on="year", how="left").with_columns(
        (pl.col("n_filings").cast(pl.Float64) / pl.col("total_filings").cast(pl.Float64)).alias("w"))

    # Class-level long-run regression: share ~ year, w ~ year
    rows = []
    for cls in sorted(df["cls"].unique().to_list()):
        sub = df.filter(pl.col("cls") == cls).sort("year").to_pandas()
        if len(sub) < 5:
            continue
        x = sub["year"].to_numpy().astype(float)
        y_share = sub["share"].to_numpy()
        y_w = sub["w"].to_numpy()
        slope_share = np.polyfit(x, y_share, 1)[0]
        slope_w = np.polyfit(x, y_w, 1)[0]
        mean_w = sub["w"].mean()
        rows.append({
            "cls": cls, "class_name": sub["class_name"].iloc[0],
            "mean_share_of_total": mean_w,
            "slope_share_per_year": slope_share,
            "slope_share_of_total_per_year": slope_w,
            "share_change_total": y_share[-1] - y_share[0],
            "share_first": y_share[0], "share_last": y_share[-1],
            "w_first": y_w[0], "w_last": y_w[-1],
        })
    summary = pl.from_dicts(rows)
    summary = summary.sort("slope_share_per_year", descending=True)
    summary.write_csv(RES / "dynamism_class_contributions.csv")

    print("Top 10 classes by long-run slope of within-class debut share:")
    print(summary.head(10).to_pandas().to_string(index=False))
    print()
    print("Bottom 10 classes:")
    print(summary.tail(10).to_pandas().to_string(index=False))
    print()
    print("Top 10 classes by long-run slope of share-of-total-filings:")
    by_growth = summary.sort("slope_share_of_total_per_year", descending=True)
    print(by_growth.head(10).to_pandas().to_string(index=False))
    print()
    print("Bottom 10 by share-of-total slope:")
    print(by_growth.tail(10).to_pandas().to_string(index=False))

    # Decomposition: contribution to Δ pooled debut share, year-over-year
    df = df.sort(["cls","year"]).with_columns([
        pl.col("share").shift(1).over("cls").alias("share_lag"),
        pl.col("w").shift(1).over("cls").alias("w_lag"),
    ]).with_columns([
        (pl.col("share") - pl.col("share_lag")).alias("d_share"),
        (pl.col("w") - pl.col("w_lag")).alias("d_w"),
    ]).with_columns([
        # within-class effect: lag weight * change in within-class share
        (pl.col("w_lag") * pl.col("d_share")).alias("within_effect"),
        # composition effect: lag within-class share * change in weight
        (pl.col("share_lag") * pl.col("d_w")).alias("comp_effect"),
    ])

    # Aggregate over 2000-2025
    contrib_2000_2025 = (df.filter(pl.col("year").is_between(2001, 2025))
                          .group_by(["cls","class_name"]).agg([
                              pl.col("within_effect").sum().alias("within_sum"),
                              pl.col("comp_effect").sum().alias("comp_sum"),
                              (pl.col("within_effect") + pl.col("comp_effect"))
                                .sum().alias("total_contrib"),
                          ]).sort("total_contrib", descending=True))
    print("\nTop 8 classes by contribution to Δ pooled debut share, 2000-2025:")
    print(contrib_2000_2025.head(8).to_pandas().to_string(index=False))
    print("\nBottom 8 (largest negative contributors):")
    print(contrib_2000_2025.tail(8).to_pandas().to_string(index=False))

    # tex table for headline
    top5 = contrib_2000_2025.head(5).to_pandas()
    bot5 = contrib_2000_2025.tail(5).to_pandas()
    tex = [r"\begin{tabular}{lrr}",
           r"\toprule",
           r"Industry & within ($\Delta$ share) & composition ($\Delta$ weight) \\",
           r"\midrule",
           r"\textbf{Largest positive contributors to debut-share rise, 2001--2025} & & \\"]
    for _, r in top5.iterrows():
        name = r["class_name"].replace("&","\\&")
        tex.append(f"  {name} & {r['within_sum']*100:+.2f} pp & {r['comp_sum']*100:+.2f} pp \\\\")
    tex.append(r"\midrule")
    tex.append(r"\textbf{Largest negative contributors} & & \\")
    for _, r in bot5.iterrows():
        name = r["class_name"].replace("&","\\&")
        tex.append(f"  {name} & {r['within_sum']*100:+.2f} pp & {r['comp_sum']*100:+.2f} pp \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    (RES / "dynamism_class_contributions.tex").write_text("\n".join(tex))

    # Plot: scatter of (slope_share, slope_share_of_total) with class labels
    fig, ax = plt.subplots(figsize=(11, 8))
    s = summary.to_pandas()
    sizes = (s["mean_share_of_total"] * 4000).clip(lower=20)
    ax.scatter(s["slope_share_of_total_per_year"]*100,
               s["slope_share_per_year"]*100,
               s=sizes, alpha=0.5, edgecolor="#333", linewidth=0.5)
    for _, r in s.iterrows():
        ax.annotate(f"{r['cls']}", (r["slope_share_of_total_per_year"]*100,
                                     r["slope_share_per_year"]*100),
                    fontsize=7, alpha=0.7)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Slope of class's share of total filings (pp/year, 1985-2025)")
    ax.set_ylabel("Slope of within-class debut share (pp/year, 1985-2025)")
    ax.set_title(
        "Per-NICE-class trend on two axes\n"
        "x: how fast the class itself has grown as a share of US TM filings\n"
        "y: how fast within-class debut share (debuts/total) has risen\n"
        "(bubble size ~ mean share of total filings)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "dynamism_class_contributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] wrote contributions outputs to {RES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
