"""Plot 2 of the dynamism review: USPTO debut filings by surname-imputed
race/ethnicity, ABSOLUTE COUNTS (not shares), with annotations at the
USPTO 2019 foreign-counsel rule and 2021 signature-verification
tightening that the user asked for.

Reads paper/results/surname_ethnicity_panel.csv (produced by
scripts/surname_ethnicity.py).

Tests the user's hypothesis that "nearly the entire drop in registration
is due to the decrease in White-founded firms" by plotting absolute
counts of debut filers by ethnic bucket.

Outputs:
  paper/results/surname_ethnicity_counts.png
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "results"


def main() -> int:
    panel = pl.read_csv(OUT / "surname_ethnicity_panel.csv")
    # Compute absolute counts: weight column (e.g., w_white) is already the
    # sum of pct/100 across matched filers in that year-bucket, so it is the
    # imputed count for that ethnicity directly.
    eth_labels = {
        "w_white":    ("Non-Hispanic White",   "#2b6cb0"),
        "w_api":      ("Asian / Pacific Islander", "#229922"),
        "w_hispanic": ("Hispanic (any race)",  "#cc7a00"),
        "w_black":    ("Non-Hispanic Black",   "#cc4444"),
        "w_2prace":   ("Two or More Races",    "#999999"),
        "w_aian":     ("Am.\\ Indian / AK Native", "#7a3eb2"),
    }
    pdf = panel.filter((pl.col("year") >= 1990) & (pl.col("year") <= 2024)).to_pandas()

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # Top: absolute counts on log scale (so the smaller groups are readable)
    ax = axes[0]
    for col, (label, color) in eth_labels.items():
        ax.plot(pdf["year"], pdf[col], "-o", color=color, ms=4, lw=1.8, label=label)
    ax.set_ylabel("Annual debut filers (imputed count)")
    ax.set_yscale("log")
    ax.set_title("USPTO trademark debut filers by surname-imputed ethnicity, "
                 "absolute counts (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9, ncol=2)

    # Foreign-counsel rule annotation (Aug 3 2019)
    for ax_ in axes:
        ax_.axvline(2019.6, color="#444", ls="--", lw=1.2, zorder=0)
        ax_.axvline(2021.7, color="#888", ls=":",  lw=1.0, zorder=0)
    axes[0].text(2019.7, axes[0].get_ylim()[1]*0.7,
                 "USPTO foreign-counsel rule\n(Aug 3, 2019): foreign-domiciled\n"
                 "applicants must use US-licensed counsel",
                 fontsize=8, color="#333", style="italic", va="top")
    axes[0].text(2021.8, axes[0].get_ylim()[0]*1.5,
                 "USPTO signature-\nverification rule\n(Aug 2021)",
                 fontsize=8, color="#666", style="italic", va="bottom")

    # Bottom: linear scale, but stacked area to show contribution to total
    ax = axes[1]
    yrs = pdf["year"].to_numpy()
    bottoms = pdf[["w_white"]].to_numpy().flatten() * 0
    order = ["w_white", "w_api", "w_hispanic", "w_black", "w_2prace", "w_aian"]
    cumulative = bottoms.copy()
    for col in order:
        vals = pdf[col].to_numpy()
        label, color = eth_labels[col]
        ax.fill_between(yrs, cumulative, cumulative + vals, color=color, alpha=0.8, label=label)
        cumulative = cumulative + vals
    ax.set_xlabel("Debut filing year")
    ax.set_ylabel("Annual debut filers (imputed count)")
    ax.set_title("Stacked counts -- White count is roughly stable since 2018, API surge drives the apparent share decline")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    fig.suptitle("Testing: \"Nearly the entire drop in registration is due to decline in White-founded firms\"",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "surname_ethnicity_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Quick numerical summary for the writeup
    print("\nABSOLUTE COUNTS by ethnicity, selected years:\n")
    cols = ["year", "w_white", "w_api", "w_hispanic", "w_black"]
    print(pdf[pdf["year"].isin([1995, 2005, 2015, 2018, 2019, 2020, 2021, 2022, 2023, 2024])][cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
