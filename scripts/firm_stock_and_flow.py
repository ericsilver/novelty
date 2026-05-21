"""Plot 1 of the dynamism review: firm STOCK (count of active US firms)
and firm FLOW (annual establishment entry rate), 1978-2023.

Source: Census Business Dynamics Statistics, national time series,
reference_data/bds/bds_national.csv (downloaded from
https://www2.census.gov/programs-surveys/bds/tables/time-series/2023/bds2023.csv).

Outputs:
  paper/results/firm_stock_flow.png
  paper/results/firm_stock_flow.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
BDS = REPO / "reference_data" / "bds" / "bds_national.csv"
OUT = REPO / "paper" / "results"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bds = pl.read_csv(BDS).select(
        "year", "firms", "estabs", "emp", "estabs_entry", "estabs_entry_rate",
        "estabs_exit_rate", "firmdeath_firms"
    )
    pdf = bds.to_pandas().sort_values("year")
    pdf.to_csv(OUT / "firm_stock_flow.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)

    # Top panel: stock of firms
    ax = axes[0]
    ax.plot(pdf["year"], pdf["firms"] / 1e6, "-o", color="#2b6cb0", ms=4, lw=2,
            label="Active US firms (BDS)")
    ax.set_ylabel("Active firms (millions)")
    ax.set_title("Stock: number of US firms in existence, 1978-2023 (Census BDS)")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    # Annotate the secular rise
    ax.annotate(f"{pdf.iloc[0]['firms']/1e6:.2f}M",
                xy=(pdf.iloc[0]["year"], pdf.iloc[0]["firms"]/1e6),
                xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.annotate(f"{pdf.iloc[-1]['firms']/1e6:.2f}M",
                xy=(pdf.iloc[-1]["year"], pdf.iloc[-1]["firms"]/1e6),
                xytext=(-30, -15), textcoords="offset points", fontsize=9)

    # Bottom panel: entry rate (the 12% → 8% story)
    ax = axes[1]
    ax.plot(pdf["year"], pdf["estabs_entry_rate"], "-o", color="#cc4444", ms=4, lw=2,
            label="Establishment entry rate (new estabs / total estabs)")
    ax.plot(pdf["year"], pdf["estabs_exit_rate"], "-s", color="#666666", ms=4, lw=1.5,
            label="Establishment exit rate")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate (% per year)")
    ax.set_title("Flow: US establishment entry and exit rates, 1978-2023 (Census BDS)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    # Annotate peak and recent
    peak_idx = pdf["estabs_entry_rate"].idxmax()
    last_idx = pdf.index[-1]
    ax.annotate(f"peak: {pdf.loc[peak_idx,'estabs_entry_rate']:.1f}% ({int(pdf.loc[peak_idx,'year'])})",
                xy=(pdf.loc[peak_idx, "year"], pdf.loc[peak_idx, "estabs_entry_rate"]),
                xytext=(5, 5), textcoords="offset points", fontsize=9, color="#cc4444")
    # The "8%" referenced in the literature is from the recent trough
    ax.axhspan(8, 12, color="#fee", alpha=0.4, zorder=0)
    ax.text(1995, 14.5, "Decker/Haltiwanger/Jarmin/Miranda (2014):\nentry rate fell from $\\sim$12% (1980s) to $\\sim$8% (2010s)",
            fontsize=9, color="#666", style="italic")

    fig.suptitle("US business dynamism: stock of firms rose 1.6x, entry rate fell ~5pp",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "firm_stock_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT/'firm_stock_flow.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
