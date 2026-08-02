"""High-resolution single-panel version of the beer/soft-drinks quadrant
scatter (the "White Claw plot") for teaching use.

Output: paper/results/quadrant_beer_hires.png  (300 dpi)
        paper/results/quadrant_beer_labeled_points.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

EXAMPLES = [
    ("LIQUID DEATH", 2023),
    ("WHITE CLAW SELTZER WORKS", 2016),
    ("ROCKSTAR", 2002),
    ("HARD MTN DEW", 2021),
    ("FIJI", 2005),
    ("SMARTWATER", 1997),
    ("ESSENTIA", 2008),
    ("KOMBUCHA", 1997),
    ("POWERADE", 1996),
    ("BUDWEISER", 2013),
    ("GATORADE", 1991),
    ("RED BULL", 1995),
    ("MONSTER ENERGY", 2002),
    ("ATHLETIC BREWING", 2017),
    ("CELSIUS", 2004),
]

CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
)


def main() -> int:
    sp = pl.read_parquet(PROC / "surprise_class032.parquet").with_columns(
        (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl")
    ).filter(CLEAN)
    print(f"class 032 clean filings: {sp.height:,}", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    sample = sp.sample(min(20000, sp.height), seed=7)
    ax.scatter(sample["kl_vs_past"], sample["kl_vs_future"],
               s=4, alpha=0.10, color="#888")

    lo, hi = 2.0, 10.0
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.7, linestyle="--")
    ax.text(hi - 0.1, hi - 0.1, "diagonal", fontsize=10,
            ha="right", va="top", color="#444")

    xs_lab, ys_lab, text_objs, rows = [], [], [], []
    for mark_q, year_q in EXAMPLES:
        hits = (
            sp.filter(
                (pl.col("year") == year_q)
                & pl.col("mark_identification").str.to_uppercase().str.contains(
                    f"^{mark_q}$|^{mark_q} ")
            ).sort("dkl", descending=True).head(1)
        )
        if hits.is_empty():
            print(f"  miss: {mark_q} ({year_q})", file=sys.stderr)
            continue
        r = hits.row(0, named=True)
        x, y = float(r["kl_vs_past"]), float(r["kl_vs_future"])
        if not (lo <= x <= hi and lo <= y <= hi):
            print(f"  out of frame: {mark_q} ({year_q}) at ({x:.2f},{y:.2f})",
                  file=sys.stderr)
            continue
        ax.scatter([x], [y], s=70, color="#d62728",
                   edgecolor="black", linewidth=0.8, zorder=4)
        xs_lab.append(x); ys_lab.append(y)
        text_objs.append(ax.text(
            x, y, f"{mark_q} ({year_q})", fontsize=10.5, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88)))
        rows.append({"mark": mark_q, "year": year_q,
                     "kl_vs_past": x, "kl_vs_future": y,
                     "dkl": x - y,
                     "serial_number": r["serial_number"]})
        print(f"  hit : {mark_q} ({year_q}) pros={x:.2f} retr={y:.2f} dkl={x-y:+.2f}",
              file=sys.stderr)

    if text_objs:
        adjust_text(text_objs, x=xs_lab, y=ys_lab, ax=ax,
                    expand=(1.6, 1.8),
                    arrowprops=dict(arrowstyle="-", color="#666", linewidth=0.6,
                                    shrinkA=2, shrinkB=2),
                    only_move={"text": "xy"},
                    force_text=(0.6, 0.8), force_static=(0.4, 0.4), max_move=22)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Prospective KL — novelty at filing time\n"
                  "(higher → less like the previous 5 years' filings in this category)",
                  fontsize=11)
    ax.set_ylabel("Retrospective KL — novelty in hindsight\n"
                  "(higher → less like the next 5 years' filings in this category)",
                  fontsize=11)
    ax.set_title("Beer & Soft Drinks (NICE 032) — USPTO trademark goods/services text",
                 fontsize=13)
    ax.text(0.98, 0.02, "innovator\n(below diagonal:\nfuture resembles\nthis filing)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            color="#117a3a",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#117a3a", alpha=0.75))
    ax.text(0.02, 0.98, "tired\n(above diagonal:\npast resembles\nthis filing)",
            transform=ax.transAxes, va="top", fontsize=10, color="#7a1111",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#7a1111", alpha=0.75))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = RES / "quadrant_beer_hires.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pl.from_dicts(rows).write_csv(RES / "quadrant_beer_labeled_points.csv")
    print(f"[done] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
