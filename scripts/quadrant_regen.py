"""Regenerate the two-panel quadrant scatter (Figure 2 of the paper) from
the current surprise files, with brand-label hits verified at run time.

Output: paper/results/quadrant.png
        paper/results/quadrant_labeled_points.csv
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

PANELS = [
    # Examples restricted to post-burn-in filings (>= 1995), consistent
    # with the paper's fragility note on pre-1995 contamination.
    ("009", "Software & Electronics", [
        # (mark, filing year, required owner substring). The owner is part of
        # the key because mark text plus year is NOT unique: several of these
        # names have same-year filings by unrelated parties, and matching on
        # text alone silently plotted the wrong company. WINDOWS (1995)
        # resolved to Softblox Incorporated rather than Microsoft and has been
        # dropped entirely; CHATGPT (2022) resolved to Brand Central LLC and
        # UBER (2014) to RYSC Corp, both now pinned to the real owner.
        ("OPENAI", 2016, "OPENAI"), ("CLAUDE", 2023, "ANTHROPIC"),
        ("INSTAGRAM", 2011, "INSTAGRAM"), ("ETHEREUM", 2018, "ETHEREUM"),
        ("KUBERNETES", 2014, "LF PROJECTS"), ("CHATGPT", 2022, "OPENAI"),
        ("UBER", 2010, "UBER TECHNOLOGIES"), ("PHOTOSHOP", 2003, "ADOBE"),
        ("AIRPODS", 2015, "APPLE"), ("KINDLE", 2010, "AMAZON"),
        ("IPOD", 2001, "APPLE"),
    ]),
    ("032", "Beer & Soft Drinks", [
        # RED BULL (1995) resolved to an individual, Robert Horky, rather than
        # Red Bull GmbH; SMARTWATER (1997) to Paul Swartz rather than Energy
        # Brands, whose own filing is 2000. Both are now pinned by owner.
        ("LIQUID DEATH", 2023, "SUPPLYING DEMAND"),
        ("WHITE CLAW SELTZER WORKS", 2016, "MARK ANTHONY"),
        ("ROCKSTAR", 2002, "ROCKSTAR"), ("HARD MTN DEW", 2021, "PEPSICO"),
        ("FIJI", 2005, "FIJI WATER"), ("SMARTWATER", 2000, "ENERGY BRANDS"),
        ("KOMBUCHA", 1997, ""), ("POWERADE", 1996, "COCA-COLA"),
        ("GATORADE", 1996, "STOKELY"), ("RED BULL", 1995, "RED BULL GMBH"),
        ("MONSTER ENERGY", 2002, "MONSTER"), ("CELSIUS", 2004, "ELITE FX"),
    ]),
]

CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
)


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    rows = []
    for ax, (cls, label, examples) in zip(axes, PANELS):
        sp = pl.read_parquet(PROC / f"surprise_class{cls}.parquet").with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl")
        ).filter(CLEAN)
        sample = sp.sample(min(15000, sp.height), seed=7)
        ax.scatter(sample["kl_vs_past"], sample["kl_vs_future"],
                   s=3, alpha=0.08, color="#888")
        lo, hi = 2.0, 10.0
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, linestyle="--")
        ax.text(hi - 0.1, hi - 0.1, "diagonal", fontsize=8,
                ha="right", va="top", color="#444")

        xs_lab, ys_lab, text_objs = [], [], []
        for mark_q, year_q, owner_q in examples:
            cond = ((pl.col("year") == year_q)
                    & pl.col("mark_identification").str.to_uppercase().str.contains(
                        f"^{mark_q}$|^{mark_q} "))
            if owner_q:
                cond = cond & pl.col("owner_name").str.to_uppercase().str.contains(owner_q)
            hits = sp.filter(cond).sort("dkl", descending=True).head(1)
            if hits.is_empty():
                print(f"  miss [{cls}]: {mark_q} ({year_q})", file=sys.stderr)
                continue
            r = hits.row(0, named=True)
            x, y = float(r["kl_vs_past"]), float(r["kl_vs_future"])
            if not (lo <= x <= hi and lo <= y <= hi):
                print(f"  oob  [{cls}]: {mark_q} ({year_q}) ({x:.2f},{y:.2f})",
                      file=sys.stderr)
                continue
            ax.scatter([x], [y], s=46, color="#d62728",
                       edgecolor="black", linewidth=0.7, zorder=4)
            xs_lab.append(x); ys_lab.append(y)
            text_objs.append(ax.text(
                x, y, f"{mark_q} ({year_q})", fontsize=8, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)))
            rows.append({"cls": cls, "mark": mark_q, "year": year_q,
                         "kl_vs_past": x, "kl_vs_future": y, "dkl": x - y})
            print(f"  hit  [{cls}]: {mark_q} ({year_q}) dkl={x-y:+.2f}",
                  file=sys.stderr)
        if text_objs:
            adjust_text(text_objs, x=xs_lab, y=ys_lab, ax=ax,
                        expand=(1.6, 1.8),
                        arrowprops=dict(arrowstyle="-", color="#666", linewidth=0.5,
                                        shrinkA=2, shrinkB=2),
                        only_move={"text": "xy"},
                        force_text=(0.6, 0.8), force_static=(0.4, 0.4), max_move=18)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Prospective KL --- novelty at filing time\n"
                      "(higher $\\rightarrow$ less like the previous 5 years' filings)")
        ax.set_ylabel("Retrospective KL --- novelty in hindsight\n"
                      "(higher $\\rightarrow$ less like the next 5 years' filings)")
        ax.set_title(label)
        ax.text(0.98, 0.02, "innovator\n(below diagonal:\nfuture resembles\nthis filing)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color="#117a3a",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#117a3a", alpha=0.7))
        ax.text(0.02, 0.98, "tired\n(above diagonal:\npast resembles\nthis filing)",
                transform=ax.transAxes, va="top", fontsize=8.5, color="#7a1111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7a1111", alpha=0.7))
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RES / "quadrant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pl.from_dicts(rows).write_csv(RES / "quadrant_labeled_points.csv")
    print("[done]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
