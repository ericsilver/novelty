"""What a surge looks like, and how its registrations fared at the five-year proof.

Panel A: the theme's share of its class's filings by filing year for three
class-scale surge episodes and, against the whole corpus, the 1999
online-software episode (theme 337 of the 500-theme model). Shares from the
25% systematic sample the surge screen uses; the marker sits at the surge
year.

Panel B: failure at the five-year proof of continued use for registrations
whose dominant theme was the surging one, filed in the surge window, against
the rest of the same class (for the corpus episode, the rest of the corpus),
with 95% binomial intervals. Numbers from theme_surge_class.json and
theme_surge.json.

Output: paper/results/fig_surges.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "paper" / "results"
CACHE = RES / "theme_novelty_cache"
CLASSES = [f"{i:03d}" for i in range(1, 46)]

EPISODES = [  # (theme, cls or None for corpus, surge_year, label, color)
    (337, None, 1999, "online software, all classes (1999)", "#c0392b"),
    (84, "035", 2002, "education services in Advertising & Retail (2002)", "#2b6cb0"),
    (207, "036", 2011, "charitable fundraising in Finance (2011)", "#16a085"),
    (438, "028", 2012, "electronic game media in Games & Toys (2012)", "#7d3c98"),
]


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    all_ = pl.concat([pl.read_parquet(CACHE / f"{c}.parquet") for c in CLASSES
                      if (CACHE / f"{c}.parquet").exists()])
    thin = all_.filter(pl.col("thin") & pl.col("fy").is_between(1990, 2022))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})
    for theme, cls, sy, label, color in EPISODES:
        base = thin if cls is None else thin.filter(pl.col("cls") == cls)
        tot = base.group_by("fy").len().rename({"len": "N"})
        th = base.filter(pl.col("theme") == theme).group_by("fy").len()
        g = tot.join(th, on="fy", how="left").with_columns(
            (pl.col("len").fill_null(0) / pl.col("N") * 100).alias("share")).sort("fy")
        axA.plot(g["fy"], g["share"], "-", color=color, lw=1.8, label=label)
        row = g.filter(pl.col("fy") == sy)
        if row.height:
            axA.plot([sy], [row["share"][0]], "o", color=color, ms=7,
                     markerfacecolor="none", markeredgewidth=1.8)
    axA.set_xlabel("filing year")
    axA.set_ylabel("% of the class's filings (corpus, for the software episode)")
    axA.legend(fontsize=8, frameon=False, loc="upper left")
    axA.grid(alpha=0.3)

    jc = json.loads((RES / "theme_surge.json").read_text())
    jk = json.loads((RES / "theme_surge_class.json").read_text())
    bars = []
    corp = jc["levels"]["1pct"]["episodes"][0]
    bars.append(("online software\n(1999)", corp["base_fail"], corp["n"],
                 jc["levels"]["1pct"]["fail_rest"], "#c0392b"))
    eps = {(e["theme"], e["cls"]): e for lv in ("1pct", "2pct")
           for e in jk["levels"][lv]["episodes"]}
    for theme, cls, sy, label, color in EPISODES[1:]:
        e = eps[(theme, cls)]
        bars.append((label.split(" in ")[0] + f"\n({sy})", e["base_fail"], e["n"],
                     e["class_fail_rest"], color))
    xs = range(len(bars))
    for i, (lab, p, n, rest, color) in zip(xs, bars):
        se = (p * (1 - p) / n) ** 0.5
        axB.bar(i - 0.19, 100 * p, width=0.38, color=color, alpha=0.85,
                yerr=196 * se, capsize=3)
        axB.bar(i + 0.19, 100 * rest, width=0.38, color="#999999")
    axB.set_xticks(list(xs))
    axB.set_xticklabels([b[0] for b in bars], fontsize=7.5, rotation=18, ha="right")
    axB.set_ylabel("% failing the five-year proof of continued use")
    axB.grid(alpha=0.3, axis="y")
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(color="#555555", label="in the surge window"),
                        Patch(color="#999999", label="rest of the class")],
               fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "fig_surges.png", dpi=150, bbox_inches="tight")
    log("[done] fig_surges.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
