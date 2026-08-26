"""The 1999 internet cohort on the lead/atypicality quadrants, registration in color.

Internet-bearing class-records filed in 1999 in the four technology classes
(009, 035, 038, 042), plotted on standardized lead (x) and standardized
atypicality (y), z-scores computed within class over all scored 1999
filings of that class so the four classes share axes. Red dots reached
registration; grey did not. Quadrant lines at zero; recognizable companies
labeled (serials verified against owner of record).

Output: paper/results/fig_internet_1999_2x2.png
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
TECH = ["009", "035", "038", "042"]
YEAR = "1999"
PATTERN = json.loads((RES / "internet_breakout.json").read_text())["pattern"]
SEED = 20260825
MAX_DOTS = 7000

# (class, serial, label, xytext offset in points)
LABELS = [
    ("042", "75978469", "GOOGLE", (10, 4)),
    ("009", "75754414", "PAYPAL", (10, -4)),
    ("035", "75669553", "EBAY", (10, 4)),
    ("035", "75642407", "MONSTER.COM", (10, -8)),
    ("035", "75697177", "MAPQUEST", (-74, -6)),
    ("035", "75625470", "DRUGSTORE.COM", (10, 4)),
    ("042", "75772855", "SHUTTERFLY", (10, 4)),
    ("035", "75791794", "BLUE NILE", (-70, 6)),
    ("035", "75737216", "PETS.COM", (10, 6)),
    ("042", "75882670", "KOZMO", (10, -8)),
    ("035", "75668880", "STAMPS.COM", (-84, -6)),
    ("035", "75662416", "AMAZON.COM (1999 filing)", (10, 6)),
]


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    parts = []
    for c in TECH:
        t = pl.read_parquet(PROC / f"tm_class{c}.parquet",
                            columns=["serial_number", "filing_date", "registration_date", "goods_services"]).filter(
            pl.col("filing_date").fill_null("").str.slice(0, 4) == YEAR)
        s = pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                            columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        j = t.join(s, on="serial_number", how="inner").with_columns(
            pl.lit(c).alias("cls"),
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8).alias("reg"),
            pl.col("goods_services").fill_null("").str.to_lowercase().str.contains(PATTERN).alias("web"),
            pl.col("topic_dkl").alias("L"),
            ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
        parts.append(j.drop("goods_services"))
        del t, s, j
        gc.collect()
    d = pl.concat(parts)
    # z within class over ALL scored 1999 filings of the class, then keep web
    d = d.with_columns(
        ((pl.col("L") - pl.col("L").mean().over("cls")) / pl.col("L").std().over("cls")).alias("zL"),
        ((pl.col("A") - pl.col("A").mean().over("cls")) / pl.col("A").std().over("cls")).alias("zA"))
    w = d.filter(pl.col("web"))
    log(f"[frame] {w.height:,} internet-bearing class-records of {d.height:,} scored, {YEAR}, "
        f"reg rate {float(w['reg'].mean()):.3f}")

    lab_keys = {(c, s) for c, s, *_ in LABELS}
    w = w.with_columns(
        pl.struct(["cls", "serial_number"]).map_elements(
            lambda r: (r["cls"], r["serial_number"]) in lab_keys, return_dtype=pl.Boolean).alias("lab"))
    rest = w.filter(~pl.col("lab"))
    if rest.height > MAX_DOTS:
        rest = rest.sample(MAX_DOTS, seed=SEED)

    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    for reg, color, z, lab in ((False, "#b9b9b9", 2, "abandoned"),
                               (True, "#c0392b", 3, "reached registration")):
        sub = rest.filter(pl.col("reg") == reg)
        ax.scatter(sub["zL"], sub["zA"], s=7, color=color, alpha=0.45, lw=0, zorder=z, label=lab)
    for c, serial, label, (ox, oy) in LABELS:
        # look up in the full scored frame: a recognizable internet company is
        # worth labeling even when its description does not match the text rule
        row = d.filter((pl.col("cls") == c) & (pl.col("serial_number") == serial))
        if not row.height:
            log(f"[warn] {label} {c}/{serial} not in frame")
            continue
        r = row.row(0, named=True)
        color = "#c0392b" if r["reg"] else "#666666"
        ax.scatter([r["zL"]], [r["zA"]], s=46, facecolor=color, edgecolor="#111",
                   lw=1.1, zorder=6)
        ax.annotate(label, (r["zL"], r["zA"]), xytext=(ox, oy), textcoords="offset points",
                    fontsize=8.6, zorder=7, color="#111",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#333"))
    ax.axvline(0, color="#333", lw=1.0)
    ax.axhline(0, color="#333", lw=1.0)
    xlim = (-3.4, 3.4)
    ylim = (-3.2, 3.6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    for x, y, txt in ((0.98, 0.985, "leading, atypical"), (0.02, 0.985, "lagging, atypical"),
                      (0.98, 0.015, "leading, typical"), (0.02, 0.015, "lagging, typical")):
        ax.text(x, y, txt, transform=ax.transAxes, fontsize=9, color="#444",
                ha="right" if x > 0.5 else "left", va="top" if y > 0.5 else "bottom")
    ax.set_xlabel("lead $L$, standardized within Nice class (1999 filings)")
    ax.set_ylabel("atypicality $A$, standardized within Nice class (1999 filings)")
    ax.legend(fontsize=9, frameon=False, loc="lower center", markerscale=2.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RES / "fig_internet_1999_2x2.png", dpi=150, bbox_inches="tight")
    log("[done] fig_internet_1999_2x2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
