"""The gate penalty by era and by sector, carried back before the 2002 cut.

The paper's cohort series starts in 2002 and reads as a story about a modern
effect: absent through 2005, positive from 2006. Carrying it back to the first
scoreable cohorts contradicts that. The penalty was two to three times larger
for registrations issuing from 1995-1999 filings than it has ever been since,
disappeared entirely for filings made 2000-2004, and returned at a smaller
magnitude from 2006.

It is also not evenly spread. Splitting on the four classes that carry most
technology filings -- 009 (electrical/software), 035 (business services), 038
(telecommunications), 042 (scientific and computer services) -- against the
other forty-one shows the swing is almost entirely theirs.

This script produces that split by registration cohort and the era summary,
and writes the figure the paper uses to replace the "modern effect" reading.

Cohorts are restricted to registrations whose ninth anniversary is inside the
record, so every reported gate outcome is settled rather than censored.

Outputs:
  paper/results/gate_era_profile.json
  paper/results/gate_era_profile.png
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
EDGE = "2026-04-02"
RESOLVED_AGE = 9.0
WIDE_LO, WIDE_HI = 4.0, 8.5
TECH = {"009", "035", "038", "042"}
ERAS = [("1995-1999", 1995, 1999), ("2000-2004", 2000, 2004),
        ("2005-2007", 2005, 2007), ("2008-2014", 2008, 2014)]

# One categorical pair, assigned by series identity and never recycled.
C_TECH, C_REST = "#2b6cb0", "#c05621"


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def contrast(df: pl.DataFrame, min_n: int = 3000) -> dict | None:
    """Top-minus-bottom lead-quintile contrast in the gate failure rate."""
    if df.height < min_n:
        return None
    s = df.sort(["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"),
                            pl.len().alias("n")).sort("q")
    if g.height < 5:
        return None
    v = [float(r["p"]) for r in g.iter_rows(named=True)]
    p1, p5 = v[0], v[4]
    n1, n5 = int(g["n"][0]), int(g["n"][4])
    se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
    return {"n": int(df.height), "base": float(df["failed"].mean()),
            "lift": p5 - p1, "se": se}


def load() -> pl.DataFrame:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())

    parts = []
    for c in CLASSES:
        sp, tp = PROC / f"{SRC}_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(
            tp, columns=["serial_number", "filing_date", "registration_date"]).filter(
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()

    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    edge = pl.lit(EDGE).str.strptime(pl.Date, "%Y-%m-%d")
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
    ).drop_nulls(["rd", "fd"]).filter(
        pl.col("rd").dt.offset_by(f"{int(RESOLVED_AGE * 365.25)}d") <= edge)
    return d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"),
        pl.col("rd").dt.year().alias("ry"),
        pl.col("fd").dt.year().alias("fy"),
        pl.col("cls").is_in(TECH).alias("tech"),
    ).with_columns(
        ((pl.col("age") >= WIDE_LO) & (pl.col("age") < WIDE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"))


def main() -> int:
    d = load()
    log(f"[frame] {d.height:,} settled registrations")

    out = {"scoring": SRC, "resolved_age": RESOLVED_AGE, "tech_classes": sorted(TECH),
           "by_era_filing_year": {}, "by_reg_cohort": {}}

    log("\nera (filing year)   all              tech             other")
    for label, lo, hi in ERAS:
        sub = d.filter(pl.col("fy").is_between(lo, hi))
        row = {"all": contrast(sub),
               "tech": contrast(sub.filter(pl.col("tech"))),
               "other": contrast(sub.filter(~pl.col("tech")))}
        out["by_era_filing_year"][label] = row

        def f(r):
            return f"{100*r['lift']:+6.2f}+/-{196*r['se']:4.2f}" if r else "     --     "
        log(f"  {label}   {f(row['all'])}   {f(row['tech'])}   {f(row['other'])}")

    years, t_l, t_e, o_l, o_e = [], [], [], [], []
    for ry in sorted(d["ry"].unique().to_list()):
        sub = d.filter(pl.col("ry") == ry)
        rt = contrast(sub.filter(pl.col("tech")))
        ro = contrast(sub.filter(~pl.col("tech")))
        out["by_reg_cohort"][str(ry)] = {"tech": rt, "other": ro,
                                         "all": contrast(sub)}
        if rt and ro:
            years.append(ry)
            t_l.append(100 * rt["lift"]); t_e.append(196 * rt["se"])
            o_l.append(100 * ro["lift"]); o_e.append(196 * ro["se"])

    lo = min(min(a - b for a, b in zip(t_l, t_e)),
             min(a - b for a, b in zip(o_l, o_e))) - 1.5
    hi = max(max(a + b for a, b in zip(t_l, t_e)),
             max(a + b for a, b in zip(o_l, o_e))) + 3.0

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhspan(lo, 0, color="#000000", alpha=0.03, zorder=0)
    ax.axhline(0, color="#4a5568", lw=1.0, zorder=1)
    ax.errorbar(years, t_l, yerr=t_e, fmt="o-", color=C_TECH, lw=2.0, ms=5,
                capsize=0, elinewidth=1.0, zorder=3,
                label="Technology classes (9, 35, 38, 42)")
    ax.errorbar(years, o_l, yerr=o_e, fmt="o-", color=C_REST, lw=2.0, ms=5,
                capsize=0, elinewidth=1.0, zorder=3,
                label="All other classes")
    ax.axvline(2001.5, color="#718096", lw=1.0, ls=":", zorder=2)
    ax.text(2001.7, lo + 0.6, "paper's original lower bound",
            ha="left", va="bottom", fontsize=8, color="#4a5568")
    # Direct-label where the series are furthest apart, not at the terminal
    # points -- the two lines converge at the right edge and the labels collide.
    k = max(range(len(years)), key=lambda i: t_l[i] - o_l[i])
    ax.annotate("Technology", (years[k], t_l[k]), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9,
                color=C_TECH, fontweight="bold")
    ax.annotate("Other", (years[k], o_l[k]), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=9,
                color=C_REST, fontweight="bold")
    ax.set_xlabel("Registration cohort")
    ax.set_ylabel("Gate failure, top minus bottom\nlead quintile (pp)")
    ax.set_xlim(min(years) - 0.6, max(years) + 0.6)
    ax.set_ylim(lo, hi)
    ax.set_xticks([y for y in years if y % 2 == 0])
    ax.set_xticklabels([str(y) for y in years if y % 2 == 0])
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="y", color="#e2e8f0", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#a0aec0")
    fig.tight_layout()
    RES.mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / "gate_era_profile.png", dpi=200)
    plt.close(fig)

    (RES / "gate_era_profile.json").write_text(json.dumps(out, indent=1))
    log("\n[done] gate_era_profile.{json,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
