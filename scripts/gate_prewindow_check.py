"""Does the gate penalty really begin in 2006, or does the 2002 cut hide it?

The paper reports the first-gate penalty by registration cohort from 2002 and
describes it as modern: absent or negative for 2002-2005, positive from 2006.
The lower bound is described as conventional and inconsequential. It is not.
Carried back, registration cohorts 1996-2000 show contrasts of +6.9 to +10.2pp,
two to four times the modern effect and with the opposite implication.

Before that can be believed it has to survive one obvious confound. Scores are
only produced for filings from 1995 on, so a registration in cohort 1996 must
have issued from a 1995 or 1996 filing -- a pendency of a year or less, which
is far faster than typical. Early registration cohorts are therefore selected
on prosecution speed, and speed plausibly travels with writing goods
descriptions in standard language, which is the regressor. The apparent early
penalty could be entirely that selection.

Two checks separate them:

  1. Index cohorts by FILING year rather than registration year. A filing-year
     cohort contains its registrations at every pendency, so the speed
     selection disappears except through the scoring floor itself.
  2. Hold pendency fixed. Re-run the registration-cohort series inside a common
     pendency band, so fast and slow prosecutions are compared like with like.

If the pre-2002 contrasts survive both, the "modern" reading is wrong and the
lower bound is load-bearing. If they collapse, the 2002 cut is doing real work
and the paper should say so rather than calling it conventional.

Output: paper/results/gate_prewindow_check.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
EDGE = "2026-04-02"
RESOLVED_AGE = 9.0
WIDE_LO, WIDE_HI = 4.0, 8.5
PEND_LO, PEND_HI = 1.0, 3.0     # common pendency band, years filing -> registration


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def contrast(df: pl.DataFrame) -> dict | None:
    if df.height < 5000:
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


def main() -> int:
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
        parts.append(tm.join(sc, on="serial_number", how="inner"))
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
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"),
        ((pl.col("rd") - pl.col("fd")).dt.total_days() / 365.25).alias("pendency"),
        pl.col("rd").dt.year().alias("ry"),
        pl.col("fd").dt.year().alias("fy"),
    ).with_columns(
        ((pl.col("age") >= WIDE_LO) & (pl.col("age") < WIDE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"))
    log(f"[frame] {d.height:,} settled registrations")

    out = {"scoring": SRC, "pendency_band": [PEND_LO, PEND_HI],
           "pendency_by_reg_cohort": {}, "by_reg_year": {},
           "by_filing_year": {}, "by_reg_year_fixed_pendency": {}}

    log("\nreg-yr  median pendency (yrs)   n")
    for ry in sorted(d["ry"].unique().to_list()):
        sub = d.filter(pl.col("ry") == ry)
        med = float(sub["pendency"].median())
        out["pendency_by_reg_cohort"][str(ry)] = {"median": med, "n": int(sub.height)}
        if ry <= 2012:
            log(f"  {ry}    {med:5.2f}                 {sub.height:>9,}")

    band = d.filter(pl.col("pendency").is_between(PEND_LO, PEND_HI))
    log(f"\n[band] {band.height:,} registrations with pendency in "
        f"[{PEND_LO}, {PEND_HI}] years")

    log("\n        by reg year        by filing year     reg year, pendency fixed")
    keys = sorted(set(d["ry"].unique().to_list()) | set(d["fy"].unique().to_list()))
    for y in keys:
        a = contrast(d.filter(pl.col("ry") == y))
        b = contrast(d.filter(pl.col("fy") == y))
        c = contrast(band.filter(pl.col("ry") == y))
        out["by_reg_year"][str(y)] = a
        out["by_filing_year"][str(y)] = b
        out["by_reg_year_fixed_pendency"][str(y)] = c
        if not (a or b or c):
            continue

        def f(r):
            return (f"{100*r['lift']:+6.2f}pp+/-{1.96*100*r['se']:4.2f} (n={r['n']:>7,})"
                    if r else "            --            ")
        log(f"  {y}  {f(a)}  {f(b)}  {f(c)}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_prewindow_check.json").write_text(json.dumps(out, indent=1))
    log("\n[done] gate_prewindow_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
