"""How many filings share a score, and does the tie-breaking rule matter?

Quintile cut points are fixed by sorting on the score and then on the serial
number. That is reproducible, but it is arbitrary wherever a tie group straddles
a cut point, and it is systematic rather than random: serial numbers run in
filing order, so a serial tiebreak sorts tied filings by when they were filed.

The paper had described exact ties as rare under per-filing reference windows.
They are not. Filings that share a class, a filing date and a goods/services
description -- the boilerplate case, and the common one for a single owner
filing a family of marks at once -- get identical topic distributions and
identical reference windows, so they get identical scores. This script measures
how often that happens inside the unit the quintiles are actually cut in
(class x registration cohort) and then bounds what the rule costs.

Three assignments are compared on the same sample:

  serial     the paper's rule: sort by score, then serial number
  reversed   sort by score, then serial number descending -- the opposite
             systematic assignment, so the gap to `serial` bounds the swing
             any serial-ordered rule can produce
  random     sort by score, then a fixed pseudorandom key, repeated over
             several seeds to give the spread under non-systematic breaking

Output: paper/results/tie_rule_check.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
SEEDS = (11, 23, 37, 59, 71)


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def contrast(d: pl.DataFrame, order: pl.Expr, desc: list[bool]) -> tuple[float, float]:
    """Q5 - Q1 in gate failure, quintiles cut within class x cohort."""
    s = d.sort(["cell", "topic_dkl", "tb"], descending=[False, False, desc[0]])
    s = s.with_columns(
        ((pl.col("topic_dkl").rank("ordinal").over("cell") - 1) * 5
         // pl.len().over("cell")).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"),
                            pl.len().alias("n")).sort("q")
    v = [float(r["p"]) for r in g.iter_rows(named=True)]
    p1, p5 = v[0], v[4]
    n1, n5 = int(g["n"][0]), int(g["n"][4])
    return p5 - p1, ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5


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
        tm = pl.read_parquet(tp, columns=["serial_number", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()

    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry")).filter(
        pl.col("ry").is_between(REG_LO, REG_HI))
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"))
    d = d.with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"),
        pl.concat_str([pl.col("cls"), pl.col("ry").cast(pl.Utf8)], separator="-").alias("cell"),
    )
    n = d.height
    log(f"[frame] {n:,} scored registrations, cohorts {REG_LO}-{REG_HI}")

    # tie prevalence inside the unit the cuts are made in
    grp = d.group_by(["cell", "topic_dkl"]).agg(pl.len().alias("k"))
    tied = int(grp.filter(pl.col("k") > 1)["k"].sum())
    biggest = int(grp["k"].max())
    log(f"[ties] {tied:,} of {n:,} registrations ({100*tied/n:.2f}%) share their "
        f"score with at least one other filing in the same class x cohort; "
        f"largest tie group {biggest:,}")

    out = {"scoring": SRC, "n": n, "tied_n": tied, "tied_share": tied / n,
           "largest_group": biggest, "reg_window": [REG_LO, REG_HI],
           "assignments": {}}

    d = d.with_columns(pl.col("serial_number").cast(pl.Utf8).alias("tb"))
    base, se = contrast(d, pl.col("tb"), [False])
    out["assignments"]["serial"] = {"lift": base, "se": se}
    log(f"\n  serial (paper's rule)   {100*base:+6.3f}pp  (SE {100*se:.3f})")

    rev, _ = contrast(d, pl.col("tb"), [True])
    out["assignments"]["serial_reversed"] = {"lift": rev, "delta": rev - base}
    log(f"  serial reversed         {100*rev:+6.3f}pp  "
        f"(delta {100*(rev-base):+.3f}pp)")

    rnd = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        dd = d.with_columns(pl.Series("tb", rng.random(n).astype(str)))
        lift, _ = contrast(dd, pl.col("tb"), [False])
        rnd.append(lift)
        del dd
        gc.collect()
    out["assignments"]["random"] = {
        "seeds": list(SEEDS), "lifts": rnd,
        "min": min(rnd), "max": max(rnd),
        "max_abs_delta": max(abs(r - base) for r in rnd)}
    log(f"  random, {len(SEEDS)} seeds        "
        f"{100*min(rnd):+6.3f} to {100*max(rnd):+6.3f}pp  "
        f"(max delta {100*max(abs(r-base) for r in rnd):+.3f}pp)")

    swing = max([abs(rev - base)] + [abs(r - base) for r in rnd])
    out["max_swing"] = swing
    log(f"\n  worst case across all rules: {100*swing:.3f}pp on a "
        f"{100*base:.2f}pp contrast ({100*swing/abs(base):.1f}% of it)")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "tie_rule_check.json").write_text(json.dumps(out, indent=1))
    log("\n[done] tie_rule_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
