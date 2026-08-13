"""Does the gate result measure a failure rate, or a failure schedule?

The main analysis codes a registration as failing the first gate if a section 8
or section 71 cancellation posts between registration age 4.0 and 8.5 years.
That is an indicator on a window, and a referee is entitled to ask whether the
regressor predicts the window rather than the event: if leading marks fail
earlier than lagging ones, a fixed window will register a rate difference that
is really a timing difference.

The question has teeth here because the narrow-band cohort comparison used to
carry the series past 2016 runs about +0.13pp above the full-window estimate on
identical samples, which can only happen if the failures excluded by the narrow
band are not a random sample of failures.

This script decomposes the two. For registration cohorts that are fully
resolved inside the record it reports, by lead quintile:

  rate    P(cancellation ever posts)          -- window-free, no censoring
  window  P(posts in 4.0-8.5), the paper's outcome
  narrow  P(posts in 6.5-7.0), the cohort-extension outcome
  timing  P(posts at age >= 7.0 | posts at all), the schedule

If the lead contrast is the same across the three rate measures, the result is
about whether marks fail, and the window is innocuous. Any contrast in `timing`
is a genuine schedule effect and is reported alongside rather than folded in.

Cohorts are restricted to those whose ninth anniversary is inside the record,
so `rate` is not censored: 99.1% of all recorded cancellations post before age
8.5 and essentially none before 6.5, so a cohort observed to age nine has its
gate outcome settled.

Output: paper/results/gate_duration.json
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
RESOLVED_AGE = 9.0          # a cohort is settled once it reaches this age
WIDE_LO, WIDE_HI = 4.0, 8.5     # the paper's outcome window
NARROW_LO, NARROW_HI = 6.5, 7.0  # where 96.1% of cancellations post
LATE = 7.0                  # "late" = after the six-month spike


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def quintile_contrast(df: pl.DataFrame, col: str, within_cell: bool) -> dict | None:
    """Q5 - Q1 on `col`, with quintiles cut inside class x cohort if asked."""
    sub = df.filter(pl.col(col).is_not_null())
    if sub.height < 5000:
        return None
    if within_cell:
        sub = sub.sort(["cell", "topic_dkl", "serial_number"]).with_columns(
            ((pl.col("topic_dkl").rank("ordinal").over("cell") - 1) * 5
             // pl.len().over("cell")).cast(pl.Int8).alias("q"))
    else:
        sub = sub.sort(["topic_dkl", "serial_number"]).with_columns(
            ((pl.col("topic_dkl").rank("ordinal") - 1) * 5
             // pl.len()).cast(pl.Int8).alias("q"))
    g = sub.group_by("q").agg(pl.col(col).mean().alias("p"),
                             pl.len().alias("n")).sort("q")
    if g.height < 5:
        return None
    v = [float(r["p"]) for r in g.iter_rows(named=True)]
    p1, p5 = v[0], v[4]
    n1, n5 = int(g["n"][0]), int(g["n"][4])
    se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
    return {"n": int(sub.height), "base": float(sub[col].mean()),
            "quintiles": v, "lift": p5 - p1, "se": se,
            "t": (p5 - p1) / se if se > 0 else None}


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())
    log(f"[events] {ev.height:,} registrations with a gate cancellation")

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

    edge = pl.lit(EDGE).str.strptime(pl.Date, "%Y-%m-%d")
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd")
    # keep only registrations whose gate outcome is settled inside the record
    d = d.filter(
        pl.col("rd").dt.offset_by(f"{int(RESOLVED_AGE * 365.25)}d") <= edge
    ).with_columns(pl.col("rd").dt.year().alias("ry"))
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"))
    d = d.with_columns(
        pl.col("age").is_not_null().cast(pl.Float64).alias("rate"),
        ((pl.col("age") >= WIDE_LO) & (pl.col("age") < WIDE_HI))
        .fill_null(False).cast(pl.Float64).alias("window"),
        ((pl.col("age") >= NARROW_LO) & (pl.col("age") < NARROW_HI))
        .fill_null(False).cast(pl.Float64).alias("narrow"),
        # timing is defined only among registrations that actually failed
        pl.when(pl.col("age").is_not_null())
          .then((pl.col("age") >= LATE).cast(pl.Float64))
          .otherwise(None).alias("late_given_fail"),
        pl.concat_str([pl.col("cls"), pl.col("ry").cast(pl.Utf8)], separator="-").alias("cell"),
    )
    log(f"[frame] {d.height:,} settled registrations, cohorts "
        f"{d['ry'].min()}-{d['ry'].max()}")

    out = {"scoring": SRC, "edge": EDGE, "resolved_age": RESOLVED_AGE,
           "wide": [WIDE_LO, WIDE_HI], "narrow": [NARROW_LO, NARROW_HI],
           "late_threshold": LATE, "pooled": {}, "within_cell": {}, "by_cohort": {}}

    for label, within in (("pooled", False), ("within_cell", True)):
        log(f"\n{label}: outcome            base      Q5-Q1        t")
        for col in ("rate", "window", "narrow", "late_given_fail"):
            r = quintile_contrast(d, col, within)
            out[label][col] = r
            if r:
                log(f"  {col:>18}  {100*r['base']:6.2f}%  {100*r['lift']:+6.2f}pp "
                    f"+/-{1.96*100*r['se']:4.2f}  t={r['t']:+6.1f}")

    log("\ncohort   rate lift        window lift      late|fail lift")
    for ry in sorted(d["ry"].unique().to_list()):
        sub = d.filter(pl.col("ry") == ry)
        row = {c: quintile_contrast(sub, c, False)
               for c in ("rate", "window", "late_given_fail")}
        out["by_cohort"][str(ry)] = row
        if all(row.values()):
            log(f"  {ry}  {100*row['rate']['lift']:+6.2f}pp      "
                f"{100*row['window']['lift']:+6.2f}pp      "
                f"{100*row['late_given_fail']['lift']:+6.2f}pp")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_duration.json").write_text(json.dumps(out, indent=1))
    log("\n[done] gate_duration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
