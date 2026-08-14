"""Does the gate penalty survive within fast and slow prosecution responders?

The worry the split answers: leading marks might fail the use-proof gate more
often not because their products fail more often, but because their owners are
already disengaging, and a disengaging owner both answers the examiner slowly
and lets the registration lapse. If that were the whole story the penalty would
live between response-speed strata rather than within them.

Response speed is the gap between the first non-final office action mailed to an
application and the applicant's first response to it, taken from the event
record. Applications with no office action, or none answered, are excluded --
the measure is undefined for them, and dropping them is not innocuous, so the
excluded count is reported.

The paper previously carried this split from a term-scored, calendar-year-
anchored build whose artifacts are not in the tree. It is recomputed here on
whichever scoring SURPRISE_SRC names, so it is on the same construction as
every other estimate.

Output: paper/results/response_speed_split.json
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
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5

# Non-final office actions mailed to the applicant, and applicant responses.
OA_CODES = ["CNRT", "GNRT", "GNRN"]
RESP_CODES = ["TROA", "EROI"]
GATE_CODES = ["C8..", "C71T"]


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def first_event(codes: list[str]) -> pl.DataFrame:
    return pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(codes) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())


def contrast(df: pl.DataFrame) -> dict | None:
    if df.height < 3000:
        return None
    s = df.sort(["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("passed").mean().alias("p"),
                            pl.len().alias("n")).sort("q")
    if g.height < 5:
        return None
    v = [float(r["p"]) for r in g.iter_rows(named=True)]
    p1, p5 = v[0], v[4]
    n1, n5 = int(g["n"][0]), int(g["n"][4])
    se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
    return {"n": int(df.height), "pass_rate": float(df["passed"].mean()),
            "quintiles": v, "lift": p5 - p1, "se": se}


def main() -> int:
    oa = first_event(OA_CODES).rename({"d": "oa"})
    rp = first_event(RESP_CODES).rename({"d": "resp"})
    gt = first_event(GATE_CODES).rename({"d": "gate"})
    log(f"[events] {oa.height:,} with an office action, {rp.height:,} with a response")

    parts = []
    for c in CLASSES:
        sp, tp = PROC / f"{SRC}_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner"))
        del tm, sc
        gc.collect()
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()

    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry")).filter(
        pl.col("ry").is_between(REG_LO, REG_HI))
    d = d.join(gt, on="serial_number", how="left").with_columns(
        ((pl.col("gate") - pl.col("rd")).dt.total_days() / 365.25).alias("age"))
    d = d.with_columns(
        (1.0 - ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
         .fill_null(False).cast(pl.Float64)).alias("passed"))
    n_all = d.height

    d = d.join(oa, on="serial_number", how="inner").join(
        rp, on="serial_number", how="inner").with_columns(
        (pl.col("resp") - pl.col("oa")).dt.total_days().alias("lag"))
    d = d.filter(pl.col("lag").is_between(0, 400))
    log(f"[frame] {n_all:,} scored registrations, {d.height:,} with a usable "
        f"office-action/response pair ({100*d.height/n_all:.1f}%)")

    d = d.with_columns(
        ((pl.col("lag").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("sq"))
    med = float(d["lag"].median())
    d = d.with_columns((pl.col("lag") <= med).alias("fast"))

    out = {"scoring": SRC, "n_scored_registrations": n_all,
           "n_with_response": int(d.height),
           "median_response_days": med, "by_speed_quintile": {}, "halves": {}}

    log("\nresponse-speed quintile   gate pass rate")
    for q in range(5):
        sub = d.filter(pl.col("sq") == q)
        out["by_speed_quintile"][str(q + 1)] = {
            "n": int(sub.height),
            "median_lag_days": float(sub["lag"].median()),
            "pass_rate": float(sub["passed"].mean())}
        log(f"  Q{q+1} (median {sub['lag'].median():>4.0f}d)   "
            f"{100*sub['passed'].mean():5.2f}%   n={sub.height:,}")

    log("\nlead contrast in gate PASS, within response-speed half")
    for lab, sub in (("fast", d.filter(pl.col("fast"))),
                     ("slow", d.filter(~pl.col("fast")))):
        r = contrast(sub)
        out["halves"][lab] = r
        if r:
            log(f"  {lab:4}  pass {100*r['pass_rate']:5.2f}%   "
                f"Q5-Q1 {100*r['lift']:+6.2f}pp +/-{196*r['se']:4.2f}  n={r['n']:,}")
    out["pooled"] = contrast(d)
    if out["pooled"]:
        r = out["pooled"]
        log(f"  both  pass {100*r['pass_rate']:5.2f}%   "
            f"Q5-Q1 {100*r['lift']:+6.2f}pp +/-{196*r['se']:4.2f}  n={r['n']:,}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "response_speed_split.json").write_text(json.dumps(out, indent=1))
    log("\n[done] response_speed_split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
