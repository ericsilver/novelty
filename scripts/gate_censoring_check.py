"""How far forward can the gate result actually be carried, and is 2019 sound?

Gate cancellations do not spread across the 4.0-8.5 year window the analysis
uses: 96.1% post between age 6.5 and 7.0, and essentially none (17 of 1.35M) before 6.5. A cohort is
therefore only near-complete once reg_date + 7.0y is inside the record, which
ends 2026-04-02. That puts the last usable cohort at 2019, and it makes the
"fully elapsed" test used elsewhere (reg + 6.5y <= cut) too lenient: for a
registration admitted right at that boundary, only the 5% of failures that post
before age 6.5 are observable.

This script re-estimates the cohort lift on a NARROW window that is fully
observed for every cohort it reports -- failures at age 6.5-7.0, where 96.1% of them
post -- so cohorts are compared on identical exposure. It misses the 3.9% that
post after age seven, so its base rate is slightly below the true gate failure
rate, but it is unbiased across cohorts.

Output: paper/results/gate_censoring_check.json
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
NARROW_LO, NARROW_HI = 6.5, 7.0     # where 96.1% of gate cancellations post
WIDE_LO, WIDE_HI = 4.0, 8.5         # the paper's window


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


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
        parts.append(tm.join(sc, on="serial_number", how="inner"))
        del tm, sc
        gc.collect()

    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry"))
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"))

    edge = pl.lit(EDGE).str.strptime(pl.Date, "%Y-%m-%d")
    d = d.with_columns(
        # exposure: is this registration's narrow window entirely inside the record?
        ((pl.col("rd").dt.offset_by(f"{int(NARROW_HI*365.25)}d")) <= edge).alias("narrow_ok"),
        ((pl.col("rd").dt.offset_by(f"{int(WIDE_HI*365.25)}d")) <= edge).alias("wide_ok"),
        ((pl.col("age") >= NARROW_LO) & (pl.col("age") < NARROW_HI))
        .fill_null(False).cast(pl.Float64).alias("fail_narrow"),
        ((pl.col("age") >= WIDE_LO) & (pl.col("age") < WIDE_HI))
        .fill_null(False).cast(pl.Float64).alias("fail_wide"),
    )
    log(f"[frame] {d.height:,} scored registrations")

    def lift(sub: pl.DataFrame, col: str) -> dict | None:
        if sub.height < 5000:
            return None
        s = sub.sort(["topic_dkl", "serial_number"]).with_columns(
            ((pl.col("topic_dkl").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
        g = s.group_by("q").agg(pl.col(col).mean().alias("p"), pl.len().alias("n")).sort("q")
        v = [float(r["p"]) for r in g.iter_rows(named=True)]
        if len(v) < 5:
            return None
        p1, p5 = v[0], v[4]
        n1 = int(g["n"][0]); n5 = int(g["n"][4])
        se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
        return {"n": int(sub.height), "base": float(sub[col].mean()),
                "q": v, "lift": p5 - p1, "se": se}

    out = {"scoring": SRC, "edge": EDGE,
           "narrow_window": [NARROW_LO, NARROW_HI], "wide_window": [WIDE_LO, WIDE_HI],
           "note": "narrow window captures 96.1% of gate failures; "
                   "only cross-cohort comparison of the lift is meaningful",
           "by_cohort": {}}

    log("\ncohort   n        narrow-window lift (fully observed)   wide-window lift (censored after 2018)")
    for ry in range(2010, 2022):
        sub = d.filter((pl.col("ry") == ry) & pl.col("narrow_ok"))
        nar = lift(sub, "fail_narrow")
        wsub = d.filter((pl.col("ry") == ry) & pl.col("wide_ok"))
        wid = lift(wsub, "fail_wide")
        out["by_cohort"][str(ry)] = {"narrow": nar, "wide": wid}
        ns = (f"{nar['lift']*100:+5.2f}pp +/- {1.96*nar['se']*100:4.2f} "
              f"(base {nar['base']*100:4.1f}%, n={nar['n']:,})") if nar else "  n/a (not fully observed)"
        ws = (f"{wid['lift']*100:+5.2f}pp (base {wid['base']*100:4.1f}%)") if wid else "n/a"
        log(f"  {ry}   {ns}   {ws}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_censoring_check.json").write_text(json.dumps(out, indent=1))
    log("\n[done] gate_censoring_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
