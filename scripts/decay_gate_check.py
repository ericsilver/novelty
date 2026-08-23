"""Does the decayed reference window change the gate estimate?

Section 5 states the flat-window result and must state the decayed one beside
it. This computes the first-gate lead and atypicality contrasts, quintiles
within class and registration year, on the identical registrations under the
flat per-filing scoring (rolling_surprise_*) and the two-year half-life
scoring (decay_surprise_*), and the correlation between the two scorings.

Output: paper/results/decay_gate_check.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5


def log(m): print(m, file=sys.stderr, flush=True)


def contrast(df, var):
    s = df.sort(["cls", "ry", var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over(["cls", "ry"]) - 1) * 5 // pl.len().over(["cls", "ry"])).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"quintiles": p, "lift": p[4] - p[0], "se": se}


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    parts = []
    for c in CLASSES:
        fp, dp, tp = (PROC / f"rolling_surprise_class{c}.parquet", PROC / f"decay_surprise_class{c}.parquet",
                      PROC / f"tm_class{c}.parquet")
        if not (fp.exists() and dp.exists() and tp.exists()):
            continue
        cols = ["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"]
        f = pl.read_parquet(fp, columns=cols).rename({"topic_kl_vs_past": "fp", "topic_kl_vs_future": "ff"})
        d = pl.read_parquet(dp, columns=cols).rename({"topic_kl_vs_past": "dp", "topic_kl_vs_future": "df"})
        tm = pl.read_parquet(tp, columns=["serial_number", "registration_date"]).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")).drop_nulls("rd")
        j = f.join(d, on="serial_number", how="inner").join(tm, on="serial_number", how="inner").with_columns(
            pl.col("rd").dt.year().alias("ry"), pl.lit(c).alias("cls")).filter(
            pl.col("ry").is_between(REG_LO, REG_HI)
            & pl.all_horizontal([pl.col(k).is_finite() for k in ("fp", "ff", "dp", "df")]))
        parts.append(j)
        del f, d, tm, j
        gc.collect()
    x = pl.concat(parts).unique("serial_number"); del parts; gc.collect()
    x = x.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64).alias("failed"),
        (pl.col("fp") - pl.col("ff")).alias("L_flat"), ((pl.col("fp") + pl.col("ff")) / 2).alias("A_flat"),
        (pl.col("dp") - pl.col("df")).alias("L_decay"), ((pl.col("dp") + pl.col("df")) / 2).alias("A_decay"))
    out = {"n": int(x.height), "base": float(x["failed"].mean()),
           "corr_L": float(x.select(pl.corr("L_flat", "L_decay")).item()),
           "corr_A": float(x.select(pl.corr("A_flat", "A_decay")).item()),
           "gate": {k: contrast(x, k) for k in ("L_flat", "L_decay", "A_flat", "A_decay")}}
    for k, v in out["gate"].items():
        log(f"  {k:8s} Q5-Q1 {100*v['lift']:+.2f}pp (SE {100*v['se']:.2f})")
    log(f"  corr L {out['corr_L']:.3f}  corr A {out['corr_A']:.3f}  n={out['n']:,}")
    (RES / "decay_gate_check.json").write_text(json.dumps(out, indent=1))
    log("[done] decay_gate_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
