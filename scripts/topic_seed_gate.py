"""Does the seed move the estimate, or only the scores?

topic_seed_replicate.py refits the T=200 model under a second seed and reports
how well the two fits' scores agree per filing. Agreement per filing is one
question; whether the paper's estimate changes is another, and the second is
the one a reader cares about. This computes the first-gate lead-quintile
contrast in class 009 under both seeds on the identical sample.

Output: paper/results/topic_seed_gate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLS = "009"
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5


def stats(df, var):
    s = df.sort(["ry", var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over("ry") - 1) * 5 // pl.len().over("ry"))
        .cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"quintiles": p, "lift": p[4] - p[0], "se": se, "t": (p[4] - p[0]) / se}


def main() -> int:
    cols = ["serial_number", "topic_dkl", "topic_kl_vs_past", "topic_kl_vs_future"]
    a = pl.read_parquet(PROC / f"rolling_surprise_class{CLS}_T200.parquet", columns=cols).rename(
        {"topic_dkl": "lead_a", "topic_kl_vs_past": "pa", "topic_kl_vs_future": "fa"})
    b = pl.read_parquet(PROC / f"rolling_surprise_class{CLS}_T200_seed7.parquet", columns=cols).rename(
        {"topic_dkl": "lead_b", "topic_kl_vs_past": "pb", "topic_kl_vs_future": "fb"})
    d = a.join(b, on="serial_number", how="inner").with_columns(
        ((pl.col("pa") + pl.col("fa")) / 2).alias("atyp_a"),
        ((pl.col("pb") + pl.col("fb")) / 2).alias("atyp_b")).filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in ["lead_a", "lead_b", "atyp_a", "atyp_b"]]))
    tm = pl.read_parquet(PROC / f"tm_class{CLS}.parquet",
                         columns=["serial_number", "registration_date"]).with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry")).filter(
        pl.col("ry").is_between(REG_LO, REG_HI))
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    g = d.join(tm, on="serial_number", how="inner").join(ev, on="serial_number", how="left"
        ).with_columns(((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
        ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
                       .fill_null(False).cast(pl.Float64).alias("failed")).unique("serial_number")
    out = {"class": CLS, "T": 200, "n_scored": int(d.height), "n_gate": int(g.height),
           "r_lead": float(d.select(pl.corr("lead_a", "lead_b")).item()),
           "r_atyp": float(d.select(pl.corr("atyp_a", "atyp_b")).item()),
           "seed42": {"lead": stats(g, "lead_a"), "atyp": stats(g, "atyp_a")},
           "seed7": {"lead": stats(g, "lead_b"), "atyp": stats(g, "atyp_b")}}
    for k in ("seed42", "seed7"):
        print(k, {m: f"{100*out[k][m]['lift']:+.2f}pp (SE {100*out[k][m]['se']:.2f})"
                  for m in ("lead", "atyp")}, file=sys.stderr)
    (RES / "topic_seed_gate.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
