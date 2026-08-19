"""Fit the year-ten renewal profile, which the paper calls a U without fitting it.

Section on the use-proof gate says of the second gate that "pooled across
classes the second-gate profile is a shallow U", and leaves it there. Every
other profile in the paper has since been fitted against linear, quadratic,
exponential and free-vertex forms; this one had not, so the claim rested on a
five-bin eyeball.

The second gate is the year-ten renewal, and unlike the first it is read from
terminal status rather than from a dated cancellation: a year-six death and a
year-ten death share a status code, so the two are separable only by
conditioning on having survived the first. Registrations that cleared the first
gate are scored 1 if renewed (status 800) and 0 if cancelled or expired
(710/900), on 2002--2013 cohorts, which is the window in which a year-ten
outcome is observable.

Output: paper/results/gate2_curve_shape.json
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
sys.path.insert(0, str(REPO / "scripts"))

SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
REG_LO, REG_HI = 2002, 2013
GATE_LO, GATE_HI = 4.0, 8.5
NBINS = 40


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def main() -> int:
    from gate_curve_shapes import fit_shapes

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("gd")
    ).drop_nulls("gd").group_by("serial_number").agg(pl.col("gd").min())

    parts = []
    for c in CLASSES:
        sp, tp = PROC / f"{SRC}_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "registration_date",
                                          "status_code"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite()
            & pl.col("topic_kl_vs_future").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner"))
        del tm, sc
        gc.collect()
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()

    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"),
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("A"),
    ).drop_nulls("rd")
    d = d.filter(pl.col("rd").dt.year().is_between(REG_LO, REG_HI))
    d = d.join(ev, on="serial_number", how="left").with_columns(
        (((pl.col("gd") - pl.col("rd")).dt.total_days() / 365.25)
         .is_between(GATE_LO, GATE_HI, closed="left")).fill_null(False).alias("failed1"))
    # survivors of the first gate only
    d = d.filter(~pl.col("failed1")).with_columns(
        pl.when(pl.col("status_code") == "800").then(1.0)
        .when(pl.col("status_code").is_in(["710", "900"])).then(0.0)
        .otherwise(None).alias("renewed")).drop_nulls("renewed")
    log(f"[frame] {d.height:,} first-gate survivors with a resolved year-ten status; "
        f"renewal rate {100*d['renewed'].mean():.2f}%")

    out = {"scoring": SRC, "nbins": NBINS, "reg_window": [REG_LO, REG_HI],
           "n": int(d.height), "base_renewed": float(d["renewed"].mean()), "axes": {}}
    for axis in ("A", "L"):
        s = d.sort(axis).with_columns(
            ((pl.col(axis).rank("ordinal") - 1) * NBINS // pl.len()).cast(pl.Int32).alias("b"))
        g = s.group_by("b").agg(pl.col(axis).mean().alias("x"),
                                pl.col("renewed").mean().alias("y"),
                                pl.len().alias("n")).sort("b")
        x = g["x"].to_numpy().astype(float)
        y = g["y"].to_numpy().astype(float)
        w = g["n"].to_numpy().astype(float)
        mu, sd = float(d[axis].mean()), float(d[axis].std())
        xz = (x - mu) / (sd if sd else 1.0)
        f = fit_shapes(xz, y, w)
        out["axes"][axis] = {"bins": {"x_z": xz.tolist(), "y": y.tolist(),
                                      "n": w.astype(int).tolist()}, "fits": f}
        v = f.get("_symmetric_vertex_z") if f.get("_best") == "symmetric" \
            else f.get("_quadratic_vertex_z")
        log(f"  {axis}: best={f['_best']:<11} dAIC {f['_aic_gain_over_linear']:6.1f}  "
            f"R2 {f['linear']['r2']:.2f} -> {f[f['_best']]['r2']:.2f}  "
            f"vertex {v if v is None else round(v, 2)}  V={f.get('_symmetric_is_V')}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate2_curve_shape.json").write_text(json.dumps(out, indent=1))
    log("[done] gate2_curve_shape.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
