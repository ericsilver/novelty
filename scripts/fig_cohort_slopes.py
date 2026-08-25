"""Survival at the five-year proof, per cohort, as continuous coefficients.

For each registration cohort, survival is regressed on lead and atypicality
jointly, both standardized within Nice class (and cohort), so each line is
the change in survival, in percentage points, for a one-standard-deviation
change in that axis at the mean of the other. Panels: all classes pooled
(class-demeaned outcome), and software/electronics alone. Settled cohorts
only. OLS standard errors, uncorrected for owner clustering.

Outputs: paper/results/fig_cohort_slopes.png
         paper/results/fig_cohort_slopes.json
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
CLASSES = [f"{i:03d}" for i in range(1, 46)]
EDGE = "2026-04-02"


def log(m): print(m, file=sys.stderr, flush=True)


def cohort_fit(sub):
    """OLS of class-demeaned survival on zL, zA; returns coef and se in pp."""
    y = (sub["surv"] - sub["surv_mean"]).to_numpy()
    X = np.column_stack([np.ones(sub.height), sub["zL"].to_numpy(), sub["zA"].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (len(y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    return (100 * beta[1], 100 * cov[1, 1] ** 0.5, 100 * beta[2], 100 * cov[2, 2] ** 0.5)


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    edge = pl.lit(EDGE).str.strptime(pl.Date, "%Y-%m-%d")
    parts = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")).drop_nulls("rd")
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()
    d = pl.concat(parts).unique("serial_number"); del parts; gc.collect()
    d = d.filter(pl.col("rd").dt.offset_by("3287d") <= edge).with_columns(
        pl.col("rd").dt.year().alias("ry"), pl.col("topic_dkl").alias("L"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        (1.0 - ((pl.col("age") >= 4.0) & (pl.col("age") < 8.5)).fill_null(False).cast(pl.Float64)).alias("surv"))
    d = d.with_columns(
        ((pl.col("L") - pl.col("L").mean().over(["cls", "ry"])) / pl.col("L").std().over(["cls", "ry"])).alias("zL"),
        ((pl.col("A") - pl.col("A").mean().over(["cls", "ry"])) / pl.col("A").std().over(["cls", "ry"])).alias("zA"),
        pl.col("surv").mean().over(["cls", "ry"]).alias("surv_mean")).filter(
        pl.col("zL").is_finite() & pl.col("zA").is_finite())
    log(f"[frame] {d.height:,} settled registrations")

    out = {"panels": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=False)
    for ax, (panel, sub) in zip(axes, (("all", d), ("009", d.filter(pl.col("cls") == "009")))):
        rows = []
        for (ry,), g in sorted(sub.group_by("ry"), key=lambda kv: kv[0][0]):
            if g.height < 2000:
                continue
            bL, sL, bA, sA = cohort_fit(g)
            rows.append({"ry": int(ry), "n": g.height, "bL": bL, "seL": sL, "bA": bA, "seA": sA})
        out["panels"][panel] = rows
        ys = [r["ry"] for r in rows]
        for key, se, color, label in (("bL", "seL", "#c0392b", "lead $L$ (pp per sd)"),
                                      ("bA", "seA", "#2b6cb0", "atypicality $A$ (pp per sd)")):
            v = [r[key] for r in rows]; s = [r[se] for r in rows]
            ax.fill_between(ys, [a - 1.96 * b for a, b in zip(v, s)],
                            [a + 1.96 * b for a, b in zip(v, s)], color=color, alpha=0.15, lw=0)
            ax.plot(ys, v, "-", color=color, lw=1.8, label=label)
        ax.axhline(0, color="#333", lw=0.9)
        ax.set_xlabel("registration cohort")
        ax.set_title({"all": "All classes", "009": "Software & Electronics (009)"}[panel], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("change in survival of the five-year proof (pp per sd)")
    axes[0].legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "fig_cohort_slopes.png", dpi=150, bbox_inches="tight")
    (RES / "fig_cohort_slopes.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
