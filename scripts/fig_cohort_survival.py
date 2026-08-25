"""Survival at the five-year proof of continued use, by registration cohort and group, as lines with bands.

Two panels: all classes pooled, and Software & Electronics (009). In each,
survival (1 - event-dated cancellation for non-use at age 4.0-8.5) by
registration cohort for four groups cut within Nice class and cohort: the
most leading fifth of lead, the most lagging fifth, the most atypical fifth,
and the least atypical fifth. 95% binomial bands, settled cohorts only
(ninth anniversary inside the record). The first cohort whose leading and
lagging bands separate answers how many years of data the contrast needs.

Outputs: paper/results/fig_cohort_survival.png
         paper/results/fig_cohort_survival.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
EDGE = "2026-04-02"
RESOLVED_AGE = 9.0
GATE_LO, GATE_HI = 4.0, 8.5


def log(m): print(m, file=sys.stderr, flush=True)


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
    d = d.filter(pl.col("rd").dt.offset_by(f"{int(RESOLVED_AGE*365.25)}d") <= edge).with_columns(
        pl.col("rd").dt.year().alias("ry"), pl.col("topic_dkl").alias("L"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        (1.0 - ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64)).alias("surv"))
    for var in ("L", "A"):
        d = d.with_columns(((pl.col(var).rank("ordinal").over(["cls", "ry"]) - 1) * 5
                            // pl.len().over(["cls", "ry"])).cast(pl.Int8).alias(f"q{var}"))
    log(f"[frame] {d.height:,} settled registrations")

    GROUPS = [("most leading fifth", pl.col("qL") == 4, "#c0392b"),
              ("most lagging fifth", pl.col("qL") == 0, "#e67e22"),
              ("most atypical fifth", pl.col("qA") == 4, "#2b6cb0"),
              ("least atypical fifth", pl.col("qA") == 0, "#16a085")]
    out = {"panels": {}}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)
    for ax, (panel, sub) in zip(axes, (("all", d), ("009", d.filter(pl.col("cls") == "009")))):
        out["panels"][panel] = {}
        for label, cond, color in GROUPS:
            g = sub.filter(cond).group_by("ry").agg(pl.col("surv").mean().alias("p"), pl.len().alias("n")).sort("ry")
            g = g.filter(pl.col("n") >= 300)
            ys = [int(v) for v in g["ry"]]
            p = [float(v) for v in g["p"]]
            se = [float((a * (1 - a) / n) ** 0.5) for a, n in zip(g["p"], g["n"])]
            out["panels"][panel][label] = {"years": ys, "p": p, "se": se}
            ax.fill_between(ys, [100 * (a - 1.96 * b) for a, b in zip(p, se)],
                            [100 * (a + 1.96 * b) for a, b in zip(p, se)], color=color, alpha=0.12, lw=0)
            ax.plot(ys, [100 * v for v in p], "-", color=color, lw=1.8, label=label)
        ax.set_xlabel("registration cohort")
        ax.set_title({"all": "All classes", "009": "Software & Electronics (009)"}[panel], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% surviving the five-year proof of continued use")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(RES / "fig_cohort_survival.png", dpi=150, bbox_inches="tight")
    # first cohort where leading and lagging bands separate (all-classes panel)
    a = out["panels"]["all"]
    lead, lag = a["most leading fifth"], a["most lagging fifth"]
    sep = None
    for y in lead["years"]:
        if y in lag["years"]:
            i, j = lead["years"].index(y), lag["years"].index(y)
            if lead["p"][i] + 1.96 * lead["se"][i] < lag["p"][j] - 1.96 * lag["se"][j]:
                sep = y
                break
    out["first_cohort_bands_separate_all"] = sep
    log(f"[sep] first cohort with non-overlapping leading/lagging bands (all): {sep}")
    (RES / "fig_cohort_survival.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
