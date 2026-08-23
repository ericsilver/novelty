"""Does the internet penalty inside each class start large and converge, on a common shape?

For each Nice class and registration cohort whose gate outcome is settled
(ninth anniversary inside the record), compute the first-gate failure rate of
internet-bearing registrations and of the rest, and their gap. Date the
internet's arrival in each class as the first filing year in which the
internet pattern matches at least ARRIVAL_SHARE of the class's filings. Align
the per-cohort gaps on years since arrival and pool.

Outputs
  paper/results/internet_convergence.json
  paper/results/internet_convergence.png   left: pooled gap by years since
      arrival with a per-class band; right: per-class curves for six named
      classes, calendar time.
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
PATTERN = (r"\binternet\b|\bonline\b|\bon-line\b|\bweb ?sites?\b|\bweb pages?\b|\bwebsites?\b"
           r"|\bworld wide web\b|\be-?commerce\b|\belectronic commerce\b|\bweb portals?\b|\bweb browsers?\b")
ARRIVAL_SHARE = 0.02
EDGE = "2026-04-02"
RESOLVED_AGE = 9.0
GATE_LO, GATE_HI = 4.0, 8.5
MIN_CELL = 100
SHOW = ["039", "007", "001", "025", "036", "009"]


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    names = json.loads((RES / "per_industry_names.json").read_text())
    edge = pl.lit(EDGE).str.strptime(pl.Date, "%Y-%m-%d")

    rows = []
    arrivals = {}
    for c in CLASSES:
        tp = PROC / f"tm_class{c}.parquet"
        if not tp.exists():
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date",
                                          "goods_services"]).filter(
            pl.col("goods_services").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        ).with_columns(
            pl.col("goods_services").str.to_lowercase().str.contains(PATTERN).alias("web"),
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        ).drop("goods_services")
        sh = tm.filter(pl.col("fy").is_between(1985, 2024)).group_by("fy").agg(
            pl.col("web").mean().alias("share")).sort("fy")
        arr = sh.filter(pl.col("share") >= ARRIVAL_SHARE)["fy"]
        arrivals[c] = int(arr.min()) if arr.len() else None
        g = tm.drop_nulls("rd").filter(pl.col("rd").dt.offset_by(f"{int(RESOLVED_AGE*365.25)}d") <= edge
                                       ).join(ev, on="serial_number", how="left").with_columns(
            ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age"),
            pl.col("rd").dt.year().alias("ry")
        ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
                       .fill_null(False).cast(pl.Float64).alias("failed"))
        t = g.group_by(["ry", "web"]).agg(pl.col("failed").mean().alias("p"), pl.len().alias("n"))
        for r in t.iter_rows(named=True):
            rows.append({"cls": c, "ry": int(r["ry"]), "web": bool(r["web"]),
                         "p": float(r["p"]), "n": int(r["n"])})
        del tm, g
        gc.collect()
        log(f"  [{c}] arrival {arrivals[c]}")
    df = pl.DataFrame(rows)
    wide = df.filter(pl.col("web")).rename({"p": "p_web", "n": "n_web"}).drop("web").join(
        df.filter(~pl.col("web")).rename({"p": "p_rest", "n": "n_rest"}).drop("web"),
        on=["cls", "ry"], how="inner").filter(
        (pl.col("n_web") >= MIN_CELL) & (pl.col("n_rest") >= MIN_CELL)).with_columns(
        (pl.col("p_web") - pl.col("p_rest")).alias("gap"),
        ((pl.col("p_web") * (1 - pl.col("p_web")) / pl.col("n_web")
          + pl.col("p_rest") * (1 - pl.col("p_rest")) / pl.col("n_rest")) ** 0.5).alias("se"))
    wide = wide.with_columns(pl.col("cls").replace_strict(arrivals, default=None).alias("arr")
                             ).drop_nulls("arr").with_columns((pl.col("ry") - pl.col("arr")).alias("t"))
    log(f"[cells] {wide.height} class-cohort cells with both groups >= {MIN_CELL}")

    TECH = {"009", "035", "038", "042"}
    SERV = {f"{i:03d}" for i in range(35, 46)} - TECH
    wide = wide.with_columns(
        pl.when(pl.col("cls").is_in(list(TECH))).then(pl.lit("technology"))
        .when(pl.col("cls").is_in(list(SERV))).then(pl.lit("services"))
        .otherwise(pl.lit("goods")).alias("grp"))
    out_groups = {}
    for gname in ("technology", "services", "goods"):
        gsub = wide.filter(pl.col("grp") == gname).group_by("t").agg(
            pl.len().alias("k"), pl.col("gap").mean().alias("gap_eq")).sort("t").filter(pl.col("k") >= 3)
        out_groups[gname] = {str(r["t"]): {"k": r["k"], "gap": r["gap_eq"]} for r in gsub.iter_rows(named=True)}
    pooled = wide.group_by("t").agg(
        pl.len().alias("k"),
        ((pl.col("gap") * pl.col("n_web")).sum() / pl.col("n_web").sum()).alias("gap_w"),
        pl.col("gap").mean().alias("gap_eq"),
        pl.col("gap").std().alias("sd")).sort("t").filter(pl.col("k") >= 5)
    out = {"pattern_share_arrival": ARRIVAL_SHARE, "arrivals": arrivals,
           "n_cells": int(wide.height),
           "pooled_by_event_time": {str(r["t"]): {"k": r["k"], "gap_weighted": r["gap_w"],
                                                  "gap_equal": r["gap_eq"], "sd": r["sd"]}
                                    for r in pooled.iter_rows(named=True)},
           "per_class": {}, "groups": out_groups}
    for c in sorted(wide["cls"].unique().to_list()):
        sub = wide.filter(pl.col("cls") == c).sort("ry")
        out["per_class"][c] = {"arrival": arrivals[c],
                               "series": {str(r["ry"]): {"gap": r["gap"], "se": r["se"],
                                                          "p_web": r["p_web"], "p_rest": r["p_rest"]}
                                          for r in sub.iter_rows(named=True)}}
    # first-vs-late summary per class
    early, late = [], []
    for c, v in out["per_class"].items():
        sub = wide.filter(pl.col("cls") == c)
        e = sub.filter(pl.col("t").is_between(0, 4))["gap"]
        l = sub.filter(pl.col("t") >= 10)["gap"]
        if e.len() >= 2 and l.len() >= 2:
            early.append(float(e.mean())); late.append(float(l.mean()))
    out["classes_with_both_phases"] = len(early)
    out["mean_gap_years_0_4"] = float(np.mean(early)) if early else None
    out["mean_gap_years_10_plus"] = float(np.mean(late)) if late else None
    out["share_classes_gap_shrinks"] = float(np.mean([e > l for e, l in zip(early, late)])) if early else None
    log(f"  classes with both phases {len(early)}: mean gap yrs0-4 {100*np.mean(early):+.1f}pp, "
        f"yrs10+ {100*np.mean(late):+.1f}pp, shrinks in {100*np.mean([e > l for e, l in zip(early, late)]):.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ts = [int(k) for k in out["pooled_by_event_time"]]
    ge = [100 * out["pooled_by_event_time"][str(t)]["gap_equal"] for t in sorted(ts)]
    gw = [100 * out["pooled_by_event_time"][str(t)]["gap_weighted"] for t in sorted(ts)]
    sd = [100 * out["pooled_by_event_time"][str(t)]["sd"] for t in sorted(ts)]
    ts = sorted(ts)
    ax.fill_between(ts, [a - b for a, b in zip(ge, sd)], [a + b for a, b in zip(ge, sd)],
                    color="#c0392b", alpha=0.12, linewidth=0)
    ax.plot(ts, ge, "o-", color="#c0392b", lw=2, ms=4, label="equal-weighted across classes")
    ax.plot(ts, gw, "s--", color="#2b6cb0", lw=1.6, ms=4, label="filing-weighted")
    ax.axhline(0, color="#4a5568", lw=1)
    ax.axvline(0, color="#718096", lw=0.8, ls=":")
    ax.set_xlabel("registration cohort, years since internet reached 2% of the class's filings")
    ax.set_ylabel("gate failure, internet-bearing minus rest (pp)")
    ax.set_title("Pooled across classes, event time", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for c in SHOW:
        if c not in out["per_class"]:
            continue
        s = out["per_class"][c]["series"]
        ys = sorted(int(y) for y in s)
        ax.plot(ys, [100 * s[str(y)]["gap"] for y in ys], "-", lw=1.6,
                label=f"{c} {names.get(c, '')[:18]}")
    ax.axhline(0, color="#4a5568", lw=1)
    ax.set_xlabel("registration cohort")
    ax.set_title("Six classes, calendar time", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "internet_convergence.png", dpi=150, bbox_inches="tight")
    (RES / "internet_convergence.json").write_text(json.dumps(out, indent=1))
    log("[done] internet_convergence.{json,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
