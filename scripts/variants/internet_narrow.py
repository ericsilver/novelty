"""Internet as a theme: do web-based filings diverge from their Nice class?

The internet cohort is the keyword pattern of theme_cohorts.py applied to the
goods/services text, so a web business in apparel is caught and a semiconductor
maker in class 009 is not. For every class this reports, for internet-bearing
filings against the rest of the class:

  registration rate          filings 1995-2018
  first-gate failure         registrations 2002-2018, event-dated, age 4.0-8.5
  mean lead and atypicality  production scoring
  the leading gate penalty   top-minus-bottom lead-quintile contrast, quintiles
                             cut within the group and registration year

and pools the classes three ways: the four technology classes, goods classes,
and service classes, so that the comparison Eric asked for -- web-based
businesses filing in a goods class against that class -- is read directly.

Outputs: paper/results/internet_breakout.json
         paper/results/internet_breakout.png
         paper/results/internet_breakout.tex   (per-class table)
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "v3" / "_eval"
SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
PATTERN = (r"\binternet\b|\bworld wide web\b|\bweb ?sites?\b|\bwebsites?\b|\be-?commerce\b|\belectronic commerce\b")
TECH = {"009", "035", "038", "042"}
SERVICES = {f"{i:03d}" for i in range(35, 46)}
EDGE = "2026-04-02"
GATE_LO, GATE_HI = 4.0, 8.5
NAMES = {}


def log(m): print(m, file=sys.stderr, flush=True)


def contrast(df: pl.DataFrame) -> dict | None:
    if df.height < 3000:
        return None
    s = df.sort(["ry", "L", "serial_number"]).with_columns(
        ((pl.col("L").rank("ordinal").over("ry") - 1) * 5 // pl.len().over("ry")).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    if len(p) < 5:
        return None
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"n": int(df.height), "lift": p[4] - p[0], "se": se}


def rate(df: pl.DataFrame, col: str) -> dict | None:
    if df.height < 500:
        return None
    p = float(df[col].mean())
    return {"n": int(df.height), "rate": p, "se": (p * (1 - p) / df.height) ** 0.5}


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    names = json.loads((RES / "internet_narrow.json").read_text()) if (RES / "internet_narrow.json").exists() else {}

    out = {"pattern": PATTERN, "per_class": {}, "pooled": {}}
    pooled_parts = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"{SRC}_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date",
                                          "goods_services"]).filter(
            pl.col("goods_services").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        ).with_columns(
            pl.col("goods_services").str.to_lowercase().str.contains(PATTERN).alias("web"),
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        ).drop("goods_services").with_columns(
            pl.col("rd").is_not_null().cast(pl.Float64).alias("registered"),
            pl.col("rd").dt.year().alias("ry"))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future",
                                          "topic_dkl"]).filter(pl.col("topic_dkl").is_finite()
            ).with_columns(pl.col("topic_dkl").alias("L"),
                           ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
        d = tm.join(sc.select("serial_number", "L", "A"), on="serial_number", how="left").unique("serial_number")
        d = d.join(ev, on="serial_number", how="left").with_columns(
            ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
        ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
                       .fill_null(False).cast(pl.Float64).alias("failed"), pl.lit(c).alias("cls"))
        reg = d.filter(pl.col("fy").is_between(1995, 2018))
        gate = d.filter(pl.col("ry").is_between(2002, 2018) & pl.col("L").is_finite())
        row = {"name": names.get(c, c), "n_filings": int(reg.height),
               "web_share": float(reg["web"].mean())}
        for lab, flag in (("web", True), ("rest", False)):
            r, g = reg.filter(pl.col("web") == flag), gate.filter(pl.col("web") == flag)
            row[lab] = {"registration": rate(r, "registered"), "gate_failure": rate(g, "failed"),
                        "mean_L": float(g["L"].mean()) if g.height else None,
                        "mean_A": float(g["A"].mean()) if g.height else None,
                        "lead_penalty": contrast(g)}
        out["per_class"][c] = row
        pooled_parts.append(d.select("serial_number", "cls", "web", "fy", "ry", "registered", "failed", "L", "A"))
        w, r_ = row["web"], row["rest"]
        log(f"  [{c}] web {100*row['web_share']:4.1f}%  reg {100*w['registration']['rate'] if w['registration'] else float('nan'):.1f} vs "
            f"{100*r_['registration']['rate'] if r_['registration'] else float('nan'):.1f}  "
            f"gate fail {100*w['gate_failure']['rate'] if w['gate_failure'] else float('nan'):.1f} vs "
            f"{100*r_['gate_failure']['rate'] if r_['gate_failure'] else float('nan'):.1f}")
        del tm, sc, d, reg, gate
        gc.collect()

    allp = pl.concat(pooled_parts)
    del pooled_parts
    groups = {"technology classes": pl.col("cls").is_in(list(TECH)),
              "other service classes": pl.col("cls").is_in(list(SERVICES - TECH)),
              "goods classes": ~pl.col("cls").is_in(list(SERVICES | TECH))}
    for gname, cond in groups.items():
        sub = allp.filter(cond)
        reg = sub.filter(pl.col("fy").is_between(1995, 2018))
        gate = sub.filter(pl.col("ry").is_between(2002, 2018) & pl.col("L").is_finite())
        out["pooled"][gname] = {}
        for lab, flag in (("web", True), ("rest", False)):
            r, g = reg.filter(pl.col("web") == flag), gate.filter(pl.col("web") == flag)
            gq = g.with_columns(pl.concat_str([pl.col("cls"), pl.col("ry")]).alias("cell"))
            out["pooled"][gname][lab] = {
                "registration": rate(r, "registered"), "gate_failure": rate(g, "failed"),
                "mean_L": float(g["L"].mean()), "mean_A": float(g["A"].mean()),
                "lead_penalty": contrast(g)}
        log(f"  pooled {gname}: " + "  ".join(
            f"{lab} reg {100*out['pooled'][gname][lab]['registration']['rate']:.1f} fail {100*out['pooled'][gname][lab]['gate_failure']['rate']:.1f}"
            for lab in ("web", "rest")))

    # Figure: per class, gate failure of web filings vs the rest, ordered by class.
    cls = [c for c in CLASSES if c in out["per_class"]
           and out["per_class"][c]["web"]["gate_failure"] and out["per_class"][c]["rest"]["gate_failure"]]
    xw = [100 * out["per_class"][c]["web"]["gate_failure"]["rate"] for c in cls]
    xr = [100 * out["per_class"][c]["rest"]["gate_failure"]["rate"] for c in cls]
    fig, ax = plt.subplots(figsize=(8, 10))
    ys = list(range(len(cls)))[::-1]
    for y, a, b, c in zip(ys, xr, xw, cls):
        ax.plot([a, b], [y, y], color="#555555", lw=1.2, zorder=1)
        ax.scatter([a], [y], color="#9aa5b1", s=34, zorder=2, edgecolor="#1a1a1a", lw=0.4)
        ax.scatter([b], [y], color="#2b6cb0" if c in TECH else "#c0392b", s=34, zorder=3,
                   edgecolor="#1a1a1a", lw=0.4)
    ax.set_yticks(ys); ax.set_yticklabels([f"{c} {names.get(c, '')}" for c in cls], fontsize=7.5)
    ax.set_xlabel("first-gate failure, % of registrations 2002-2018", fontsize=9)
    ax.scatter([], [], color="#9aa5b1", edgecolor="#1a1a1a", lw=0.4, s=34, label="rest of class")
    ax.scatter([], [], color="#c0392b", edgecolor="#1a1a1a", lw=0.4, s=34, label="internet-bearing filings")
    ax.scatter([], [], color="#2b6cb0", edgecolor="#1a1a1a", lw=0.4, s=34, label="internet-bearing, technology class")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    for s in ax.spines.values():
        s.set_color("#d8d8d8")
    fig.tight_layout()
    fig.savefig(RES / "internet_breakout.png", dpi=150, bbox_inches="tight")

    rows = [r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Class & & Web share & \multicolumn{2}{c}{Registration (\%)} & \multicolumn{2}{c}{Gate failure (\%)} \\",
            r" & & (\%) & web & rest & web & rest \\", r"\midrule"]
    for c in cls:
        r = out["per_class"][c]
        f = lambda x: f"{100*x['rate']:.1f}" if x else "---"
        rows.append(f"{c} & {names.get(c, '')[:28].replace('&', chr(92)+'&')} & {100*r['web_share']:.1f} & {f(r['web']['registration'])} & "
                    f"{f(r['rest']['registration'])} & {f(r['web']['gate_failure'])} & {f(r['rest']['gate_failure'])} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (RES / "internet_breakout.tex").write_text("\n".join(rows) + "\n")
    (RES / "internet_narrow.json").write_text(json.dumps(out, indent=1))
    log("[done] internet_breakout.{json,png,tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
