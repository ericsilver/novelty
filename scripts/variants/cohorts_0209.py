"""The first gate under three theme scorings, on the identical registrations.

  global50   rolling_surprise_class{c}.parquet        (production)
  global500  rolling_surprise_class{c}_T500.parquet   (rolling_rescore_all.py 500)
  perclass50 perclass_surprise_class{c}.parquet       (perclass_lda_rescore.py)

For each scoring: lead and atypicality top-minus-bottom quintile contrasts in
event-dated first-gate failure, quintiles within class and registration year,
pooled over 2002-2018 registrations scored under all three; per-class lead
contrasts; and pairwise correlations of lead and atypicality between scorings.

Output: paper/results/resolution_compare.json
        paper/results/resolution_compare.tex
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "v3" / "_eval"
BASERES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
REG_LO, REG_HI = 2002, 2009
GATE_LO, GATE_HI = 4.0, 8.5
SCORINGS = {"global50": "rolling_surprise_class{c}.parquet",
            "global500": "rolling_surprise_class{c}_T500.parquet",
            "perclass50": "perclass_surprise_class{c}.parquet"}


SCORINGS = {"global50": "rolling_surprise_class{c}.parquet"}

def log(m): print(m, file=sys.stderr, flush=True)


def contrast(df, var, cells):
    s = df.sort(cells + [var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over(cells) - 1) * 5 // pl.len().over(cells)).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"n": int(df.height), "quintiles": p, "lift": p[4] - p[0], "se": se, "t": (p[4] - p[0]) / se}


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    parts, missing = [], []
    for c in CLASSES:
        paths = {k: PROC / v.format(c=c) for k, v in SCORINGS.items()}
        if not all(p.exists() for p in paths.values()):
            missing.append(c); continue
        tm = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["serial_number", "registration_date"]).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")).drop_nulls("rd").with_columns(
            pl.col("rd").dt.year().alias("ry")).filter(pl.col("ry").is_between(REG_LO, REG_HI))
        j = tm
        for k, p in paths.items():
            s = pl.read_parquet(p, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future"]).filter(
                pl.col("topic_kl_vs_past").is_finite() & pl.col("topic_kl_vs_future").is_finite()).with_columns(
                (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias(f"L_{k}"),
                ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias(f"A_{k}")).select(
                "serial_number", f"L_{k}", f"A_{k}")
            j = j.join(s, on="serial_number", how="inner")
        parts.append(j.with_columns(pl.lit(c).alias("cls")))
        del tm, j; gc.collect()
    if missing:
        log(f"[skip] no complete scoring for classes: {missing}")
    x = pl.concat(parts).unique("serial_number"); del parts; gc.collect()
    x = x.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64).alias("failed"))
    log(f"[sample] {x.height:,} registrations scored under all three; base {100*x['failed'].mean():.1f}%")
    out = {"n": int(x.height), "base": float(x["failed"].mean()), "classes_missing": missing, "pooled": {}, "corr": {}, "per_class": {}}
    for k in SCORINGS:
        out["pooled"][k] = {"L": contrast(x, f"L_{k}", ["cls", "ry"]), "A": contrast(x, f"A_{k}", ["cls", "ry"])}
        log(f"  {k:10s} lead {100*out['pooled'][k]['L']['lift']:+.2f}pp (t {out['pooled'][k]['L']['t']:+.1f})  "
            f"atyp {100*out['pooled'][k]['A']['lift']:+.2f}pp (t {out['pooled'][k]['A']['t']:+.1f})")
    ks = list(SCORINGS)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            out["corr"][f"{a}~{b}"] = {"L": float(x.select(pl.corr(f"L_{a}", f"L_{b}")).item()),
                                      "A": float(x.select(pl.corr(f"A_{a}", f"A_{b}")).item())}
            log(f"  corr {a}~{b}: L {out['corr'][f'{a}~{b}']['L']:.3f}  A {out['corr'][f'{a}~{b}']['A']:.3f}")
    for c in sorted(x["cls"].unique().to_list()):
        sub = x.filter(pl.col("cls") == c)
        if sub.height < 3000:
            continue
        out["per_class"][c] = {k: contrast(sub, f"L_{k}", ["ry"])["lift"] for k in SCORINGS}
    agree = sum(1 for c, v in out["per_class"].items() if all(val > 0 for val in v.values()))
    out["per_class_all_positive"] = agree
    out["per_class_n"] = len(out["per_class"])
    log(f"  per-class lead contrast positive under all three: {agree} of {len(out['per_class'])}")
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Scoring & Lead Q5$-$Q1 (pp) & SE & Atypicality Q5$-$Q1 (pp) & SE \\", r"\midrule"]
    names = {"global50": "Global, 50 themes", "global500": "Global, 500 themes", "perclass50": "Per class, 50 themes"}
    for k in SCORINGS:
        L, A = out["pooled"][k]["L"], out["pooled"][k]["A"]
        rows.append(f"{names[k]} & ${100*L['lift']:+.2f}$ & {100*L['se']:.2f} & ${100*A['lift']:+.2f}$ & {100*A['se']:.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (RES / "resolution_compare.tex").write_text("\n".join(rows) + "\n")
    (RES / "cohorts_0209.json").write_text(json.dumps(out, indent=1))
    log("[done] resolution_compare.{json,tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
