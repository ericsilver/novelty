"""Is the lead signal just atypicality through a ceiling effect?

|L| <= 2A by construction (both levels are nonnegative), and sd(L) rises
with A, so a gradient in L could be A in disguise. Three checks, on the two
frames the paper's outcomes use:

1. Correlations of A with L and with |L| (registrations 2002-2018; and
   registered debuts 1995-2018).
2. sd(L) by within-class-year decile of A.
3. The lead contrast inside atypicality strata: quintiles of A and of L cut
   independently within Nice class and year; for each A-quintile, the
   outcome by L-quintile. If the L pattern is a ceiling artifact it
   disappears within A strata.

Outcomes: failure at the five-year proof (settled registrations) and
P(owner in SEC EDGAR | registered debut).

Output: paper/results/ceiling_check.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
EDGE = "2026-04-02"


def log(m): print(m, file=sys.stderr, flush=True)


def add_q(d, var, cells, name, k=5):
    return d.with_columns(((pl.col(var).rank("ordinal").over(cells) - 1) * k
                           // pl.len().over(cells)).cast(pl.Int8).alias(name))


def grid(d, outcome):
    out = {}
    for qa in range(5):
        sub = d.filter(pl.col("qA") == qa)
        g = sub.group_by("qL").agg(pl.col(outcome).mean().alias("p"), pl.len().alias("n")).sort("qL")
        p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
        se51 = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5 if len(p) == 5 else None
        out[f"A_q{qa+1}"] = {"pL": p, "n": n, "L_q5_q1": (p[4] - p[0]) if len(p) == 5 else None,
                             "se": se51}
    return out


def main() -> int:
    # --- gate frame: settled registrations ---
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
        ((pl.col("age") >= 4.0) & (pl.col("age") < 8.5)).fill_null(False).cast(pl.Float64).alias("fail"))
    d = d.with_columns(pl.col("L").abs().alias("absL"))
    out = {"gate": {}, "sec": {}}
    out["gate"]["n"] = int(d.height)
    out["gate"]["corr_A_L"] = float(np.corrcoef(d["A"].to_numpy(), d["L"].to_numpy())[0, 1])
    out["gate"]["corr_A_absL"] = float(np.corrcoef(d["A"].to_numpy(), d["absL"].to_numpy())[0, 1])
    dq = d.with_columns(((pl.col("A").rank("ordinal").over(["cls", "ry"]) - 1) * 10
                         // pl.len().over(["cls", "ry"])).cast(pl.Int8).alias("dA"))
    out["gate"]["sdL_by_A_decile"] = [float(v) for v in
                                      dq.group_by("dA").agg(pl.col("L").std()).sort("dA")["L"]]
    dq = add_q(add_q(d, "A", ["cls", "ry"], "qA"), "L", ["cls", "ry"], "qL")
    out["gate"]["L_within_A"] = grid(dq, "fail")
    log("[gate] corr(A,L)={:.3f} corr(A,|L|)={:.3f}".format(out["gate"]["corr_A_L"], out["gate"]["corr_A_absL"]))
    for k, v in out["gate"]["L_within_A"].items():
        log(f"  gate {k}: L Q5-Q1 = {100*v['L_q5_q1']:+.2f}pp (se {100*v['se']:.2f})")
    del d, dq
    gc.collect()

    # --- SEC frame: registered debuts ---
    parts = []
    for cls in CLASSES:
        p = PROC / f"tm_class{cls}.parquet"
        if not p.exists():
            continue
        parts.append(pl.read_parquet(p, columns=["owner_name", "filing_date"]).filter(
            pl.col("owner_name").is_not_null() & (pl.col("filing_date").str.len_chars() == 8)
        ).group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date")))
    debut = pl.concat(parts).group_by("owner_name").agg(pl.col("debut_date").min().alias("debut_date"))
    del parts; gc.collect()
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name").unique().with_columns(
        pl.lit(True).alias("in_sec"))
    parts = []
    for cls in CLASSES:
        tp = PROC / f"rolling_surprise_class{cls}.parquet"
        if not tp.exists():
            continue
        topic = pl.read_parquet(tp).filter(pl.col("topic_dkl").is_finite()
                                           & pl.col("year").is_between(1995, 2018))
        tok = pl.read_parquet(PROC / f"surprise_class{cls}.parquet",
                              columns=["serial_number", "n_terms"]).filter(pl.col("n_terms") >= 3)
        t = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                            columns=["serial_number", "owner_name", "filing_date", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8
        ).join(debut, on="owner_name", how="left").filter(pl.col("filing_date") == pl.col("debut_date"))
        j = topic.join(tok, on="serial_number", how="inner").join(t, on="serial_number", how="inner").join(
            sec, on="owner_name", how="left").with_columns(pl.col("in_sec").fill_null(False).cast(pl.Float64)).select(
            pl.col("year").alias("fy"), pl.lit(cls).alias("cls"),
            "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl", "in_sec")
        if j.height:
            parts.append(j)
        del topic, tok, t, j
        gc.collect()
    e = pl.concat(parts).with_columns(
        pl.col("topic_dkl").alias("L"), pl.col("topic_dkl").abs().alias("absL"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    out["sec"]["n"] = int(e.height)
    out["sec"]["corr_A_L"] = float(np.corrcoef(e["A"].to_numpy(), e["L"].to_numpy())[0, 1])
    out["sec"]["corr_A_absL"] = float(np.corrcoef(e["A"].to_numpy(), e["absL"].to_numpy())[0, 1])
    eq = e.with_columns(((pl.col("A").rank("ordinal").over(["cls", "fy"]) - 1) * 10
                         // pl.len().over(["cls", "fy"])).cast(pl.Int8).alias("dA"))
    out["sec"]["sdL_by_A_decile"] = [float(v) for v in
                                     eq.group_by("dA").agg(pl.col("L").std()).sort("dA")["L"]]
    eq = add_q(add_q(e, "A", ["cls", "fy"], "qA"), "L", ["cls", "fy"], "qL")
    out["sec"]["L_within_A"] = grid(eq, "in_sec")
    log("[sec] corr(A,L)={:.3f} corr(A,|L|)={:.3f}".format(out["sec"]["corr_A_L"], out["sec"]["corr_A_absL"]))
    for k, v in out["sec"]["L_within_A"].items():
        log(f"  sec {k}: rates by L quintile " + " ".join(f"{100*x:.3f}" for x in v["pL"])
            + f"  Q5-Q1 {100*v['L_q5_q1']:+.3f}pp (se {100*v['se']:.3f})")
    (RES / "ceiling_check.json").write_text(json.dumps(out, indent=1))
    log("[done] ceiling_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
