"""Outcomes for exact-copy filings, their originals, and their neighbours.

Text categories, per class-record 1995-2018 (normalized-text clusters
recomputed with first-filer identity):

  unique          text appears once in the class
  manual          cluster of >= 100 identical filings (institutional language)
  copy_small      word-for-word repeat of a DIFFERENT owner's EARLIER text,
                  cluster < 100 (the appropriation/imitation candidates)
  original_copied first filing of a 2-99 cluster that other owners later
                  repeat (the imitated)
  own_reuse       later filing of the owner's own earlier text (house reuse)
  other_dup       remainder (same-day ties, single-owner clusters' tails)

Outcomes per category: registration rate; failure at the five-year proof
(registrations 2002-2018, event-dated); owner ever in SEC reporting and
IPO marker (registered debuts 1995-2018). Raw rates plus within
class x filing-year (or registration-year) demeaned deltas. Also each
category's mean lead and atypicality, standardized within class-year --
copies should read as lagging and imitated originals as leading if the
measure captures literal adoption.

Output: paper/results/copy_outcomes.json
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
FY_LO, FY_HI = 1995, 2018
EDGE_MANUAL = 100


def log(m): print(m, file=sys.stderr, flush=True)


def load_class(c):
    tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
    if not (tp.exists() and sp.exists()):
        return None
    tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date",
                                      "owner_name", "goods_services"]).filter(
        pl.col("filing_date").fill_null("").str.len_chars() >= 8).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8).alias("reg"),
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        pl.col("goods_services").fill_null("").str.to_lowercase()
          .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ")
          .str.strip_chars().alias("norm"),
        pl.col("owner_name").fill_null("").str.to_uppercase()
          .str.replace_all(r"[^A-Z0-9 ]", "").str.replace_all(r"\s+", " ")
          .str.strip_chars().alias("own"))
    tm = tm.filter(pl.col("fy").is_between(FY_LO, FY_HI)
                   & (pl.col("norm").str.len_chars() > 0)).with_columns(
        pl.col("norm").hash(seed=7).alias("h")).drop("norm")
    cl = tm.group_by("h").agg(pl.len().alias("dupN"),
                              pl.col("own").n_unique().alias("dup_owners"),
                              pl.col("filing_date").min().alias("first_fd"),
                              pl.col("own").sort_by("filing_date").first().alias("first_own"))
    tm = tm.join(cl, on="h", how="left")
    del cl
    tm = tm.with_columns(
        pl.when(pl.col("dupN") == 1).then(pl.lit("unique"))
          .when(pl.col("dupN") >= EDGE_MANUAL).then(pl.lit("manual"))
          .when((pl.col("filing_date") > pl.col("first_fd"))
                & (pl.col("own") != pl.col("first_own"))
                & (pl.col("dup_owners") >= 2)).then(pl.lit("copy_small"))
          .when((pl.col("filing_date") == pl.col("first_fd"))
                & (pl.col("own") == pl.col("first_own"))
                & (pl.col("dup_owners") >= 2)).then(pl.lit("original_copied"))
          .when((pl.col("filing_date") > pl.col("first_fd"))
                & (pl.col("own") == pl.col("first_own"))).then(pl.lit("own_reuse"))
          .otherwise(pl.lit("other_dup")).alias("cat"))
    sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                      "topic_kl_vs_future", "topic_dkl"]).filter(
        pl.col("topic_dkl").is_finite())
    tm = tm.join(sc, on="serial_number", how="left").with_columns(
        pl.col("topic_dkl").alias("L"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"),
        pl.lit(c).alias("cls"))
    del sc
    return tm.select("serial_number", "cls", "fy", "reg", "rd", "own", "cat",
                     "L", "A", "dupN", "dup_owners")


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    sec = set(pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet",
                              columns=["owner_name"])["owner_name"].unique())
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet",
                         columns=["owner_name", "in_8a", "ipo_date"])
    ipo = set(fm.filter((pl.col("in_8a").fill_null(0) == 1)
                        | pl.col("ipo_date").is_not_null())["owner_name"])

    parts = []
    for c in CLASSES:
        tm = load_class(c)
        if tm is None:
            continue
        parts.append(tm)
        log(f"[{c}] {tm.height:,}")
        gc.collect()
    d = pl.concat(parts); del parts; gc.collect()
    log(f"[frame] {d.height:,} class-records")

    d = d.with_columns(
        ((pl.col("L") - pl.col("L").mean().over(["cls", "fy"]))
         / pl.col("L").std().over(["cls", "fy"])).alias("zL"),
        ((pl.col("A") - pl.col("A").mean().over(["cls", "fy"]))
         / pl.col("A").std().over(["cls", "fy"])).alias("zA"),
        (pl.col("reg").cast(pl.Float64)
         - pl.col("reg").cast(pl.Float64).mean().over(["cls", "fy"])).alias("reg_dm"))

    out = {"categories": {}}
    for (cat,), g in sorted(d.group_by("cat"), key=lambda kv: kv[0][0]):
        out["categories"][cat] = {
            "n": g.height,
            "reg_rate": float(g["reg"].mean()),
            "reg_delta_within": float(g["reg_dm"].mean()),
            "mean_zL": float(g["zL"].mean()),
            "mean_zA": float(g["zA"].mean()),
        }

    # gate: registrations 2002-2018, unique serials (category of that class-record;
    # for multi-class serials keep the first row)
    r = d.filter(pl.col("reg") & pl.col("rd").is_not_null()).with_columns(
        pl.col("rd").dt.year().alias("ry")).filter(pl.col("ry").is_between(2002, 2018))
    r = r.unique("serial_number").join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        ((pl.col("age") >= 4.0) & (pl.col("age") < 8.5)).fill_null(False)
        .cast(pl.Float64).alias("fail")).with_columns(
        (pl.col("fail") - pl.col("fail").mean().over(["cls", "ry"])).alias("fail_dm"))
    for (cat,), g in sorted(r.group_by("cat"), key=lambda kv: kv[0][0]):
        out["categories"][cat]["n_gate"] = g.height
        out["categories"][cat]["gate_fail"] = float(g["fail"].mean())
        out["categories"][cat]["gate_delta_within"] = float(g["fail_dm"].mean())
    del r
    gc.collect()

    # thriving: registered debuts (owner's first filing date anywhere), category
    # of the debut class-record
    debut = d.group_by("own").agg(pl.col("fy").min().alias("debut_fy"))
    first = d.join(debut, on="own", how="inner").filter(pl.col("fy") == pl.col("debut_fy"))
    first = first.filter(pl.col("reg")).sort("serial_number").unique("own", keep="first")
    first = first.with_columns(pl.col("own").is_in(sorted(sec)).cast(pl.Float64).alias("in_sec"),
                               pl.col("own").is_in(sorted(ipo)).cast(pl.Float64).alias("in_ipo"))
    first = first.with_columns(
        (pl.col("in_sec") - pl.col("in_sec").mean().over(["cls", "fy"])).alias("sec_dm"),
        (pl.col("in_ipo") - pl.col("in_ipo").mean().over(["cls", "fy"])).alias("ipo_dm"))
    for (cat,), g in sorted(first.group_by("cat"), key=lambda kv: kv[0][0]):
        out["categories"][cat]["n_debut_reg"] = g.height
        out["categories"][cat]["sec_rate"] = float(g["in_sec"].mean())
        out["categories"][cat]["sec_delta_within"] = float(g["sec_dm"].mean())
        out["categories"][cat]["ipo_rate"] = float(g["in_ipo"].mean())
        out["categories"][cat]["ipo_delta_within"] = float(g["ipo_dm"].mean())

    for cat, v in out["categories"].items():
        log(f"[{cat}] " + json.dumps(v))
    (RES / "copy_outcomes.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
