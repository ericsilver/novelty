"""Ladder strata: the SEC event ladder split by representation and by debut half.

Rebuilds the sec_event_ladder debut frame (2009-2018 registered debut owners,
production scores, length in words), joins counsel of record on the debut
serial, and reports the length-held lead and atypicality quintile contrasts
for reporting and IPO within four strata: counsel debuts, self-filed debuts,
debut years 2009-2013, debut years 2014-2018. Baselines: sec_event_ladder.json
(pooled: reporting L -0.206pp t -9.4, A +0.113pp t +5.2; ipo L -0.065 t -4.5,
A +0.065 t +4.8).

Output: paper/v3/_eval/ladder_strata.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
EV = REPO / "paper" / "v3" / "_eval"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
DEBUT_LO, DEBUT_HI = 2009, 2018


def log(m): print(m, file=sys.stderr, flush=True)


def contrast(df, var, y):
    d = df.filter(pl.col(var).is_finite())
    if d.height < 2000:
        return None
    d = d.with_columns(((pl.col("n_words").rank("ordinal").over(["cls", "fy"]) - 1) * 5
                        // pl.len().over(["cls", "fy"])).cast(pl.Int8).alias("lenq"))
    cells = ["cls", "fy", "lenq"]
    s = d.sort(cells + [var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over(cells) - 1) * 5 // pl.len().over(cells)).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col(y).mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    if len(p) < 5 or min(n) < 50:
        return None
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"n": int(d.height), "base": float(d[y].mean()), "lift": p[4] - p[0], "se": se,
            "t": (p[4] - p[0]) / se if se else None}


def main() -> int:
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet").select(
        "owner_name", "first_formd_date", "in_sec", "in_fsds", "in_8a", "ipo_date")
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet", columns=["owner_name", "cik"]).unique("owner_name")
    att = pl.read_parquet(PROC / "case_extras.parquet", columns=["serial_number", "attorney_name"]).with_columns(
        (pl.col("attorney_name").fill_null("").str.len_chars() > 0).alias("counsel")).select("serial_number", "counsel")
    parts = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date", "owner_name",
                                          "goods_services"]).filter(
            pl.col("owner_name").is_not_null() & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        d = tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.lit(c).alias("cls"),
            pl.col("goods_services").str.to_lowercase().str.count_matches(r"[a-z]+").alias("n_words")
        ).drop("goods_services")
        parts.append(d)
        del tm, sc, d
    d = pl.concat(parts); del parts; gc.collect()
    d = d.with_columns(pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
                       pl.col("topic_dkl").alias("L"),
                       ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    debut = d.sort(["owner_name", "fd", "serial_number"]).unique(subset="owner_name", keep="first").with_columns(
        pl.col("fd").dt.year().alias("fy")).filter(pl.col("fy").is_between(DEBUT_LO, DEBUT_HI))
    x = debut.join(att, on="serial_number", how="left").with_columns(pl.col("counsel").fill_null(False))
    x = x.join(fm, on="owner_name", how="left").join(cw, on="owner_name", how="left").with_columns(
        ((pl.col("in_sec").fill_null(0) == 1) | (pl.col("in_fsds").fill_null(0) == 1) | pl.col("cik").is_not_null()).cast(pl.Float64).alias("reporting"),
        ((pl.col("in_8a").fill_null(0) == 1) | pl.col("ipo_date").is_not_null()).cast(pl.Float64).alias("ipo"))
    log(f"[debut] {x.height:,} owners; counsel share {x['counsel'].mean():.2f}")
    strata = {"counsel": pl.col("counsel"), "self": ~pl.col("counsel"),
              "debut_2009_2013": pl.col("fy") <= 2013, "debut_2014_2018": pl.col("fy") >= 2014}
    out = {}
    for name, cond in strata.items():
        sub = x.filter(cond)
        out[name] = {"n": int(sub.height)}
        for rung in ("reporting", "ipo"):
            for var in ("L", "A"):
                c = contrast(sub, var, rung)
                out[name][f"{rung}_{var}"] = c
                if c:
                    log(f"  {name:16s} {rung:9s} {var}: lift {100*c['lift']:+.3f}pp (t {c['t']:+.1f}) base {100*c['base']:.2f}%")
    EV.mkdir(parents=True, exist_ok=True)
    (EV / "ladder_strata.json").write_text(json.dumps(out, indent=1))
    log("[done] ladder_strata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
