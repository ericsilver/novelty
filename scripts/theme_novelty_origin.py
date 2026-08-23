"""Wholly new themes against themes new to a class: does the gate treat them differently?

A theme can be new in two senses. It can be new to the corpus: no class was
filing it until recently. Or it can be established elsewhere and only now
arriving in this class. The 500-theme model is fine enough to separate the
two. For each theme and each class, the first sustained year is the first
calendar year by which the theme has been the dominant theme of at least
MIN_SUSTAINED filings in that class; the corpus-wide first sustained year is
the earliest of those across classes.

Each registration in the gate sample is assigned its dominant theme under the
T=500 model and classified at its filing date as:

  new to corpus     filed within NEW_YEARS of the theme's corpus-wide first sustained year
  new to class      not new to corpus, but within NEW_YEARS of the theme's first
                    sustained year in this class
  established       neither

Gate failure is then compared across the three groups within class and
registration year, with and without the filing's own lead and atypicality held.

Cost: transforming the 3.1M gate registrations at T=500 is the expensive step
(about an hour); theta is cached per class so the run can resume.

Output: paper/results/theme_novelty_origin.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CACHE = RES / "theme_novelty_cache"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
T = 500
MIN_SUSTAINED = 100
NEW_YEARS = 3
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    m = joblib.load(PROC / f"topic_model_T{T}.joblib")
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True, token_pattern=TOKEN, ngram_range=(1, 2))
    words = json.loads((PROC / f"topic_lda_meta_T{T}.json").read_text())["top_words"]
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    CACHE.mkdir(parents=True, exist_ok=True)

    # 1. Dominant theme for every filing 1985-2024 (for first-sustained years) is
    #    expensive; use the gate sample plus all filings for the sustained-year
    #    table at a 1-in-4 thinning outside the gate sample.
    parts = []
    for c in CLASSES:
        p = CACHE / f"{c}.parquet"
        if p.exists():
            parts.append(pl.read_parquet(p)); log(f"  [{c}] cached"); continue
        tp = PROC / f"tm_class{c}.parquet"
        if not tp.exists():
            continue
        d = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date", "goods_services"]).filter(
            pl.col("goods_services").is_not_null() & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        ).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"))
        d = d.with_columns(pl.col("rd").dt.year().alias("ry"),
                           (pl.col("serial_number").cast(pl.Int64, strict=False) % 4 == 0).alias("thin"))
        keep = d.filter(pl.col("ry").is_between(REG_LO, REG_HI) | pl.col("thin"))
        texts = keep["goods_services"].to_list()
        dom = np.empty(len(texts), dtype=np.int16)
        for s in range(0, len(texts), 100_000):
            dom[s:s + 100_000] = m["lda"].transform(vec.transform(texts[s:s + 100_000])).argmax(axis=1)
        keep = keep.drop("goods_services", "filing_date", "registration_date").with_columns(
            pl.Series("theme", dom), pl.lit(c).alias("cls"))
        keep.write_parquet(p); parts.append(keep)
        log(f"  [{c}] {keep.height:,} filings themed")
        del d, keep, texts, dom; gc.collect()
    all_ = pl.concat(parts); del parts; gc.collect()

    # 2. First sustained year per (theme, class) and per theme corpus-wide, using
    #    the thinned universe (every 4th serial) so classes are comparable.
    thin = all_.filter(pl.col("thin")).sort("fy")
    cum = thin.group_by(["theme", "cls", "fy"]).len().sort(["theme", "cls", "fy"]).with_columns(
        pl.col("len").cum_sum().over(["theme", "cls"]).alias("cum"))
    first_cls = cum.filter(pl.col("cum") * 4 >= MIN_SUSTAINED).group_by(["theme", "cls"]).agg(
        pl.col("fy").min().alias("first_cls"))
    cum_all = thin.group_by(["theme", "fy"]).len().sort(["theme", "fy"]).with_columns(
        pl.col("len").cum_sum().over("theme").alias("cum"))
    first_all = cum_all.filter(pl.col("cum") * 4 >= MIN_SUSTAINED).group_by("theme").agg(
        pl.col("fy").min().alias("first_all"))

    # 3. Classify gate registrations.
    g = all_.filter(pl.col("ry").is_between(REG_LO, REG_HI)).join(first_cls, on=["theme", "cls"], how="left"
        ).join(first_all, on="theme", how="left").join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64).alias("failed"))
    g = g.with_columns(
        pl.when(pl.col("first_all").is_null() | (pl.col("fy") < pl.col("first_all") + NEW_YEARS)).then(pl.lit("new to corpus"))
        .when(pl.col("first_cls").is_null() | (pl.col("fy") < pl.col("first_cls") + NEW_YEARS)).then(pl.lit("new to class"))
        .otherwise(pl.lit("established")).alias("origin"))
    # Scores for holding lead/atypicality.
    sc = pl.concat([pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                                    columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"])
                    for c in CLASSES if (PROC / f"rolling_surprise_class{c}.parquet").exists()]).unique("serial_number")
    g = g.join(sc, on="serial_number", how="left").with_columns(
        pl.col("topic_dkl").alias("L"), ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))

    out = {"T": T, "min_sustained": MIN_SUSTAINED, "new_years": NEW_YEARS, "n_gate": int(g.height), "groups": {}}
    base = g.group_by("origin").agg(pl.len().alias("n"), pl.col("failed").mean().alias("fail"),
                                    pl.col("L").mean().alias("mean_L"), pl.col("A").mean().alias("mean_A")).sort("origin")
    for r in base.iter_rows(named=True):
        out["groups"][r["origin"]] = {"n": r["n"], "fail_raw": r["fail"], "mean_L": r["mean_L"], "mean_A": r["mean_A"]}
    # Within class x year: demean failure by cell, then group means; also residualised on L and A.
    g = g.with_columns((pl.col("failed") - pl.col("failed").mean().over(["cls", "ry"])).alias("f_dm"))
    gg = g.filter(pl.col("L").is_finite() & pl.col("A").is_finite())
    X = np.column_stack([np.ones(gg.height), gg["L"].to_numpy(), gg["A"].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, gg["f_dm"].to_numpy(), rcond=None)
    gg = gg.with_columns(pl.Series("f_res", gg["f_dm"].to_numpy() - X @ beta))
    for name, df, col in (("within_class_year", g, "f_dm"), ("within_class_year_net_LA", gg, "f_res")):
        t = df.group_by("origin").agg(pl.col(col).mean().alias("m"), pl.col(col).std().alias("s"), pl.len().alias("n"))
        for r in t.iter_rows(named=True):
            out["groups"][r["origin"]][name] = {"mean": r["m"], "se": r["s"] / r["n"] ** 0.5}
    for k, v in out["groups"].items():
        log(f"  {k:14s} n={v['n']:9,} fail {100*v['fail_raw']:.1f}%  within {100*v['within_class_year']['mean']:+.2f} "
            f"(SE {100*v['within_class_year']['se']:.2f})  net L,A {100*v['within_class_year_net_LA']['mean']:+.2f}")
    # Top new-to-corpus themes by count in the gate sample, with words.
    top = g.filter(pl.col("origin") == "new to corpus").group_by("theme").agg(pl.len().alias("n"), pl.col("failed").mean().alias("fail"),
                                                                             pl.col("first_all").min().alias("first_all")).sort("n", descending=True).head(25)
    out["top_new_themes"] = [{"theme": int(r["theme"]), "n": int(r["n"]), "fail": r["fail"], "first_year": r["first_all"],
                              "words": words[str(int(r["theme"]))][:6]} for r in top.iter_rows(named=True)]
    (RES / "theme_novelty_origin.json").write_text(json.dumps(out, indent=1))
    log("[done] theme_novelty_origin.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
