"""Owners who reached an IPO marker without a priced Reg D round: are they the boring ones?

Hypothesis (Eric): unfunded companies that IPO tend to be negative-lead
("boring") companies doing something useful but hard to fund. Test on the
2009-2018 debut ladder frame: among owners with an IPO marker, split by
whether a post-debut Form D round is observed; compare mean lead and
atypicality (within-class-and-year standardized) and the lead-quintile
distribution. Small-n caveat expected.

Output: paper/results/unfunded_ipo.json
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
DEBUT_LO, DEBUT_HI = 2009, 2018


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet").select(
        "owner_name", "first_formd_date", "in_8a", "ipo_date")
    parts = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date", "owner_name"]).filter(
            pl.col("owner_name").is_not_null() & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(pl.lit(c).alias("cls")))
        del tm, sc
    d = pl.concat(parts); del parts; gc.collect()
    d = d.with_columns(pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
                       pl.col("topic_dkl").alias("L"),
                       ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    debut = d.sort(["owner_name", "fd", "serial_number"]).unique(subset="owner_name", keep="first").with_columns(
        pl.col("fd").dt.year().alias("fy")).filter(pl.col("fy").is_between(DEBUT_LO, DEBUT_HI))
    debut = debut.with_columns(
        ((pl.col("L") - pl.col("L").mean().over(["cls", "fy"])) / pl.col("L").std().over(["cls", "fy"])).alias("zL"),
        ((pl.col("A") - pl.col("A").mean().over(["cls", "fy"])) / pl.col("A").std().over(["cls", "fy"])).alias("zA"))
    x = debut.join(fm, on="owner_name", how="left").with_columns(
        (pl.col("first_formd_date").is_not_null() & (pl.col("first_formd_date") >= pl.col("fd"))).alias("funded"),
        ((pl.col("in_8a").fill_null(0) == 1) | pl.col("ipo_date").is_not_null()).alias("ipo"))
    ip = x.filter(pl.col("ipo"))
    out = {"n_ipo": int(ip.height)}
    for lab, cond in (("ipo_unfunded", ~pl.col("funded")), ("ipo_funded", pl.col("funded"))):
        sub = ip.filter(cond)
        out[lab] = {"n": int(sub.height),
                    "mean_zL": float(sub["zL"].mean()), "se_zL": float(sub["zL"].std() / max(sub.height, 1) ** 0.5),
                    "mean_zA": float(sub["zA"].mean()), "se_zA": float(sub["zA"].std() / max(sub.height, 1) ** 0.5),
                    "share_zL_negative": float((sub["zL"] < 0).mean()),
                    "share_bottom_lead_fifth": float((sub["zL"] < -0.84).mean()),
                    "share_top_lead_fifth": float((sub["zL"] > 0.84).mean())}
        v = out[lab]
        log(f"  {lab}: n={v['n']:,} zL {v['mean_zL']:+.3f} (SE {v['se_zL']:.3f}) zA {v['mean_zA']:+.3f} "
            f"neg-lead share {v['share_zL_negative']:.2f} bottom-fifth {v['share_bottom_lead_fifth']:.2f}")
    base = x.filter(~pl.col("ipo"))
    out["non_ipo_mean_zL"] = float(base["zL"].mean())
    out["non_ipo_mean_zA"] = float(base["zA"].mean())
    (RES / "unfunded_ipo.json").write_text(json.dumps(out, indent=1))
    log("[done] unfunded_ipo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
