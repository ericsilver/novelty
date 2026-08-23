"""How much of listed-company revenue sits with a few trademark-debut firms, and how did they read at debut?

Among trademark owners linked to SEC financial statements (uspto_sec_crosswalk
-> sec_firm_year), rank firms by their peak annual revenue in the panel. Report
the share of the panel's total peak revenue held by the top 10, 50, 100, 500 and
1,000 firms, and the vocabulary position (lead, atypicality) of each firm's
first registered filing against the rest of the linked firms and against all
debut filers in the same class and year.

The panel is Census-era FSDS (2004 on), so revenue is observed only for firms
reporting in that window; firms that listed and delisted before 2004 are
absent. Peak revenue is used rather than latest because latest would favour
firms still reporting at the edge.

Output: paper/results/value_concentration.json
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
TOPS = (10, 50, 100, 500, 1000)


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet", columns=["owner_name", "cik"]).unique("owner_name")
    fy = pl.read_parquet(PROC / "sec_firm_year.parquet", columns=["cik", "fy", "revenue"]).filter(
        pl.col("revenue").is_finite() & (pl.col("revenue") > 0))
    peak = fy.group_by("cik").agg(pl.col("revenue").max().alias("peak_rev"), pl.col("fy").min().alias("first_fy"))
    log(f"[sec] {peak.height:,} CIKs with revenue; {cw.height:,} crosswalked owners")

    # Debut filings: every owner's first registered filing, with its scores.
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
    d = d.with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
                       pl.col("topic_dkl").alias("L"),
                       ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    debut = d.sort(["owner_name", "filing_date", "serial_number"]).unique(subset="owner_name", keep="first")
    debut = debut.filter(pl.col("fy").is_between(1995, 2018))
    # Within class x year standardisation so positions compare across classes.
    debut = debut.with_columns(
        ((pl.col("L") - pl.col("L").mean().over(["cls", "fy"])) / pl.col("L").std().over(["cls", "fy"])).alias("zL"),
        ((pl.col("A") - pl.col("A").mean().over(["cls", "fy"])) / pl.col("A").std().over(["cls", "fy"])).alias("zA"))
    log(f"[debut] {debut.height:,} registered debut filers 1995-2018")

    linked = debut.join(cw, on="owner_name", how="inner").join(peak, on="cik", how="inner").sort("peak_rev", descending=True)
    linked = linked.unique(subset="cik", keep="first").sort("peak_rev", descending=True)
    total = float(linked["peak_rev"].sum())
    log(f"[linked] {linked.height:,} debut filers with SEC revenue; total peak revenue ${total/1e12:.2f}T")
    out = {"n_debut_filers": int(debut.height), "n_linked": int(linked.height), "total_peak_revenue": total,
           "mean_zL_all_debuts": float(debut["zL"].mean()), "mean_zA_all_debuts": float(debut["zA"].mean()),
           "mean_zL_linked": float(linked["zL"].mean()), "mean_zA_linked": float(linked["zA"].mean()),
           "tops": {}, "top_firms": []}
    for k in TOPS:
        top = linked.head(k)
        rest = linked.slice(k)
        out["tops"][str(k)] = {
            "share_of_linked_revenue": float(top["peak_rev"].sum() / total),
            "mean_zL": float(top["zL"].mean()), "mean_zA": float(top["zA"].mean()),
            "se_zL": float(top["zL"].std() / k ** 0.5), "se_zA": float(top["zA"].std() / k ** 0.5),
            "rest_mean_zL": float(rest["zL"].mean()), "rest_mean_zA": float(rest["zA"].mean()),
            "share_leading_fifth": float((top["zL"] > 0.84).mean()),
            "share_atypical_fifth": float((top["zA"] > 0.84).mean())}
        t = out["tops"][str(k)]
        log(f"  top {k:5d}: {100*t['share_of_linked_revenue']:.1f}% of revenue; zL {t['mean_zL']:+.2f} (rest {t['rest_mean_zL']:+.2f}); zA {t['mean_zA']:+.2f} (rest {t['rest_mean_zA']:+.2f})")
    for r in linked.head(40).iter_rows(named=True):
        out["top_firms"].append({"owner": r["owner_name"], "cik": int(r["cik"]), "peak_rev": r["peak_rev"],
                                 "debut_year": int(r["fy"]), "cls": r["cls"], "zL": r["zL"], "zA": r["zA"]})
    # Revenue-weighted position of the whole linked set, against equal-weighted.
    w = linked["peak_rev"].to_numpy() / total
    out["revenue_weighted_zL"] = float((linked["zL"].to_numpy() * w).sum())
    out["revenue_weighted_zA"] = float((linked["zA"].to_numpy() * w).sum())
    (RES / "value_concentration.json").write_text(json.dumps(out, indent=1, default=float))
    log("[done] value_concentration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
