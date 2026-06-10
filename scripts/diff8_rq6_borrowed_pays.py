"""RQ6 -- does borrowed novelty pay? (firm-level outcome bridge)

For firms in our 4-class corpus that match a SEC CIK, aggregate per-filing
borrowed_novelty to firm-year (mean borrowed_gap weighted by n_filings).
Then test whether high borrowed-novelty firms outperform on subsequent
financial outcomes (gross margin, revenue growth, R&D intensity).

This is the localized innovation test the construct-validity critique
demanded: if borrowed novelty (high-within, low-cross resonance) tracks a
real firm-level signal, firms scoring high should show measurable financial
follow-through.

Inputs:
  data/processed/borrowed_novelty.parquet         (per-filing within/cross)
  data/processed/uspto_sec_crosswalk.parquet      (owner_name -> CIK)
  data/processed/sec_firm_year.parquet            (CIK x fy financials)
  data/processed/tm_class{009,042,039,035}.parquet (owner_name per serial)

Output: paper/results/diff8_rq6.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT_JSON = REPO / "paper" / "results" / "diff8_rq6.json"
CLASSES = ["009", "042", "039", "035"]


def two_way_fe(d, y, x, firm_col, year_col, clusters=True):
    """firm + year FE via firm within-transform + year dummies; firm-clustered SE."""
    d = d[[firm_col, year_col, y] + x].dropna().copy()
    if len(d) < 50:
        return None
    yd = d[y] - d[y].groupby(d[firm_col]).transform("mean")
    Xparts = {c: d[c] - d[c].groupby(d[firm_col]).transform("mean") for c in x}
    yr = pd.get_dummies(d[year_col].astype("category"), drop_first=True).astype(float)
    yr.index = d.index
    yr_d = yr.apply(lambda col: col - col.groupby(d[firm_col]).transform("mean"))
    X = pd.concat([pd.DataFrame(Xparts, index=d.index), yr_d], axis=1).astype(float)
    res = sm.OLS(yd.astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d[firm_col].astype("category").cat.codes.values}
        if clusters else "HC1"
    )
    out = {"n": int(len(d)), "n_firms": int(d[firm_col].nunique())}
    for c in x:
        out[c] = {"coef": float(res.params[c]), "se": float(res.bse[c]),
                  "t": float(res.tvalues[c]), "p": float(res.pvalues[c])}
    return out


def main() -> int:
    t0 = time.time()
    bn = pl.read_parquet(DATA / "borrowed_novelty.parquet")
    xw = pl.read_parquet(DATA / "uspto_sec_crosswalk.parquet")
    sec = pl.read_parquet(DATA / "sec_firm_year.parquet")
    print(f"[load] borrowed: {bn.height:,}, crosswalk: {xw.height:,}, "
          f"sec firm-years: {sec.height:,}", file=sys.stderr)

    # serial -> owner_name across the 4 classes
    own_rows = []
    for c in CLASSES:
        d = pl.read_parquet(DATA / f"tm_class{c}.parquet",
                            columns=["serial_number", "owner_name"])
        own_rows.append(d)
    serials = pl.concat(own_rows)

    # join: filing -> owner -> CIK
    f = (bn.join(serials, on="serial_number", how="inner")
           .join(xw.select("owner_name", "cik"), on="owner_name", how="inner"))
    print(f"[link] {f.height:,} filings link owner->CIK", file=sys.stderr)
    print(f"       distinct CIKs: {f['cik'].n_unique():,}", file=sys.stderr)

    # aggregate to (CIK, year): filing-weighted means
    firm_year = (f.group_by(["cik", "year"]).agg(
        pl.col("within_nov").mean().alias("within_mean"),
        pl.col("cross_nov").mean().alias("cross_mean"),
        pl.col("borrowed_gap").mean().alias("gap_mean"),
        pl.len().alias("n_filings"),
    ).rename({"year": "filing_year"}))
    print(f"[firmyr] {firm_year.height:,} (CIK, filing_year) rows  "
          f"distinct CIKs={firm_year['cik'].n_unique():,}", file=sys.stderr)

    # join with SEC financials at filing_year and filing_year+1, +2
    sec_pd = (sec.select("cik", "fy", "revenue", "gross_margin", "net_income",
                         "rd_expense", "operating_income")
                 .to_pandas())
    sec_pd["log_rev"] = np.log(sec_pd["revenue"].clip(lower=1))
    sec_pd["rev_growth"] = (sec_pd.sort_values(["cik", "fy"])
                                  .groupby("cik")["log_rev"].diff())
    sec_pd["rd_intensity"] = sec_pd["rd_expense"] / sec_pd["revenue"]
    sec_pd = sec_pd.replace([np.inf, -np.inf], np.nan)

    fy_pd = firm_year.to_pandas()
    fy_pd["cik"] = fy_pd["cik"].astype(int)

    # at filing year t and t+1, t+2
    results = {"cik_linked_filings": int(f.height), "distinct_ciks": int(firm_year["cik"].n_unique()),
               "regressions": {}}
    print(f"\n[fit ] outcome regressions:")
    sd_gap = fy_pd["gap_mean"].std()
    fy_pd["gap_z"] = (fy_pd["gap_mean"] - fy_pd["gap_mean"].mean()) / sd_gap

    for lag in (0, 1, 2):
        sub = fy_pd.merge(sec_pd[["cik", "fy", "gross_margin", "rev_growth", "rd_intensity",
                                  "operating_income", "revenue"]],
                          left_on=["cik"], right_on=["cik"])
        sub = sub[sub.fy == sub.filing_year + lag].copy()
        sub["log_rev"] = np.log(sub["revenue"].clip(lower=1))
        if len(sub) < 100:
            continue
        # winsorize outcomes 1/99
        for outc in ["gross_margin", "rev_growth", "rd_intensity"]:
            lo, hi = sub[outc].quantile([0.01, 0.99])
            sub[outc] = sub[outc].clip(lo, hi)
        print(f"   lag = +{lag}y, n_pairs = {len(sub):,}, CIKs = {sub.cik.nunique():,}")
        for outc in ["gross_margin", "rev_growth", "rd_intensity"]:
            r = two_way_fe(sub, outc, ["gap_z"], "cik", "fy")
            if r is None:
                continue
            results["regressions"][f"{outc}_t+{lag}"] = r
            print(f"     {outc:>13}: coef={r['gap_z']['coef']:+.4f}  "
                  f"se={r['gap_z']['se']:.4f}  t={r['gap_z']['t']:+.2f}  "
                  f"n={r['n']:,} firms={r['n_firms']:,}")

    # Also report cross-sectional firm-mean test: collapse to one row per firm
    firm_mean = fy_pd.groupby("cik").agg(
        gap_mean=("gap_mean", "mean"), n_filings=("n_filings", "sum")).reset_index()
    sec_mean = sec_pd.groupby("cik").agg(gross_margin=("gross_margin", "mean"),
                                          rev_growth=("rev_growth", "mean"),
                                          rd_intensity=("rd_intensity", "mean")).reset_index()
    cross = firm_mean.merge(sec_mean, on="cik")
    cross["gap_z"] = (cross.gap_mean - cross.gap_mean.mean()) / cross.gap_mean.std()
    print(f"\n[cross] firm-level mean, n={len(cross):,} firms")
    cross_rows = {}
    for outc in ["gross_margin", "rev_growth", "rd_intensity"]:
        c = cross[["gap_z", outc]].dropna()
        if len(c) < 100:
            continue
        x = sm.add_constant(c["gap_z"])
        res = sm.OLS(c[outc].astype(float), x).fit(cov_type="HC1")
        cross_rows[outc] = {"coef": float(res.params["gap_z"]),
                             "se": float(res.bse["gap_z"]),
                             "t": float(res.tvalues["gap_z"]),
                             "p": float(res.pvalues["gap_z"]),
                             "n": int(len(c))}
        print(f"   {outc:>13}: coef={cross_rows[outc]['coef']:+.4f}  "
              f"se={cross_rows[outc]['se']:.4f}  t={cross_rows[outc]['t']:+.2f}  "
              f"n={cross_rows[outc]['n']:,}")
    results["cross_section"] = cross_rows

    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] -> {OUT_JSON}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
