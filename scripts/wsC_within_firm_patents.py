"""Workstream C: within-firm co-movement of trademark dKL and patenting.

Question (the sharp version of "is dKL an innovation measure?"): the
cross-sectional correlation between dKL and patents is ~0.01. Does dKL
nonetheless track invention *within a firm over time* -- i.e. in the years a
firm patents more, does its trademark language also sit further ahead of its
class? A positive within-firm slope says dKL carries firm-level innovation
information that the between-firm levels wash out (because firms differ in
branding propensity). A null says dKL and invention are temporally orthogonal
even inside the same firm.

Confound we CANNOT fully kill with this aggregated file (no NICE-class column):
product-line entry can spike patents AND class-churn exposure together. So a
positive result is "consistent with a firm-innovation component," not proof of
one. Reported honestly.

Input : data_publish/firm_year_patents_and_dkl.csv
Output: paper/results/wsC_within_firm_patents.json  (+ stdout summary)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data_publish" / "firm_year_patents_and_dkl.csv"
OUT = REPO / "paper" / "results" / "wsC_within_firm_patents.json"


def two_way_fe(df, yvar, xvars, firm_col, year_col):
    """Firm + year FE via firm within-transform + year dummies; cluster by firm.

    Returns dict of coef/se/t/n for each xvar.
    """
    d = df[[firm_col, year_col, yvar] + xvars].dropna().copy()
    # firm within-transform (absorb firm FE)
    def demean(s):
        return s - s.groupby(d[firm_col]).transform("mean")
    yd = demean(d[yvar])
    Xparts = {x: demean(d[x]) for x in xvars}
    # year dummies, also firm-demeaned so they absorb year FE within the FE model
    yr = pd.get_dummies(d[year_col].astype("category"), drop_first=True).astype(float)
    yr.index = d.index
    yr_d = yr.apply(lambda col: col - col.groupby(d[firm_col]).transform("mean"))
    X = pd.concat([pd.DataFrame(Xparts, index=d.index), yr_d], axis=1)
    X = X.astype(float)
    model = sm.OLS(yd.astype(float), X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d[firm_col].astype("category").cat.codes.values})
    out = {}
    for x in xvars:
        out[x] = dict(coef=float(res.params[x]), se=float(res.bse[x]),
                      t=float(res.tvalues[x]), p=float(res.pvalues[x]))
    out["n_obs"] = int(d.shape[0])
    out["n_firms"] = int(d[firm_col].nunique())
    return out


def main() -> int:
    df = pd.read_csv(SRC)
    for c in ["normalized_name", "uspto_owner_name", "patentsview_name"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    print("=== raw file ===")
    print(f"rows={len(df):,}  years {df.year.min()}-{df.year.max()}")
    for c in ["normalized_name", "uspto_owner_name", "patentsview_name"]:
        empty = (df[c] == "").mean()
        print(f"  {c:20s}: {df[c].nunique():>7,} distinct, {empty:6.1%} empty")
    print(f"mean_dkl: mean={df.mean_dkl.mean():.4f} sd={df.mean_dkl.std():.4f} "
          f"median={df.mean_dkl.median():.4f}")
    print(f"n_patents: mean={df.n_patents.mean():.2f} median={df.n_patents.median():.0f} "
          f"max={df.n_patents.max():,} p99={df.n_patents.quantile(.99):.0f}")
    print(f"n_filings: mean={df.n_filings.mean():.2f} median={df.n_filings.median():.0f}")

    # firm identifier: prefer normalized_name; fall back to uspto_owner_name
    df["firm_id"] = np.where(df.normalized_name != "", df.normalized_name, df.uspto_owner_name)
    df = df[df.firm_id != ""].copy()

    # collapse to one row per (firm_id, year): dkl = filing-weighted mean,
    # patents = mean across matched-assignee rows (robustness: also try max)
    def collapse(g):
        w = g.n_filings.clip(lower=1)
        return pd.Series({
            "mean_dkl": np.average(g.mean_dkl, weights=w),
            "n_filings": g.n_filings.sum(),
            "n_patents": g.n_patents.mean(),
            "n_patents_max": g.n_patents.max(),
        })
    panel = df.groupby(["firm_id", "year"], as_index=False).apply(collapse, include_groups=False)
    print(f"\n=== collapsed panel ===\nfirm-years={len(panel):,}  firms={panel.firm_id.nunique():,}")

    panel["logpat"] = np.log1p(panel.n_patents)
    panel["logpat_max"] = np.log1p(panel.n_patents_max)
    panel["log_nfilings"] = np.log1p(panel.n_filings)
    # standardize dkl to per-sigma units (overall)
    sd_dkl = panel.mean_dkl.std()
    panel["dkl_z"] = (panel.mean_dkl - panel.mean_dkl.mean()) / sd_dkl
    print(f"sigma(mean_dkl) over collapsed panel = {sd_dkl:.4f}")

    # require within-firm variation: >=3 years
    yr_counts = panel.groupby("firm_id").year.transform("count")
    p3 = panel[yr_counts >= 3].copy()
    print(f"firms with >=3 years: {p3.firm_id.nunique():,}  ({len(p3):,} firm-years)")

    results = {"sigma_dkl": float(sd_dkl)}

    # (1) between-firm: firm-mean dkl_z vs firm-mean logpat
    fm = p3.groupby("firm_id").agg(dkl_z=("dkl_z", "mean"), logpat=("logpat", "mean"))
    r_between = float(np.corrcoef(fm.dkl_z, fm.logpat)[0, 1])
    results["r_between_firm"] = r_between
    print(f"\n[between-firm] corr(mean dkl_z, mean logpat) = {r_between:+.4f}  (n_firms={len(fm):,})")

    # (2) within-firm correlation: partial out firm means AND year means
    def dd(col):
        x = p3[col] - p3[col].groupby(p3.firm_id).transform("mean")
        x = x - x.groupby(p3.year).transform("mean")
        return x
    wy, wx = dd("dkl_z"), dd("logpat")
    r_within = float(np.corrcoef(wy, wx)[0, 1])
    results["r_within_firm"] = r_within
    print(f"[within-firm ] corr(dkl_z, logpat | firm+year demeaned) = {r_within:+.4f}")

    # (3) contemporaneous two-way FE
    fe = two_way_fe(p3, "dkl_z", ["logpat"], "firm_id", "year")
    results["fe_contemp"] = fe
    print(f"\n[two-way FE] dkl_z ~ logpat (+firm+year FE, firm-clustered)")
    print(f"   coef={fe['logpat']['coef']:+.4f}  se={fe['logpat']['se']:.4f}  "
          f"t={fe['logpat']['t']:+.2f}  p={fe['logpat']['p']:.3g}  "
          f"n={fe['n_obs']:,} firms={fe['n_firms']:,}")

    # robustness: patents=max reduction
    fe_max = two_way_fe(p3, "dkl_z", ["logpat_max"], "firm_id", "year")
    results["fe_contemp_maxpat"] = fe_max
    print(f"   [robust max-patent] coef={fe_max['logpat_max']['coef']:+.4f} "
          f"se={fe_max['logpat_max']['se']:.4f} t={fe_max['logpat_max']['t']:+.2f}")

    # robustness: control for log(n_filings) -- kills the "more filings shrink
    # the firm-year mean toward the class average" mechanical channel
    fe_ctrl = two_way_fe(p3, "dkl_z", ["logpat", "log_nfilings"], "firm_id", "year")
    results["fe_contemp_ctrl_nfilings"] = fe_ctrl
    print(f"   [ctrl log_nfilings] logpat coef={fe_ctrl['logpat']['coef']:+.4f} "
          f"se={fe_ctrl['logpat']['se']:.4f} t={fe_ctrl['logpat']['t']:+.2f}; "
          f"nfilings coef={fe_ctrl['log_nfilings']['coef']:+.4f} t={fe_ctrl['log_nfilings']['t']:+.2f}")
    # within-firm co-movement of patents and filings (is the confound even present?)
    wpat = p3.logpat - p3.logpat.groupby(p3.firm_id).transform("mean")
    wfil = p3.log_nfilings - p3.log_nfilings.groupby(p3.firm_id).transform("mean")
    results["r_within_pat_filings"] = float(np.corrcoef(wpat, wfil)[0, 1])
    print(f"   within-firm corr(logpat, log_nfilings) = {results['r_within_pat_filings']:+.4f}")

    # (4) leads/lags: dkl_t on patents_{t-1,t,t+1}
    p3 = p3.sort_values(["firm_id", "year"])
    g = p3.groupby("firm_id")
    p3["logpat_lag"] = g.logpat.shift(1)      # patents last year
    p3["logpat_lead"] = g.logpat.shift(-1)    # patents next year
    # only keep consecutive-year shifts
    p3["yr_prev"] = g.year.shift(1)
    p3["yr_next"] = g.year.shift(-1)
    ll = p3.copy()
    ll.loc[ll.yr_prev != ll.year - 1, "logpat_lag"] = np.nan
    ll.loc[ll.yr_next != ll.year + 1, "logpat_lead"] = np.nan
    fe_ll = two_way_fe(ll, "dkl_z", ["logpat_lag", "logpat", "logpat_lead"], "firm_id", "year")
    results["fe_leadlag"] = fe_ll
    print(f"\n[two-way FE leads/lags] dkl_z ~ patents(t-1,t,t+1)")
    for k in ["logpat_lag", "logpat", "logpat_lead"]:
        print(f"   {k:12s} coef={fe_ll[k]['coef']:+.4f} se={fe_ll[k]['se']:.4f} t={fe_ll[k]['t']:+.2f}")
    print(f"   n={fe_ll['n_obs']:,} firms={fe_ll['n_firms']:,}")

    # robustness: clean subset (normalized_name nonempty already used as firm_id when present)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
