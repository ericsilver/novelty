"""Lead-lag T4+T5 only: within-firm FE regressions with application-
date patent lags. Reads the saved panel and strips empty-norm rows
(which caused a many-to-many merge explosion in v2)."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "results"


def main() -> int:
    p = pl.read_parquet(OUT / "leadlag_firm_year.parquet")
    # Strip empty-norm rows (3195/year duplicates) + null norm
    p = p.filter(pl.col("norm").is_not_null() & (pl.col("norm") != ""))
    # Patent-active firms only
    active = p.group_by("norm").agg(pl.col("n_patents_application").sum().alias("tot_pat"))
    keep = active.filter(pl.col("tot_pat") > 0).select("norm")
    p = p.join(keep, on="norm", how="inner")
    # Deduplicate (norm, year) just in case
    p = p.unique(["norm","year"])
    pdf = p.to_pandas().sort_values(["norm","year"])
    print(f"panel: {len(pdf):,} rows, {pdf['norm'].nunique():,} firms")

    # ----- T4: mean_dkl ~ pat_app lags 0..3, within-firm FE
    print("\n=== T4: Within-firm FE, mean_dkl ~ pat_lag0..3 ===")
    work = pdf[["norm","year","mean_dkl","n_patents_application"]].copy()
    base = work[["norm","year","n_patents_application"]].copy()
    for lag in [0, 1, 2, 3]:
        s = base.copy()
        s["year"] = s["year"] + lag
        s = s.rename(columns={"n_patents_application": f"pat_lag{lag}"})
        work = work.merge(s[["norm","year",f"pat_lag{lag}"]], on=["norm","year"], how="left")
    work = work.dropna(subset=["mean_dkl"])
    cnt = work.groupby("norm")["mean_dkl"].count()
    keep_norms = cnt[cnt >= 5].index
    work = work[work["norm"].isin(keep_norms)].copy()
    for lag in [0,1,2,3]: work[f"pat_lag{lag}"] = work[f"pat_lag{lag}"].fillna(0)
    fmean = work.groupby("norm")[["mean_dkl","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]].transform("mean")
    w = work[["mean_dkl","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]] - fmean
    w = w.dropna()
    X = sm.add_constant(w[["pat_lag0","pat_lag1","pat_lag2","pat_lag3"]])
    y = w["mean_dkl"]
    mod = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": work.loc[w.index, "norm"]})
    print(mod.summary().tables[1])
    print(f"N firms = {work['norm'].nunique():,}, N firm-years = {len(w):,}")

    # ----- T5: n_positive_dkl_filings ~ pat_app lags 0..3
    print("\n=== T5: Within-firm FE, n_positive_dkl_filings ~ pat_lag0..3 ===")
    work2 = pdf[["norm","year","n_positive_dkl_filings","n_patents_application","n_filings_clean"]].copy()
    work2 = work2[work2["n_filings_clean"] >= 1]
    base2 = work2[["norm","year","n_patents_application"]].copy()
    for lag in [0, 1, 2, 3]:
        s = base2.copy()
        s["year"] = s["year"] + lag
        s = s.rename(columns={"n_patents_application": f"pat_lag{lag}"})
        work2 = work2.merge(s[["norm","year",f"pat_lag{lag}"]], on=["norm","year"], how="left")
    cnt = work2.groupby("norm")["n_positive_dkl_filings"].count()
    keep_norms2 = cnt[cnt >= 5].index
    work2 = work2[work2["norm"].isin(keep_norms2)].copy()
    for lag in [0,1,2,3]: work2[f"pat_lag{lag}"] = work2[f"pat_lag{lag}"].fillna(0)
    fmean2 = work2.groupby("norm")[["n_positive_dkl_filings","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]].transform("mean")
    w2 = work2[["n_positive_dkl_filings","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]] - fmean2
    w2 = w2.dropna()
    X2 = sm.add_constant(w2[["pat_lag0","pat_lag1","pat_lag2","pat_lag3"]])
    y2 = w2["n_positive_dkl_filings"]
    mod2 = sm.OLS(y2, X2).fit(cov_type="cluster", cov_kwds={"groups": work2.loc[w2.index, "norm"]})
    print(mod2.summary().tables[1])
    print(f"N firms = {work2['norm'].nunique():,}, N firm-years = {len(w2):,}")

    # Append to leadlag_v2.txt
    extra = []
    extra.append("\n" + "="*70)
    extra.append("T4: Within-firm FE, mean_dkl ~ pat_app lags 0..3 (clustered SE)")
    extra.append(str(mod.summary().tables[1]))
    extra.append(f"N firms = {work['norm'].nunique():,}, N firm-years = {len(w):,}")
    extra.append("")
    extra.append("T5: Within-firm FE, n_positive_dkl_filings ~ pat_app lags 0..3")
    extra.append(str(mod2.summary().tables[1]))
    extra.append(f"N firms = {work2['norm'].nunique():,}, N firm-years = {len(w2):,}")
    with open(OUT/"leadlag_v2.txt", "a") as f: f.write("\n".join(extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
