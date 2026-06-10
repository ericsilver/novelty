"""RQ5 -- margin vs revenue demand-pull at the class level.

For each (NICE class c, year t) we compute:
  - class margin = median gross_margin of firms whose primary NICE class is c
  - class revenue growth = mean of firm log-revenue changes within class c
And the diffusion target:
  - theme inflow rate = change in mean share_top across the class's themes
    from t-1 to t (i.e., how much theme uptake the class is absorbing).

Then regress theme inflow on lagged margin and lagged revenue growth, with
class and year fixed effects. Schmookler (demand-pull) predicts revenue
coefficient > 0; appropriability/rents predicts margin coefficient > 0.

With only 4 NICE classes the panel is thin (4 classes x ~14 years = 56 rows);
we present this as associational with class/year FE and label it clearly.

Inputs:
  data/processed/theme_panel.parquet
  data/processed/uspto_sec_crosswalk.parquet
  data/processed/sec_firm_year.parquet
  data/processed/firm_year_class{009,042,039,035}.parquet

Output: paper/results/diff9_rq5.json
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
OUT_JSON = REPO / "paper" / "results" / "diff9_rq5.json"
CLASSES = ["009", "042", "039", "035"]


def main() -> int:
    t0 = time.time()
    # ---- 1) firm -> primary NICE class via firm_year_class ----------------
    firm_class_rows = []
    for c in CLASSES:
        d = pl.read_parquet(DATA / f"firm_year_class{c}.parquet",
                            columns=["owner_name", "year", "n_filings"])
        d = d.with_columns(pl.lit(c).alias("nice"))
        firm_class_rows.append(d)
    fc = pl.concat(firm_class_rows)
    primary = (fc.group_by(["owner_name", "nice"]).agg(pl.col("n_filings").sum().alias("n"))
                 .sort(["owner_name", "n"], descending=[False, True])
                 .group_by("owner_name", maintain_order=True).first()
                 .select("owner_name", pl.col("nice").alias("primary_nice")))
    print(f"[primary] {primary.height:,} owners with primary-class assignment",
          file=sys.stderr)

    # ---- 2) link to CIK via crosswalk -------------------------------------
    xw = pl.read_parquet(DATA / "uspto_sec_crosswalk.parquet")
    firm_cik = primary.join(xw.select("owner_name", "cik"), on="owner_name", how="inner").unique("cik")
    print(f"[link] {firm_cik.height:,} CIKs with a primary NICE class", file=sys.stderr)
    for r in firm_cik.group_by("primary_nice").len().sort("primary_nice").iter_rows(named=True):
        print(f"   {r['primary_nice']}: {r['len']:,}")

    # ---- 3) class-year financial aggregates -------------------------------
    sec = pl.read_parquet(DATA / "sec_firm_year.parquet").select(
        "cik", "fy", "revenue", "gross_margin", "rd_expense", "net_income"
    ).filter(pl.col("revenue").is_not_null() & (pl.col("revenue") > 0))
    sec_pd = sec.to_pandas()
    sec_pd["log_rev"] = np.log(sec_pd["revenue"])
    sec_pd["rev_growth"] = (sec_pd.sort_values(["cik", "fy"])
                                  .groupby("cik")["log_rev"].diff())
    # winsorize gross_margin and rev_growth at 1/99
    for col in ["gross_margin", "rev_growth"]:
        lo, hi = sec_pd[col].quantile([0.01, 0.99])
        sec_pd[col] = sec_pd[col].clip(lo, hi)

    fp = firm_cik.to_pandas()[["cik", "primary_nice"]]
    fp["cik"] = fp["cik"].astype(int)
    sec_pd["cik"] = sec_pd["cik"].astype(int)
    sec_class = sec_pd.merge(fp, on="cik")
    print(f"[merge] {len(sec_class):,} firm-year rows have a primary NICE class",
          file=sys.stderr)

    class_year = sec_class.groupby(["primary_nice", "fy"]).agg(
        class_margin=("gross_margin", "median"),
        class_rev_growth=("rev_growth", "mean"),
        class_rd=("rd_expense", "median"),
        n_firms=("cik", "nunique"),
    ).reset_index().rename(columns={"primary_nice": "nice", "fy": "year"})
    print(f"[class-year] {len(class_year):,} (class, year) rows",
          file=sys.stderr)

    # ---- 4) theme inflow per (class, year) --------------------------------
    panel = pl.read_parquet(DATA / "theme_panel.parquet").to_pandas()
    inflow = (panel.groupby(["nice", "year"])
                   .agg(mean_share=("share_top", "mean"),
                        n_active=("share_top", lambda x: (x >= 0.01).sum()))
                   .reset_index())
    inflow = inflow.sort_values(["nice", "year"])
    inflow["inflow"] = inflow.groupby("nice")["mean_share"].diff()
    inflow["n_active_growth"] = inflow.groupby("nice")["n_active"].diff()
    print(f"[inflow] {len(inflow):,} (class, year) rows",
          file=sys.stderr)

    # ---- 5) merge + regress -----------------------------------------------
    cy = class_year.copy()
    cy["margin_lag"] = cy.sort_values(["nice", "year"]).groupby("nice")["class_margin"].shift(1)
    cy["rev_lag"] = cy.sort_values(["nice", "year"]).groupby("nice")["class_rev_growth"].shift(1)
    panel_reg = inflow.merge(cy[["nice", "year", "margin_lag", "rev_lag"]], on=["nice", "year"])
    panel_reg = panel_reg.dropna(subset=["inflow", "margin_lag", "rev_lag"])
    print(f"[panel] {len(panel_reg):,} regression rows",
          file=sys.stderr)

    res = {"n_obs": int(len(panel_reg)),
           "by_class_n": {c: int((panel_reg["nice"] == c).sum()) for c in CLASSES}}

    if len(panel_reg) >= 16:
        # OLS with class FE + year FE
        X = pd.concat([
            pd.get_dummies(panel_reg["nice"], drop_first=True).astype(float),
            pd.get_dummies(panel_reg["year"].astype("category"), drop_first=True).astype(float),
            panel_reg[["margin_lag", "rev_lag"]].astype(float),
        ], axis=1)
        for outc in ("inflow", "n_active_growth"):
            y = panel_reg[outc].astype(float)
            try:
                m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC1")
            except Exception as e:
                print(f"     {outc}: failed ({e})", file=sys.stderr)
                continue
            res[outc] = {
                "margin_lag": {"coef": float(m.params["margin_lag"]),
                               "se": float(m.bse["margin_lag"]),
                               "t": float(m.tvalues["margin_lag"]),
                               "p": float(m.pvalues["margin_lag"])},
                "rev_lag": {"coef": float(m.params["rev_lag"]),
                            "se": float(m.bse["rev_lag"]),
                            "t": float(m.tvalues["rev_lag"]),
                            "p": float(m.pvalues["rev_lag"])},
                "r2": float(m.rsquared),
            }
            print(f"\n[{outc}] OLS w/ class+year FE  (n={len(panel_reg)}, R^2={m.rsquared:.3f}):")
            print(f"   margin_lag: coef={m.params['margin_lag']:+.4f}  "
                  f"t={m.tvalues['margin_lag']:+.2f}  p={m.pvalues['margin_lag']:.3f}")
            print(f"   rev_lag   : coef={m.params['rev_lag']:+.4f}  "
                  f"t={m.tvalues['rev_lag']:+.2f}  p={m.pvalues['rev_lag']:.3f}")

    OUT_JSON.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[done] -> {OUT_JSON}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
