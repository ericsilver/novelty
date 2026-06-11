"""Gross-margin regression under topic scoring, mirroring
financials_regression.py exactly (classes 009+032, CIK-year aggregation,
trailing 3-year means, controls log_revenue + rd_intensity + year poly,
CIK-clustered SEs).

Because topic scores cover 1995-2019, the token benchmark is re-run on the
same year range so coefficients are comparable.

Output: paper/results/topic_firm_margin.json
"""
from __future__ import annotations

import json
import os
import sys
SUFFIX = os.environ.get("TOPIC_SUFFIX", "")
from pathlib import Path

import polars as pl
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

CLASSES = ("009", "032")
YEAR_MAX = 2019


def firm_year_topic() -> pl.DataFrame:
    parts = []
    for cls in CLASSES:
        ts = pl.read_parquet(PROC / f"topic_surprise_class{cls}{SUFFIX}.parquet").filter(
            pl.col("topic_dkl").is_finite())
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                             columns=["serial_number", "owner_name"])
        parts.append(ts.join(tm, on="serial_number", how="inner"))
    j = pl.concat(parts)
    return j.group_by(["owner_name", "year"]).agg(
        pl.len().alias("n_filings"),
        pl.col("topic_dkl").mean().alias("mean_dkl"),
        pl.col("topic_pros").mean().alias("mean_pros_kl"),
        pl.col("topic_retr").mean().alias("mean_retr_kl"),
    )


def firm_year_token() -> pl.DataFrame:
    parts = []
    for cls in CLASSES:
        fy = pl.read_parquet(PROC / f"firm_year_class{cls}.parquet")
        parts.append(fy.select("owner_name", "year", "n_filings", "mean_dkl",
                               "mean_pros_kl", "mean_retr_kl"))
    return pl.concat(parts)


def build_panel(tm_fy: pl.DataFrame) -> "object":
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik", "sec_name")
    tm_fy = tm_fy.join(cw, on="owner_name", how="inner")
    tm_cy = tm_fy.group_by(["cik", "year"]).agg(
        pl.col("n_filings").sum().alias("n_filings_yr"),
        pl.col("mean_dkl").mean().alias("mean_dkl_yr"),
        pl.col("mean_pros_kl").mean().alias("mean_pros_yr"),
        pl.col("mean_retr_kl").mean().alias("mean_retr_yr"),
    ).sort(["cik", "year"])
    pd_df = tm_cy.to_pandas().sort_values(["cik", "year"])
    roll = pd_df.groupby("cik").rolling(3, on="year", min_periods=1)[[
        "mean_dkl_yr", "mean_pros_yr", "mean_retr_yr"]].mean().reset_index()
    roll.columns = ["cik", "year", "mean_dkl_3y", "mean_pros_3y", "mean_retr_3y"]
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet").select(
        "cik", "fy", "revenue", "gross_margin", "rd_expense"
    ).rename({"fy": "year"})
    panel = pl.from_pandas(roll).join(sec, on=["cik", "year"], how="inner")
    panel = panel.filter(
        pl.col("gross_margin").is_not_null()
        & (pl.col("gross_margin") > -1) & (pl.col("gross_margin") < 1.5)
        & (pl.col("revenue") > 0)
        & pl.col("mean_dkl_3y").is_not_null()
        & (pl.col("year") <= YEAR_MAX)
    ).with_columns(
        pl.col("revenue").log().alias("log_revenue"),
        (pl.col("rd_expense") / pl.col("revenue")).fill_null(0).alias("rd_intensity"),
    )
    return panel.to_pandas()


def fit(pdf, predictors: list[str]) -> dict:
    cols = predictors + ["log_revenue", "rd_intensity"]
    df = pdf.dropna(subset=["gross_margin", *cols, "year"]).copy()
    for c in predictors:
        df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
    df["year_c"] = df["year"] - df["year"].mean()
    df["year_c2"] = df["year_c"] ** 2
    X = sm.add_constant(df[[c + "_z" for c in predictors]
                           + ["log_revenue", "rd_intensity", "year_c", "year_c2"]])
    m = sm.OLS(df["gross_margin"], X).fit(cov_type="cluster",
                                          cov_kwds={"groups": df["cik"]})
    out = {"n": int(len(df)), "n_ciks": int(df["cik"].nunique())}
    for c in predictors:
        out[f"coef_{c}"] = float(m.params[c + "_z"])
        out[f"se_{c}"] = float(m.bse[c + "_z"])
        out[f"t_{c}"] = float(m.params[c + "_z"] / m.bse[c + "_z"])
    return out


def main() -> int:
    results = {}
    for tag, fy in (("topic", firm_year_topic()), ("token_same_years", firm_year_token())):
        pdf = build_panel(fy)
        results[tag] = {
            "dkl": fit(pdf, ["mean_dkl_3y"]),
            "decomposed": fit(pdf, ["mean_pros_3y", "mean_retr_3y"]),
        }
        d = results[tag]["dkl"]
        print(f"[{tag}] n={d['n']:,} ciks={d['n_ciks']:,} "
              f"coef={d['coef_mean_dkl_3y']:+.4f} t={d['t_mean_dkl_3y']:+.2f}",
              file=sys.stderr, flush=True)
        dd = results[tag]["decomposed"]
        print(f"        pros {dd['coef_mean_pros_3y']:+.4f} (t {dd['t_mean_pros_3y']:+.1f})"
              f"  retr {dd['coef_mean_retr_3y']:+.4f} (t {dd['t_mean_retr_3y']:+.1f})",
              file=sys.stderr, flush=True)
    (RES / f"topic_firm_margin{SUFFIX}.json").write_text(json.dumps(results, indent=1, default=float))
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
