"""Re-score the within-firm patent table and the BCG/MIT expert-list
discriminant under token, topic T=50, and topic T=200 scoring, with all
three owner-year panels built symmetrically from per-class scores
(years 1995-2019, all 45 classes).

Patents: replicate wsC's collapse + two-way FE spec on the published
patent join (data_publish/firm_year_patents_and_dkl.csv supplies
n_patents per owner-year; the dKL column is replaced per scoring).

Expert lists: replicate wsD's population/listed contrast with firm-level
means per CIK via the USPTO-SEC crosswalk and the comparators crosswalk.

Output: paper/results/topic_patent_expert_rescore.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
PUB = REPO / "data_publish"
RES = REPO / "paper" / "results"

YEAR_LO, YEAR_HI = 1995, 2019


def all_classes() -> list[str]:
    return sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def owner_year_panel(kind: str) -> pl.DataFrame:
    """kind in {token, t50, t200}: owner-year mean dkl + scored filing count."""
    parts = []
    for cls in all_classes():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                             columns=["serial_number", "owner_name"])
        if kind == "token":
            s = pl.read_parquet(
                PROC / f"surprise_class{cls}.parquet",
                columns=["serial_number", "year", "kl_vs_past",
                         "kl_vs_future", "n_ref_past",
                         "n_ref_future", "n_terms"],
            ).filter(
                (pl.col("n_ref_past") >= 1000)
                & (pl.col("n_ref_future") >= 1000)
                & (pl.col("n_terms") >= 3)
                & pl.col("kl_vs_past").is_finite()
                & pl.col("kl_vs_future").is_finite()
                & pl.col("year").is_between(YEAR_LO, YEAR_HI)
            ).with_columns(
                (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"))
        else:
            suffix = "" if kind == "t50" else "_T200"
            s = pl.read_parquet(
                PROC / f"topic_surprise_class{cls}{suffix}.parquet"
            ).filter(pl.col("topic_dkl").is_finite()
                     & pl.col("year").is_between(YEAR_LO, YEAR_HI)
            ).rename({"topic_dkl": "dkl"})
        j = s.select("serial_number", "year", "dkl").join(
            tm, on="serial_number", how="inner").filter(
            pl.col("owner_name").is_not_null())
        parts.append(j.group_by(["owner_name", "year"]).agg(
            pl.len().alias("n_scored"), pl.col("dkl").mean().alias("mean_dkl")))
        del tm, s, j
        gc.collect()
    return pl.concat(parts).group_by(["owner_name", "year"]).agg(
        (pl.col("mean_dkl") * pl.col("n_scored")).sum().alias("_wd"),
        pl.col("n_scored").sum().alias("n_scored"),
    ).with_columns((pl.col("_wd") / pl.col("n_scored")).alias("mean_dkl")).drop("_wd")


def two_way_fe(df, yvar, xvars, firm_col, year_col):
    d = df[[firm_col, year_col, yvar] + xvars].dropna().copy()

    def demean(s):
        return s - s.groupby(d[firm_col]).transform("mean")

    yd = demean(d[yvar])
    Xparts = {x: demean(d[x]) for x in xvars}
    yr = pd.get_dummies(d[year_col].astype("category"), drop_first=True).astype(float)
    yr.index = d.index
    yr_d = yr.apply(lambda col: col - col.groupby(d[firm_col]).transform("mean"))
    X = pd.concat([pd.DataFrame(Xparts, index=d.index), yr_d], axis=1).astype(float)
    res = sm.OLS(yd.astype(float), X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[firm_col].astype("category").cat.codes.values})
    out = {x: dict(coef=float(res.params[x]), se=float(res.bse[x]),
                   t=float(res.tvalues[x])) for x in xvars}
    out["n_obs"] = int(len(d))
    out["n_firms"] = int(d[firm_col].nunique())
    return out


def patents_block(panels: dict[str, pl.DataFrame]) -> dict:
    pat = pd.read_csv(PUB / "firm_year_patents_and_dkl.csv")
    pat["uspto_owner_name"] = pat.uspto_owner_name.fillna("").astype(str).str.strip()
    pat = pat[pat.uspto_owner_name != ""]
    pat = pat.groupby(["uspto_owner_name", "year"], as_index=False).agg(
        n_patents=("n_patents", "mean"))
    out = {}
    for kind, oy in panels.items():
        oyp = oy.to_pandas()
        m = oyp.merge(pat, left_on=["owner_name", "year"],
                      right_on=["uspto_owner_name", "year"], how="inner")
        m["logpat"] = np.log1p(m.n_patents)
        sd = m.mean_dkl.std()
        m["dkl_z"] = (m.mean_dkl - m.mean_dkl.mean()) / sd
        yr_counts = m.groupby("owner_name").year.transform("count")
        p3 = m[yr_counts >= 3].copy()
        fe = two_way_fe(p3, "dkl_z", ["logpat"], "owner_name", "year")
        out[kind] = {"sigma_dkl": float(sd),
                     "matched_firm_years": int(len(m)),
                     "fe_contemp": fe}
        print(f"[patents:{kind}] n={fe['n_obs']:,} firms={fe['n_firms']:,} "
              f"coef={fe['logpat']['coef']:+.4f} t={fe['logpat']['t']:+.2f}",
              file=sys.stderr, flush=True)
    return out


def experts_block(panels: dict[str, pl.DataFrame]) -> dict:
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik").unique().to_pandas()
    cw["cik"] = cw.cik.astype(str)
    xw = pd.read_csv(PUB / "comparators" / "external_crosswalk.csv")
    xw["matched_cik"] = xw.matched_cik.apply(
        lambda v: str(int(float(v))) if str(v).strip() not in ("", "nan") else "")
    xw = xw[xw.matched_cik != ""]
    out = {}
    for kind, oy in panels.items():
        oyp = oy.to_pandas().merge(cw, on="owner_name", how="inner")
        firm = oyp.groupby("cik").apply(
            lambda g: np.average(g.mean_dkl, weights=g.n_scored.clip(lower=1)),
            include_groups=False).rename("mean_dkl").reset_index()
        pop_mean, pop_sd = firm.mean_dkl.mean(), firm.mean_dkl.std()
        block = {"n_firms": int(len(firm)), "pop_mean": float(pop_mean),
                 "pop_sd": float(pop_sd), "lists": {}}
        for comp, g in list(xw.groupby("comparator")) + [("POOLED", xw)]:
            listed_ciks = set(g.matched_cik)
            listed = firm[firm.cik.isin(listed_ciks)].mean_dkl.values
            base = firm[~firm.cik.isin(listed_ciks)].mean_dkl.values
            if len(listed) < 5:
                continue
            mw = stats.mannwhitneyu(listed, base, alternative="two-sided")
            block["lists"][comp] = {
                "n_matched": int(len(listed)),
                "diff_in_pop_sd": float((listed.mean() - base.mean()) / pop_sd),
                "mannwhitney_p": float(mw.pvalue),
            }
        out[kind] = block
        pooled = block["lists"].get("POOLED", {})
        print(f"[experts:{kind}] n_firms={block['n_firms']:,} pooled "
              f"diff={pooled.get('diff_in_pop_sd', float('nan')):+.3f}sd "
              f"p={pooled.get('mannwhitney_p', float('nan')):.4f} "
              f"(n={pooled.get('n_matched')})",
              file=sys.stderr, flush=True)
    return out


def main() -> int:
    panels = {}
    for kind in ("token", "t50", "t200"):
        panels[kind] = owner_year_panel(kind)
        print(f"[panel:{kind}] {panels[kind].height:,} owner-years",
              file=sys.stderr, flush=True)
    results = {"patents": patents_block(panels), "experts": experts_block(panels)}
    (RES / "topic_patent_expert_rescore.json").write_text(
        json.dumps(results, indent=1, default=float))
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
