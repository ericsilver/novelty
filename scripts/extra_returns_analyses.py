"""Three additions to the returns analysis:
  1. Mean cumulative excess return by years-since-debut, split by debut
     dKL quartile (cohort plot).
  2. Per-industry correlation between firm-mean dKL and firm cumulative
     return (over the firm's whole observation window).
  3. Patent-counts-as-predictor regressions, parallel to the dKL ones,
     so the patent and trademark predictors can be compared on the same
     return outcomes.

Outputs:
  paper/results/cohort_returns.png
  paper/results/industry_returns_corr.tex
  paper/results/patent_returns_table.tex
  paper/results/extra_returns_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"
PROC = REPO_ROOT / "data" / "processed"


def load_filings_with_dkl() -> pl.DataFrame:
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        df = pl.read_parquet(path).with_columns(pl.lit(cls).alias("nice_class"))
        parts.append(df)
    return pl.concat(parts).filter(
        pl.col("owner_name").is_not_null()
        & (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= 1995) & (pl.col("year") <= 2020)
    )


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    fil = load_filings_with_dkl()
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name", "cik")
    rets = pl.read_parquet(PROC / "stock_returns.parquet")
    matched = fil.join(cw, on="owner_name", how="inner")

    debut_firm = matched.sort("filing_date").unique(subset=["owner_name"], keep="first").select(
        "owner_name", "cik", "year", "dkl", "prospective_kl"
    ).rename({"year": "debut_year", "dkl": "debut_dkl"})
    debut_firm = debut_firm.group_by("cik").agg(
        pl.col("debut_year").min(),
        pl.col("debut_dkl").mean(),
    )

    # ----- (1) Cohort plot: cum excess return by year-since-debut, by debut dkl quartile
    rets_pd = rets.to_pandas().sort_values(["cik", "base_year"])
    rets_pd["excess_1y"] = rets_pd["ret_1y"] - rets_pd["sp500_ret_1y"]
    rets_pd = rets_pd.dropna(subset=["excess_1y"])
    debut_pd = debut_firm.to_pandas()
    debut_pd["q"] = pd.qcut(debut_pd["debut_dkl"], 4, labels=["Q1 (least novel)", "Q2", "Q3", "Q4 (most novel)"])

    panel = rets_pd.merge(debut_pd[["cik", "debut_year", "q"]], on="cik")
    panel["years_since_debut"] = panel["base_year"] - panel["debut_year"] + 1  # base_year+1 = end of return window
    panel = panel[panel["years_since_debut"].between(0, 12)]
    # Winsorize per horizon
    lo, hi = panel["excess_1y"].quantile([0.01, 0.99])
    panel["excess_1y"] = panel["excess_1y"].clip(lo, hi)

    # Cumulative excess return per firm by chaining 1y returns
    panel = panel.sort_values(["cik", "base_year"])
    panel["log_excess"] = np.log1p(panel["excess_1y"])
    panel["cum_log"] = panel.groupby("cik")["log_excess"].cumsum()
    panel["cum_excess"] = np.expm1(panel["cum_log"])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = {"Q1 (least novel)": "#7a1111", "Q2": "#bf6b3a", "Q3": "#3a8bbf", "Q4 (most novel)": "#117a3a"}
    for q in ["Q1 (least novel)", "Q2", "Q3", "Q4 (most novel)"]:
        sub = panel[panel["q"] == q]
        agg = sub.groupby("years_since_debut")["cum_excess"].agg(["mean", "count", "sem"])
        agg = agg[agg["count"] >= 50]
        ax.plot(agg.index, agg["mean"] * 100, "o-", color=colors[q], label=q, linewidth=1.6, markersize=5)
        ax.fill_between(
            agg.index,
            (agg["mean"] - 1.96 * agg["sem"]) * 100,
            (agg["mean"] + 1.96 * agg["sem"]) * 100,
            color=colors[q], alpha=0.12,
        )
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Years after firm's first trademark filing")
    ax.set_ylabel("Mean cumulative excess return over S&P 500 (%)")
    ax.set_title("Cumulative excess return by debut-$\\Delta KL$ quartile")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "cohort_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[cohort] saved cohort_returns.png ({len(panel):,} firm-year rows)")

    # ----- (2) Per-industry correlation -----
    fy_cls = matched.with_columns(pl.col("year").alias("base_year")).group_by(["cik", "nice_class"]).agg(
        pl.col("dkl").mean().alias("firm_mean_dkl"),
        pl.col("prospective_kl").mean().alias("firm_mean_pros"),
        pl.len().alias("n_filings"),
    )
    # Firm cumulative return = mean of 1y excess return windows we have
    cum = panel.groupby("cik").agg(
        firm_mean_excess=("excess_1y", "mean"),
        firm_n_years=("excess_1y", "count"),
    ).reset_index()
    cum_pl = pl.from_pandas(cum)

    rows = []
    for cls_str in sorted(set(fy_cls["nice_class"].to_list())):
        sub = fy_cls.filter(pl.col("nice_class") == cls_str).join(cum_pl, on="cik", how="inner").filter(pl.col("firm_n_years") >= 3)
        if sub.height < 30:
            continue
        x = sub["firm_mean_dkl"].to_numpy()
        y = sub["firm_mean_excess"].to_numpy()
        xp = sub["firm_mean_pros"].to_numpy()
        if x.std() == 0 or y.std() == 0:
            continue
        r_dkl = float(np.corrcoef(x, y)[0, 1])
        r_pros = float(np.corrcoef(xp, y)[0, 1])
        # one-sided two-sample t-test of correlation: t = r * sqrt((n-2)/(1-r^2))
        n = len(x)
        from scipy import stats as st
        # use scipy.stats.pearsonr for p-value
        _, p_dkl = st.pearsonr(x, y)
        _, p_pros = st.pearsonr(xp, y)
        rows.append({
            "cls": cls_str, "industry": industry_name(cls_str),
            "n_firms": int(n), "r_dkl": r_dkl, "p_dkl": float(p_dkl),
            "r_pros": r_pros, "p_pros": float(p_pros),
        })
    rows.sort(key=lambda r: -r["n_firms"])

    # Top-15 by industry size
    body = ""
    for r in rows[:20]:
        ind = r["industry"].replace("&", r"\&")
        sig_dkl = "*" if r["p_dkl"] < 0.05 else ""
        sig_pros = "*" if r["p_pros"] < 0.05 else ""
        body += (
            f"{ind} ({r['cls']}) & {r['n_firms']:,} & "
            f"{r['r_dkl']:+.3f}{sig_dkl} & {r['p_dkl']:.3f} & "
            f"{r['r_pros']:+.3f}{sig_pros} & {r['p_pros']:.3f} \\\\\n"
        )
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Industry (NICE) & N firms & "
        "$r$ ($\\Delta KL$, mean excess) & $p$ & "
        "$r$ (pros KL, mean excess) & $p$ \\\\\n\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}%\n}\n"
    )
    (RESULTS / "industry_returns_corr.tex").write_text(tex)
    print(f"[indcorr] {len(rows)} industries; wrote top-20 to industry_returns_corr.tex")

    # ----- (3) Patent counts as predictor -----
    pat = pl.read_parquet(PROC / "patent_firm_year.parquet")  # patentsview_name, year, n_patents, norm
    # Need a CIK map for patents. Re-use the normalized-name link via crosswalk.
    cw_full = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("norm", "cik")
    pat_with_cik = pat.join(cw_full, on="norm", how="inner").select("cik", "year", "n_patents").rename({"year": "base_year"})
    pat_yr = pat_with_cik.group_by(["cik", "base_year"]).agg(pl.col("n_patents").sum())
    panel_pat = pat_yr.join(rets, on=["cik", "base_year"], how="inner")
    for k in (1, 2, 3, 4, 5):
        panel_pat = panel_pat.with_columns((pl.col(f"ret_{k}y") - pl.col(f"sp500_ret_{k}y")).alias(f"excess_{k}y"))

    pdf_p = panel_pat.to_pandas()
    pdf_p["log1p_n_patents"] = np.log1p(pdf_p["n_patents"])

    pat_rows = []
    for k in (1, 2, 3, 4, 5):
        d = pdf_p[["log1p_n_patents", f"excess_{k}y", "base_year", "cik"]].dropna().copy()
        if len(d) < 200:
            continue
        lo, hi = d[f"excess_{k}y"].quantile([0.01, 0.99])
        d[f"excess_{k}y"] = d[f"excess_{k}y"].clip(lo, hi)
        d["pat_z"] = (d["log1p_n_patents"] - d["log1p_n_patents"].mean()) / d["log1p_n_patents"].std()
        d["year_c"] = d["base_year"] - d["base_year"].mean()
        d["year_c2"] = d["year_c"] ** 2
        X = sm.add_constant(d[["pat_z", "year_c", "year_c2"]])
        m = sm.OLS(d[f"excess_{k}y"], X).fit(cov_type="cluster", cov_kwds={"groups": d["cik"]})
        pat_rows.append({"horizon": k, "n": int(len(d)), "coef": float(m.params["pat_z"]), "se": float(m.bse["pat_z"])})

    body = ""
    for r in pat_rows:
        body += f"{r['horizon']}y & {r['n']:,} & {r['coef']:+.4f} & {r['se']:.4f} \\\\\n"
    tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Horizon & N & $\\beta$ for $\\log(1+\\text{patents})_z$ & SE \\\\\n\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "patent_returns_table.tex").write_text(tex)

    print(f"\n[patent-returns] (excess vs S&P, log1p patent count z-scored)")
    for r in pat_rows:
        print(f"  {r['horizon']}y: β={r['coef']:+.4f} (SE {r['se']:.4f}) n={r['n']:,}")

    metrics = {
        "industry_corr_top20": rows[:20],
        "patent_returns": pat_rows,
    }
    (RESULTS / "extra_returns_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[done] wrote {RESULTS}/cohort_returns.png + industry_returns_corr.tex + patent_returns_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
