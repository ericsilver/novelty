"""Regress 1y..5y stock returns on filing novelty.

Two specifications:
  (A) Firm-year level. For each (CIK, filing year), mean dKL across the
      firm's filings that year is the predictor; outcome is the stock
      return from end-of-filing-year to end-of-(filing-year + k).
  (B) Debut-firm level. The firm's first-ever filing supplies the
      predictor (one observation per firm); outcome is the same.

In both, we regress raw return AND excess return over the S&P 500.

Outputs:
  paper/results/returns_table.tex
  paper/results/returns_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"


def load_filings_with_dkl() -> pl.DataFrame:
    parts = []
    for path in sorted((REPO_ROOT / "data/processed").glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        df = pl.read_parquet(path)
        parts.append(df)
    return pl.concat(parts).filter(
        pl.col("owner_name").is_not_null()
        & (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= 1995)
        & (pl.col("year") <= 2020)
    )


def fit(pdf, predictor: str, outcome: str, label: str, cik_col: str = "cik"):
    if outcome not in pdf.columns or predictor not in pdf.columns:
        return None
    d = pdf[[predictor, outcome, "base_year", cik_col]].dropna().copy()
    if len(d) < 200:
        return None
    # Winsorize the outcome at the 1st and 99th percentile to deflate
    # influence of outlier returns (penny-stock spikes, splits/mergers
    # mishandled by yfinance, etc.)
    lo, hi = d[outcome].quantile([0.01, 0.99])
    d[outcome] = d[outcome].clip(lo, hi)
    d[f"{predictor}_z"] = (d[predictor] - d[predictor].mean()) / d[predictor].std()
    d["year_c"] = d["base_year"] - d["base_year"].mean()
    d["year_c2"] = d["year_c"] ** 2
    X = sm.add_constant(d[[f"{predictor}_z", "year_c", "year_c2"]])
    cluster_kwds = {"groups": d[cik_col]} if d[cik_col].nunique() > 1 else None
    if cluster_kwds:
        m = sm.OLS(d[outcome], X).fit(cov_type="cluster", cov_kwds=cluster_kwds)
    else:
        m = sm.OLS(d[outcome], X).fit(cov_type="HC1")
    return {
        "label": label,
        "outcome": outcome,
        "predictor": predictor,
        "n": int(len(d)),
        "coef": float(m.params[f"{predictor}_z"]),
        "se": float(m.bse[f"{predictor}_z"]),
        "n_clusters": int(d[cik_col].nunique()),
    }


def main() -> int:
    print("[load] surprise + crosswalk + returns…", flush=True)
    fil = load_filings_with_dkl()
    cw = pl.read_parquet(REPO_ROOT / "data/processed/uspto_sec_crosswalk.parquet").select("owner_name", "cik")
    rets = pl.read_parquet(REPO_ROOT / "data/processed/stock_returns.parquet")

    matched = fil.join(cw, on="owner_name", how="inner")
    print(f"       {matched.height:,} matched filings ({matched['cik'].n_unique():,} CIKs)")

    # ---- (A) Firm-year mean dKL ----
    fy = matched.group_by(["cik", "year"]).agg(
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.col("prospective_kl").mean().alias("mean_pros"),
        pl.col("retrospective_kl").mean().alias("mean_retr"),
        pl.len().alias("n_filings_yr"),
    ).rename({"year": "base_year"})

    panel_a = fy.join(rets, on=["cik", "base_year"], how="inner")
    print(f"       firm-year panel: {panel_a.height:,} rows × {panel_a['cik'].n_unique():,} firms")

    # Excess returns
    for k in (1, 2, 3, 4, 5):
        panel_a = panel_a.with_columns(
            (pl.col(f"ret_{k}y") - pl.col(f"sp500_ret_{k}y")).alias(f"excess_{k}y")
        )

    pdf_a = panel_a.to_pandas()

    # ---- (B) Debut filing per firm ----
    debut = matched.sort("filing_date").unique(subset=["owner_name"], keep="first")
    debut_firm = debut.group_by("cik").agg(
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.col("prospective_kl").mean().alias("mean_pros"),
        pl.col("retrospective_kl").mean().alias("mean_retr"),
        pl.col("year").min().alias("base_year"),
    )
    panel_b = debut_firm.join(rets, on=["cik", "base_year"], how="inner")
    print(f"       firm-debut panel: {panel_b.height:,} rows ({panel_b['cik'].n_unique():,} firms)")
    for k in (1, 2, 3, 4, 5):
        panel_b = panel_b.with_columns(
            (pl.col(f"ret_{k}y") - pl.col(f"sp500_ret_{k}y")).alias(f"excess_{k}y")
        )
    pdf_b = panel_b.to_pandas()

    rows = []
    for k in (1, 2, 3, 4, 5):
        for predictor in ("mean_dkl", "mean_pros"):
            for outcome in (f"ret_{k}y", f"excess_{k}y"):
                r1 = fit(pdf_a, predictor, outcome, f"FirmYear[{k}y]/{predictor}/{outcome}")
                r2 = fit(pdf_b, predictor, outcome, f"Debut[{k}y]/{predictor}/{outcome}")
                if r1: rows.append({**r1, "spec": "firm_year", "horizon": k})
                if r2: rows.append({**r2, "spec": "debut",     "horizon": k})

    pretty_pred = {"mean_dkl": "$\\Delta KL$", "mean_pros": "Prospective KL"}
    pretty_out = {
        **{f"ret_{k}y": f"raw {k}y return" for k in (1,2,3,4,5)},
        **{f"excess_{k}y": f"excess vs.\\ S\\&P 500, {k}y" for k in (1,2,3,4,5)},
    }

    by_spec_pred: dict = {}
    for r in rows:
        by_spec_pred.setdefault((r["spec"], r["predictor"]), []).append(r)

    body = ""
    for (spec, pred), rs in sorted(by_spec_pred.items()):
        spec_label = "Firm-year mean" if spec == "firm_year" else "Debut filing only"
        body += (
            "\\midrule\n"
            f"\\multicolumn{{4}}{{l}}{{\\textit{{{spec_label}, predictor = {pretty_pred[pred]}}}}} \\\\\n"
            "\\midrule\n"
        )
        rs.sort(key=lambda r: (r["horizon"], r["outcome"]))
        for r in rs:
            body += (
                f"{pretty_out[r['outcome']]} & {r['n']:,} & "
                f"{r['coef']:+.4f} & {r['se']:.4f} \\\\\n"
            )
    tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Outcome & N & $\\beta$ (per $\\sigma$) & SE \\\\\n"
        + body
        + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "returns_table.tex").write_text(tex)

    (RESULTS / "returns_metrics.json").write_text(
        json.dumps({"panel_a_n": panel_a.height, "panel_b_n": panel_b.height, "rows": rows}, indent=2, default=str)
    )

    print("\n[results] selected:")
    for r in rows:
        if r["predictor"] == "mean_dkl" and r["outcome"].startswith("excess"):
            print(f"  {r['spec']:<10} {r['outcome']:<14} N={r['n']:>6,} β={r['coef']:+.4f} (SE {r['se']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
