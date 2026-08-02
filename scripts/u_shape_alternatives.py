"""Test five candidate alternative explanations for the U-shape:

  (1) Foreign-extension hypothesis: are the very-stale filings just
      international marks being extended to the US (Madrid Protocol /
      Section 44 routes)? Use owner-name foreign-country indicators
      (Ltd, GmbH, S.A., AB, OY, KK, BV, NV, PLC, Pty, Pte, etc.) as
      a proxy. Refit the U-shape with and without foreign-owner
      filings.

  (2) First-time vs recurring filings: is the U-shape in BOTH?
      Already partially answered in first_filing.py (debut shows a
      sign-flip on reach_registration); here we report side-by-side.

  (3) Financial U-shape: re-fit the firm-year gross-margin regression
      with both dKL and dKL^2. If only the linear coefficient is
      significant, the U-shape lives only in survival.

  (4) Survivor bias: are bottom-decile dKL filings dominated by
      mega-cap incumbents (filing thousands of marks per decade)?
      Compute the average filer-size in each decile.

  (5) The "Liquid Death" check: among debut filings, what is the
      survival rate for filings with very short G/S text (like
      "still water; sparkling water")? Are short, generic-looking
      filings that survive disproportionately driving the
      bottom-decile arm of the U?

Outputs:
  paper/results/u_shape_alternatives.tex
  paper/results/u_shape_alternatives_metrics.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1990) & (pl.col("year") <= 2020)
)

# Owner-name indicators of likely foreign-domiciled firms
FOREIGN_RE = re.compile(
    r"\b("
    r"GMBH|GMBH\.|GMBH\\&\\sCO|S\\.?A\\.?|"
    r"S\\.?R\\.?L\\.?|S\\.?P\\.?A\\.?|S\\.?A\\.?S\\.?|"
    r"BV|B\\.?V\\.?|NV|N\\.?V\\.?|"
    r"AB|OY|KK|KABUSHIKI\\sKAISHA|"
    r"PTY|PTE|PLC|"
    r"AKTIENGESELLSCHAFT|AKTIEBOLAG|"
    r"SOCIETE|SOCIEDAD|SOCIETA|"
    r"LIMITADA|LTDA|LIMITED\\sCOMPANY|"
    r"BHD\\.?|SDN\\.?\\sBHD"
    r")\b",
    re.IGNORECASE,
)


def load_outcomes() -> pl.DataFrame:
    """Outcomes only -- no raw text. Saves ~8 GB of memory."""
    out_parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit(): continue
        out_parts.append(pl.read_parquet(path).filter(CLEAN).with_columns(pl.lit(cls).alias("nice_class")))
    return pl.concat(out_parts)


def load_text_for(serials: set[str]) -> pl.DataFrame:
    """Load goods/services text only for a small set of serials."""
    rec_parts = []
    for path in sorted(PROC.glob("tm_class*.parquet")):
        cls = path.stem.replace("tm_class", "")
        if not cls.isdigit(): continue
        df = pl.read_parquet(path).filter(pl.col("serial_number").is_in(serials)).select("serial_number", "goods_services")
        if df.height: rec_parts.append(df)
    return pl.concat(rec_parts) if rec_parts else pl.DataFrame()


def fit_quad(d: pd.DataFrame, outcome: str):
    if len(d) < 1000:
        return None
    d = d.copy()
    d["dkl_z"] = (d["dkl"] - d["dkl"].mean()) / d["dkl"].std()
    d["dkl_z_sq"] = d["dkl_z"] ** 2
    d["year_c"] = d["year"] - d["year"].mean()
    d["year_c2"] = d["year_c"] ** 2
    X = sm.add_constant(d[["dkl_z", "dkl_z_sq", "year_c", "year_c2"]])
    m = sm.GLM(d[outcome], X, family=sm.families.Binomial()).fit(disp=False, maxiter=200)
    return {
        "n": int(len(d)),
        "beta_dkl": float(m.params["dkl_z"]),
        "se_dkl": float(m.bse["dkl_z"]),
        "beta_dkl_sq": float(m.params["dkl_z_sq"]),
        "se_dkl_sq": float(m.bse["dkl_z_sq"]),
    }


def main() -> int:
    print("[load] outcomes…", flush=True)
    df_pl = load_outcomes()
    print(f"  {df_pl.height:,} clean filings", flush=True)
    # Sample once for the GLM splits; don't pull text into pandas
    sampled = df_pl.sample(n=min(600_000, df_pl.height), seed=11)
    pdf = sampled.select(
        "serial_number", "owner_name", "dkl", "year", "n_terms",
        "reached_registration", "survived_5y",
    ).to_pandas()
    pdf["reached_registration"] = pdf["reached_registration"].astype(int)
    pdf["survived_5y"] = pdf["survived_5y"].astype(int)
    pdf["foreign_proxy"] = pdf["owner_name"].astype(str).str.contains(FOREIGN_RE, regex=True, na=False)
    print(f"  share of sample with foreign-owner proxy: {pdf['foreign_proxy'].mean()*100:.1f}%")

    metrics = {}
    rows_for_table = []

    # ---- (1) Foreign-extension hypothesis ----
    for label, sub in [
        ("Full sample", pdf),
        ("US-domiciled only (foreign proxy = False)", pdf[~pdf["foreign_proxy"]]),
        ("Foreign-domiciled only (foreign proxy = True)", pdf[pdf["foreign_proxy"]]),
    ]:
        if len(sub) < 1000: continue
        d = sub[sub["year"] <= 2020]
        res = fit_quad(d, "survived_5y")
        if res is None: continue
        rate = float(d["survived_5y"].mean())
        rows_for_table.append({
            "section": "Foreign-extension hypothesis",
            "label": label, "n": res["n"], "rate": rate,
            "beta_dkl": res["beta_dkl"], "se_dkl": res["se_dkl"],
            "beta_dkl_sq": res["beta_dkl_sq"], "se_dkl_sq": res["se_dkl_sq"],
        })
        print(f"  {label:<55}  n={res['n']:>8,}  rate={rate*100:.1f}%  beta_lin={res['beta_dkl']:+.4f}  beta_quad={res['beta_dkl_sq']:+.4f}")

    # ---- (2) First-time vs recurring filings ----
    pdf = pdf.sort_values(["owner_name", "year"])
    pdf["filing_idx"] = pdf.groupby("owner_name").cumcount()
    pdf["is_first"] = pdf["filing_idx"] == 0
    for label, sub in [
        ("First-time filings only", pdf[pdf["is_first"]]),
        ("Recurring (2nd+) filings only", pdf[~pdf["is_first"]]),
    ]:
        if len(sub) < 1000: continue
        d = sub[sub["year"] <= 2020]
        res = fit_quad(d, "survived_5y")
        if res is None: continue
        rate = float(d["survived_5y"].mean())
        rows_for_table.append({
            "section": "First-time vs recurring",
            "label": label, "n": res["n"], "rate": rate,
            "beta_dkl": res["beta_dkl"], "se_dkl": res["se_dkl"],
            "beta_dkl_sq": res["beta_dkl_sq"], "se_dkl_sq": res["se_dkl_sq"],
        })
        print(f"  {label:<55}  n={res['n']:>8,}  rate={rate*100:.1f}%  beta_lin={res['beta_dkl']:+.4f}  beta_quad={res['beta_dkl_sq']:+.4f}")

    # ---- (3) Financial U-shape: refit gross-margin OLS with quadratic dKL ----
    print("\n[3] Financial U-shape: gross_margin ~ dKL + dKL^2", flush=True)
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name", "cik")
    matched = df_pl.join(cw, on="owner_name", how="inner")
    fy = matched.group_by(["cik", "year"]).agg(
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.len().alias("n_filings_yr"),
    ).rename({"year": "fy"})
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet").filter(
        (pl.col("revenue").is_not_null()) & (pl.col("revenue") > 0)
        & pl.col("gross_margin").is_not_null()
        & (pl.col("gross_margin") > -1) & (pl.col("gross_margin") < 1.5)
    ).select("cik", "fy", "revenue", "gross_margin")
    panel = fy.join(sec, on=["cik", "fy"], how="inner").with_columns(pl.col("revenue").log().alias("log_revenue"))
    pdf_fy = panel.to_pandas()
    pdf_fy["dkl_z"] = (pdf_fy["mean_dkl"] - pdf_fy["mean_dkl"].mean()) / pdf_fy["mean_dkl"].std()
    pdf_fy["dkl_z_sq"] = pdf_fy["dkl_z"] ** 2
    pdf_fy["year_c"] = pdf_fy["fy"] - pdf_fy["fy"].mean()
    Xf = sm.add_constant(pdf_fy[["dkl_z", "dkl_z_sq", "log_revenue", "year_c"]])
    mf = sm.OLS(pdf_fy["gross_margin"], Xf).fit(cov_type="cluster", cov_kwds={"groups": pdf_fy["cik"]})
    print(f"  n={len(pdf_fy):,}  beta_lin={mf.params['dkl_z']:+.4f} (SE {mf.bse['dkl_z']:.4f})  beta_quad={mf.params['dkl_z_sq']:+.4f} (SE {mf.bse['dkl_z_sq']:.4f})  R^2={mf.rsquared:.4f}")
    fin_quad = {
        "n": len(pdf_fy),
        "beta_lin": float(mf.params["dkl_z"]), "se_lin": float(mf.bse["dkl_z"]),
        "beta_quad": float(mf.params["dkl_z_sq"]), "se_quad": float(mf.bse["dkl_z_sq"]),
        "rsq": float(mf.rsquared),
    }
    metrics["financial_quadratic"] = fin_quad

    # ---- (4) Survivor bias: average firm filing-volume by dKL decile ----
    print("\n[4] Survivor bias: avg firm-filing-volume by dKL decile", flush=True)
    firm_size_pl = df_pl.group_by("owner_name").len().rename({"len": "firm_size"})
    decile_pdf = df_pl.select("owner_name", "dkl").join(firm_size_pl, on="owner_name").to_pandas()
    edges = decile_pdf["dkl"].quantile(np.linspace(0, 1, 11)).values
    decile_pdf["dec"] = np.digitize(decile_pdf["dkl"], edges[1:-1])
    grp = decile_pdf.groupby("dec").agg(
        x=("dkl", "mean"),
        mean_firm_size=("firm_size", "mean"),
        median_firm_size=("firm_size", "median"),
        n=("firm_size", "size"),
    ).reset_index()
    print(f"  decile  mean_dKL  mean_firm_size  median_firm_size  n")
    for r in grp.itertuples():
        print(f"  D{int(r.dec)+1:>2}     {r.x:+.3f}     {r.mean_firm_size:>10.1f}        {r.median_firm_size:>3.0f}        {r.n:,}")
    metrics["survivor_bias"] = grp.to_dict(orient="records")

    # ---- (5) "Liquid Death" check: short-text bottom-decile filings ----
    print("\n[5] 'Liquid Death' check: are short-text bottom-decile filings high-survival?", flush=True)
    bot_serials = set(df_pl.filter(pl.col("dkl") <= edges[1])["serial_number"].to_list())
    bot_text = load_text_for(bot_serials)
    bot_join = df_pl.filter(pl.col("dkl") <= edges[1]).join(bot_text, on="serial_number", how="left").to_pandas()
    bot_join["gs_len"] = bot_join["goods_services"].astype(str).str.len()
    bot_join["gs_quartile"] = pd.qcut(bot_join["gs_len"], 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
    print(f"  bottom-decile dKL filings (n={len(bot_join):,}):")
    for q, sub in bot_join.groupby("gs_quartile", observed=True):
        print(f"    {q:<10}  n={len(sub):>6,}  mean gs_len={sub['gs_len'].mean():.0f}  surv5={sub[sub['year']<=2020]['survived_5y'].mean()*100:.1f}%  reach={sub['reached_registration'].mean()*100:.1f}%")

    # ---- LaTeX table for the survival U-shape across the (1) + (2) splits ----
    body = ""
    last_section = None
    for r in rows_for_table:
        if r["section"] != last_section:
            body += f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\textit{{{r['section']}}}}} \\\\\n\\midrule\n"
            last_section = r["section"]
        body += (
            f"{r['label']} & {r['n']:,} & {r['rate']*100:.1f}\\% & "
            f"{r['beta_dkl']:+.4f} ({r['se_dkl']:.4f}) & "
            f"{r['beta_dkl_sq']:+.4f} ({r['se_dkl_sq']:.4f}) \\\\\n"
        )
    # Add financial quadratic
    body += f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\textit{{Financial U-shape: firm-year gross margin}}}} \\\\\n\\midrule\n"
    body += (
        f"Public firms (gross margin OLS) & {fin_quad['n']:,} & --- & "
        f"{fin_quad['beta_lin']:+.4f} ({fin_quad['se_lin']:.4f}) & "
        f"{fin_quad['beta_quad']:+.4f} ({fin_quad['se_quad']:.4f}) \\\\\n"
    )
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Sub-sample & N & Outcome rate & "
        "Linear $\\beta_{\\Delta KL}^z$ & Quadratic $\\beta_{\\Delta KL^2}^z$ \\\\\n"
        + body + "\\bottomrule\n\\end{tabular}%\n}\n"
    )
    (RESULTS / "u_shape_alternatives.tex").write_text(tex)

    metrics["splits"] = rows_for_table
    (RESULTS / "u_shape_alternatives_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\n[done] wrote u_shape_alternatives.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
