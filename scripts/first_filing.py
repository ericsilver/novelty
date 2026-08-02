"""First-filing-only analysis.

Two questions:
  1. Among each owner's FIRST trademark filing in our corpus, does novelty
     still predict registration success?
  2. For matched public firms, does the novelty of the FIRM'S DEBUT FILING
     predict the firm's average future gross margin? (Removes the
     correlated-success concern that current-period novelty is being
     funded by prior period success.)

Outputs:
  paper/results/first_filing_table.tex
  paper/results/first_filing_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"


def load_universe() -> pl.DataFrame:
    """One row per (owner, class, filing) across all 45 classes."""
    parts = []
    for path in sorted((REPO_ROOT / "data/processed").glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        df = pl.read_parquet(path).with_columns(pl.lit(cls).alias("nice_class"))
        parts.append(df)
    return pl.concat(parts)


def main() -> int:
    print("[load] all-class outcomes…", flush=True)
    universe = load_universe().filter(pl.col("owner_name").is_not_null())
    print(f"       {universe.height:,} owner-filings across all classes", flush=True)

    CLEAN = (
        (pl.col("n_ref_past") >= 1000)
        & (pl.col("n_ref_future") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= 1990)
        & (pl.col("year") <= 2020)
    )
    clean = universe.filter(CLEAN)
    print(f"       {clean.height:,} clean (1995-2020, refs populated)")

    # First filing per owner, across all classes
    first = clean.sort("filing_date").unique(subset=["owner_name"], keep="first")
    print(f"       {first.height:,} first-filings (one per owner)")

    # ---- Survival regression on first filings only ----
    rows = []
    for outcome, cap in (("reached_registration", 2025), ("survived_5y", 2020), ("survived_10y", 2015)):
        d = first.filter(pl.col("year") <= cap).select(
            pl.col(outcome).cast(pl.Int8),
            pl.col("dkl"),
            pl.col("kl_vs_past"),
            pl.col("kl_vs_future"),
            pl.col("n_terms").log().alias("log_n_terms"),
            pl.col("year"),
        ).drop_nulls().to_pandas()
        if len(d) < 1000:
            continue
        for col in ("dkl", "kl_vs_past", "kl_vs_future"):
            d[f"{col}_z"] = (d[col] - d[col].mean()) / d[col].std()
        d["year_c"] = d["year"] - d["year"].mean()
        d["year_c2"] = d["year_c"] ** 2

        # Model 1: dKL alone
        X1 = sm.add_constant(d[["dkl_z", "log_n_terms", "year_c", "year_c2"]])
        m1 = sm.GLM(d[outcome], X1, family=sm.families.Binomial()).fit(disp=False, maxiter=200)
        # Model 2: pros + retro jointly
        X2 = sm.add_constant(d[["kl_vs_past_z", "kl_vs_future_z", "log_n_terms", "year_c", "year_c2"]])
        m2 = sm.GLM(d[outcome], X2, family=sm.families.Binomial()).fit(disp=False, maxiter=200)
        rows.append(
            {
                "outcome": outcome,
                "n": len(d),
                "rate": float(d[outcome].mean()),
                "coef_dkl_z": float(m1.params["dkl_z"]),
                "se_dkl_z": float(m1.bse["dkl_z"]),
                "coef_pros_z": float(m2.params["kl_vs_past_z"]),
                "se_pros_z": float(m2.bse["kl_vs_past_z"]),
                "coef_retr_z": float(m2.params["kl_vs_future_z"]),
                "se_retr_z": float(m2.bse["kl_vs_future_z"]),
            }
        )

    # ---- Firm-debut financial analysis ----
    crosswalk = pl.read_parquet(REPO_ROOT / "data/processed/uspto_sec_crosswalk.parquet")
    debut = first.join(crosswalk.select("owner_name", "cik", "sec_name"), on="owner_name", how="inner")
    print(f"       {debut.height:,} debut filings matched to a CIK")

    # Average gross margin per CIK from SEC FSDS
    sec = pl.read_parquet(REPO_ROOT / "data/processed/sec_firm_year.parquet")
    firm_finance = sec.filter(
        (pl.col("gross_margin").is_not_null())
        & (pl.col("gross_margin") > -1) & (pl.col("gross_margin") < 1.5)
        & (pl.col("revenue") > 0)
    ).group_by("cik").agg(
        pl.col("gross_margin").mean().alias("mean_gross_margin"),
        pl.col("gross_margin").std().alias("std_gross_margin"),
        pl.col("revenue").mean().log().alias("mean_log_revenue"),
        pl.col("rd_expense").mean().alias("mean_rd"),
        pl.col("net_income").mean().alias("mean_ni"),
        pl.len().alias("n_firm_years"),
    )

    panel = debut.join(firm_finance, on="cik", how="inner").filter(pl.col("n_firm_years") >= 2)
    print(f"       {panel.height:,} firms in firm-debut financial panel")

    pdf = panel.select(
        pl.col("cik"), pl.col("dkl"), pl.col("kl_vs_past"), pl.col("kl_vs_future"),
        pl.col("year"), pl.col("mean_gross_margin"), pl.col("std_gross_margin"),
        pl.col("mean_log_revenue"), pl.col("n_firm_years"),
    ).drop_nulls(["dkl", "mean_gross_margin", "mean_log_revenue"]).to_pandas()
    if len(pdf) >= 100:
        for col in ("dkl", "kl_vs_past", "kl_vs_future"):
            pdf[f"{col}_z"] = (pdf[col] - pdf[col].mean()) / pdf[col].std()
        pdf["year_c"] = pdf["year"] - pdf["year"].mean()

        # OLS: mean future gross margin ~ debut dKL + controls
        X1 = sm.add_constant(pdf[["dkl_z", "mean_log_revenue", "year_c"]])
        f1 = sm.OLS(pdf["mean_gross_margin"], X1).fit(cov_type="HC1")
        X2 = sm.add_constant(pdf[["kl_vs_past_z", "kl_vs_future_z", "mean_log_revenue", "year_c"]])
        f2 = sm.OLS(pdf["mean_gross_margin"], X2).fit(cov_type="HC1")
        debut_fin = {
            "n_firms": int(len(pdf)),
            "outcome": "mean_gross_margin",
            "coef_dkl_z": float(f1.params["dkl_z"]),
            "se_dkl_z": float(f1.bse["dkl_z"]),
            "coef_pros_z": float(f2.params["kl_vs_past_z"]),
            "se_pros_z": float(f2.bse["kl_vs_past_z"]),
            "coef_retr_z": float(f2.params["kl_vs_future_z"]),
            "se_retr_z": float(f2.bse["kl_vs_future_z"]),
        }

        # Variance: std of margin ~ debut dKL
        var_mask = pdf["std_gross_margin"].notna() & (pdf["n_firm_years"] >= 4)
        v_pdf = pdf[var_mask].copy()
        if len(v_pdf) >= 100:
            Xv = sm.add_constant(v_pdf[["dkl_z", "mean_log_revenue"]])
            v_model = sm.OLS(v_pdf["std_gross_margin"], Xv).fit(cov_type="HC1")
            var_result = {
                "n_firms": int(len(v_pdf)),
                "coef_dkl_z": float(v_model.params["dkl_z"]),
                "se_dkl_z": float(v_model.bse["dkl_z"]),
            }
        else:
            var_result = None
    else:
        debut_fin = None
        var_result = None

    # ---- LaTeX table ----
    survival_lines = ""
    label = {"reached_registration": "Completed registration",
             "survived_5y": "Renewed 5y registration",
             "survived_10y": "Renewed 10y registration"}
    for r in rows:
        survival_lines += (
            f"{label[r['outcome']]} & {r['n']:,} & "
            f"{r['coef_dkl_z']:+.3f} ({r['se_dkl_z']:.3f}) \\\\\n"
        )

    fin_block = ""
    if debut_fin is not None:
        n_firms = debut_fin['n_firms']
        fin_block = (
            "\\midrule\n"
            "\\multicolumn{3}{l}{\\textit{Firm-debut financial regression "
            f"(N={n_firms:,} matched public firms)" + "}} \\\\\n"
            "\\midrule\n"
            f"Mean future gross margin & {n_firms:,} & "
            f"{debut_fin['coef_dkl_z']:+.3f} ({debut_fin['se_dkl_z']:.3f}) \\\\\n"
        )
        if var_result is not None:
            fin_block += (
                f"Within-firm gross-margin std.\\ & {var_result['n_firms']:,} & "
                f"{var_result['coef_dkl_z']:+.4f} ({var_result['se_dkl_z']:.4f}) \\\\\n"
            )

    tex = (
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Outcome & N & $\\beta$ for $\\Delta KL$ ($z$-scored) \\\\\n"
        "\\midrule\n"
        "\\multicolumn{3}{l}{\\textit{Survival on first filings (one per owner across all classes)}} \\\\\n"
        "\\midrule\n"
        + survival_lines
        + fin_block
        + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "first_filing_table.tex").write_text(tex)

    metrics = {"survival": rows, "debut_financial": debut_fin, "debut_variance": var_result}
    (RESULTS / "first_filing_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print("\n[done] wrote first_filing_table.tex + first_filing_metrics.json")
    print()
    for r in rows:
        print(f"  {r['outcome']:<25} n={r['n']:>10,}  β_dKL={r['coef_dkl_z']:+.3f} (SE {r['se_dkl_z']:.3f})  β_pros={r['coef_pros_z']:+.3f}  β_retr={r['coef_retr_z']:+.3f}")
    if debut_fin is not None:
        print(f"  Firm-debut financial: n={debut_fin['n_firms']:,} firms")
        print(f"    β_dKL={debut_fin['coef_dkl_z']:+.4f} (SE {debut_fin['se_dkl_z']:.4f})")
        print(f"    β_pros={debut_fin['coef_pros_z']:+.4f} (SE {debut_fin['se_pros_z']:.4f}); β_retr={debut_fin['coef_retr_z']:+.4f}")
        if var_result is not None:
            print(f"    Variance β_dKL={var_result['coef_dkl_z']:+.4f} (SE {var_result['se_dkl_z']:.4f}, n={var_result['n_firms']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
