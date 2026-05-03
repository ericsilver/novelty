"""Three follow-ups on the U-shape:

  (1) Imitated-but-failed waves: identify the blockchain / crypto /
      NFT / web3 cohort by goods/services keyword and report its
      survival outcomes. The hypothesis is that this is a high-dKL
      cohort that was widely imitated yet failed at a high rate.

  (2) Single-predictor regression on |dKL| against the same survival
      outcomes. This collapses the U-shape to one number: how much
      does sheer distance from the industry-median predict survival,
      ignoring direction?

  (3) What else might explain the U-shape: report a likelihood-of-
      confusion proxy from the office-action codes, by dKL decile,
      to test whether middle-dKL filings collide more with prior
      marks than tail-dKL filings.

Outputs:
  paper/results/blockchain_wave.tex
  paper/results/abs_dkl_table.tex
  paper/results/loc_by_decile.tex
  paper/results/u_shape_followup_metrics.json
"""
from __future__ import annotations

import json
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
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1995) & (pl.col("year") <= 2020)
)


def load_universe_with_text() -> pl.DataFrame:
    """We need the actual goods_services text for the wave search."""
    parts = []
    for path in sorted(PROC.glob("tm_class*.parquet")):
        cls = path.stem.replace("tm_class", "")
        if not cls.isdigit():
            continue
        df = pl.read_parquet(path).select("serial_number", "owner_name", "goods_services").with_columns(pl.lit(cls).alias("nice_class"))
        parts.append(df)
    return pl.concat(parts)


def load_outcomes() -> pl.DataFrame:
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        parts.append(pl.read_parquet(path).filter(CLEAN).with_columns(pl.lit(cls).alias("nice_class")))
    return pl.concat(parts)


def main() -> int:
    print("[load] outcomes…", flush=True)
    out = load_outcomes()
    print(f"       {out.height:,} clean filings, all 45 NICE classes")

    # ---- (1) Blockchain / crypto wave ----
    print("\n[1] Blockchain wave: load full G/S text for matching…", flush=True)
    text = load_universe_with_text()
    crypto_re = r"(?i)\b(blockchain|cryptocurrenc|crypto[- ]?currenc|crypto[- ]?asset|crypto[- ]?token|nft|non[- ]?fungible|smart[- ]?contract|web3|defi|ethereum|bitcoin|ico\b|stablecoin)\b"
    matched = text.filter(pl.col("goods_services").str.contains(crypto_re))
    print(f"       {matched.height:,} filings whose G/S text mentions a blockchain/crypto term")

    # Join with outcomes (clean only)
    crypto_out = out.join(matched.select("serial_number", "nice_class"), on=["serial_number", "nice_class"], how="inner")
    print(f"       {crypto_out.height:,} of these are clean (1995-2020 cohort with both KL windows)")

    # Per-year filing counts and survival rates
    by_year = crypto_out.group_by("year").agg(
        pl.len().alias("n"),
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.col("reached_registration").mean().alias("reach"),
        pl.col("survived_5y").mean().alias("surv5"),
    ).sort("year")
    print("\nBlockchain-wave filings by year (median dKL, registration & 5y survival):")
    for r in by_year.iter_rows(named=True):
        if r["n"] < 5: continue
        print(f"  {r['year']}: n={r['n']:>5}  dkl_mean={r['mean_dkl']:+.3f}  reach={r['reach']*100:.0f}%  surv5={r['surv5']*100:.0f}%")

    # Compare cohort survival to the matched non-crypto cohort with similar dKL
    # Aggregate: blockchain-wave overall vs all-clean overall
    blk_n = crypto_out.height
    blk_reach = float(crypto_out["reached_registration"].mean())
    blk_surv5 = float(crypto_out.filter(pl.col("year") <= 2020)["survived_5y"].mean())
    blk_dkl = float(crypto_out["dkl"].mean())
    all_reach = float(out["reached_registration"].mean())
    all_surv5 = float(out.filter(pl.col("year") <= 2020)["survived_5y"].mean())
    all_dkl = float(out["dkl"].mean())
    print(f"\n  blockchain wave: n={blk_n:,}  mean dKL={blk_dkl:+.3f}  reach={blk_reach*100:.1f}%  surv5={blk_surv5*100:.1f}%")
    print(f"  all clean:        n={out.height:,}  mean dKL={all_dkl:+.3f}  reach={all_reach*100:.1f}%  surv5={all_surv5*100:.1f}%")

    # Top-decile-of-dKL crypto filings vs same-dkl-decile non-crypto filings
    out_pd = out.select("serial_number", "dkl", "reached_registration", "survived_5y", "year").to_pandas()
    out_pd["is_crypto"] = out_pd["serial_number"].isin(crypto_out["serial_number"].to_list())
    edges = out_pd["dkl"].quantile([0.1, 0.5, 0.9, 0.95]).values
    top10 = out_pd[out_pd["dkl"] >= edges[2]]  # top decile by dkl
    print(f"\n  Among the top-decile-by-dKL filings ({len(top10):,}):")
    for grp_name, grp in top10.groupby("is_crypto"):
        gn = "crypto wave" if grp_name else "non-crypto"
        print(f"    {gn:<13}  n={len(grp):>7,}  reach={grp['reached_registration'].mean()*100:.1f}%  surv5={grp[grp['year']<=2020]['survived_5y'].mean()*100:.1f}%")

    # Persist a small TeX summary
    tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Sample & N & Reach reg.\\ rate & 5y survival rate \\\\\n\\midrule\n"
        f"All clean filings & {out.height:,} & {all_reach*100:.1f}\\% & {all_surv5*100:.1f}\\% \\\\\n"
        f"Blockchain/crypto wave (any G/S match) & {blk_n:,} & {blk_reach*100:.1f}\\% & {blk_surv5*100:.1f}\\% \\\\\n"
    )
    # Top-decile dKL split
    crypto_top10 = top10[top10["is_crypto"]]
    non_top10 = top10[~top10["is_crypto"]]
    tex += (
        "\\midrule\n"
        "\\multicolumn{4}{l}{\\textit{Restricted to filings in the top decile of $\\Delta KL$:}} \\\\\n"
        "\\midrule\n"
        f"Top-dKL non-crypto & {len(non_top10):,} & {non_top10['reached_registration'].mean()*100:.1f}\\% & {non_top10[non_top10['year']<=2020]['survived_5y'].mean()*100:.1f}\\% \\\\\n"
        f"Top-dKL crypto wave & {len(crypto_top10):,} & {crypto_top10['reached_registration'].mean()*100:.1f}\\% & {crypto_top10[crypto_top10['year']<=2020]['survived_5y'].mean()*100:.1f}\\% \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "blockchain_wave.tex").write_text(tex)

    # ---- (2) |dKL| single-predictor regression ----
    print("\n[2] |dKL| as a single predictor (collapses U to magnitude)", flush=True)
    rows = []
    pdf = out.select("dkl", "reached_registration", "survived_5y", "survived_10y", "n_terms", "year").drop_nulls().to_pandas()
    # Sample down to 500k for speed (GLM on 8.6M is slow; SEs on 500k are still tiny on a real effect)
    if len(pdf) > 500_000:
        pdf = pdf.sample(n=500_000, random_state=7).reset_index(drop=True)
    pdf["abs_dkl"] = pdf["dkl"].abs()
    pdf["abs_dkl_z"] = (pdf["abs_dkl"] - pdf["abs_dkl"].mean()) / pdf["abs_dkl"].std()
    pdf["log_n_terms"] = np.log(pdf["n_terms"])
    pdf["year_c"] = pdf["year"] - pdf["year"].mean()
    pdf["year_c2"] = pdf["year_c"] ** 2
    for outcome, cap in (("reached_registration", 2025), ("survived_5y", 2020), ("survived_10y", 2015)):
        d = pdf[pdf["year"] <= cap].copy()
        if outcome == "survived_10y":
            d = d.dropna(subset=["survived_10y"])
        d[outcome] = d[outcome].astype(int)
        X = sm.add_constant(d[["abs_dkl_z", "log_n_terms", "year_c", "year_c2"]])
        m = sm.GLM(d[outcome], X, family=sm.families.Binomial()).fit(disp=False, maxiter=200)
        # McFadden pseudo R^2
        m_null = sm.GLM(d[outcome], np.ones((len(d), 1)), family=sm.families.Binomial()).fit(disp=False)
        r2 = 1 - m.llf / m_null.llf
        # Linear OLS for "true" R^2 reference
        Xlin = sm.add_constant(d[["abs_dkl_z", "log_n_terms", "year_c", "year_c2"]])
        m_ols = sm.OLS(d[outcome], Xlin).fit()
        beta = float(m.params["abs_dkl_z"])
        se = float(m.bse["abs_dkl_z"])
        z = beta / se
        from scipy import stats as st
        p = float(2 * (1 - st.norm.cdf(abs(z))))
        rows.append({
            "outcome": outcome, "n": int(len(d)),
            "beta": beta, "se": se, "z": z, "p": p,
            "mcfadden_r2": float(r2),
            "ols_r2_full": float(m_ols.rsquared),
            "ols_r2_abs_dkl_only_marginal": None,  # filled below
        })
        # Marginal R^2 from |dKL| alone over controls only
        m_ctrl = sm.OLS(d[outcome], sm.add_constant(d[["log_n_terms", "year_c", "year_c2"]])).fit()
        rows[-1]["ols_r2_abs_dkl_only_marginal"] = float(m_ols.rsquared - m_ctrl.rsquared)
        print(f"  {outcome:<25}  n={len(d):>9,}  beta={beta:+.4f} (SE {se:.4f})  z={z:+.1f}  p={p:.2e}  pseudo-R2={r2:.4f}  Δ-R2(linear)={(m_ols.rsquared - m_ctrl.rsquared)*100:.3f} pp")

    label = {"reached_registration": "Completed registration", "survived_5y": "Renewed 5y", "survived_10y": "Renewed 10y"}
    body = ""
    for r in rows:
        body += (
            f"{label[r['outcome']]} & {r['n']:,} & "
            f"{r['beta']:+.4f} ({r['se']:.4f}) & "
            f"{r['z']:+.1f} & "
            f"{r['p']:.1e} & "
            f"{r['mcfadden_r2']:.4f} & "
            f"{r['ols_r2_abs_dkl_only_marginal']*100:.3f}\\,pp \\\\\n"
        )
    abs_tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Outcome & N & $\\beta_{|\\Delta KL|}^z$ (logit) & $z$ & $p$ & "
        "McFadden pseudo-$R^2$ & $\\Delta R^2$ (linear, vs.\\ controls) \\\\\n\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}%\n}\n"
    )
    (RESULTS / "abs_dkl_table.tex").write_text(abs_tex)

    # ---- (3) Alternative explanation: likelihood-of-confusion (status code 700) by dKL decile ----
    print("\n[3] Likelihood-of-confusion proxy by dKL decile…", flush=True)
    # Status 700 = "abandoned, no response to office action" -- often a likelihood-of-confusion outcome
    # We use the records parquet for status codes
    rec_parts = []
    for path in sorted(PROC.glob("tm_class*.parquet")):
        cls = path.stem.replace("tm_class", "")
        if not cls.isdigit(): continue
        rec_parts.append(pl.read_parquet(path).select("serial_number", "status_code"))
    rec = pl.concat(rec_parts)
    pdf2 = out.select("serial_number", "dkl", "year").drop_nulls().to_pandas()
    pdf2 = pdf2.merge(rec.to_pandas(), on="serial_number", how="left")
    pdf2["loc_proxy"] = pdf2["status_code"].astype(str).str.startswith("700")
    edges = pdf2["dkl"].quantile(np.linspace(0, 1, 11)).values
    pdf2["dec"] = np.digitize(pdf2["dkl"], edges[1:-1])
    grp = pdf2.groupby("dec").agg(
        rate=("loc_proxy", "mean"),
        x=("dkl", "mean"),
        n=("loc_proxy", "size"),
    ).reset_index()
    print("\nShare of filings ending in status 700 ('abandoned, no response to office action') by dKL decile:")
    for r in grp.itertuples():
        print(f"  decile {r.dec} (mean dKL {r.x:+.2f}): {r.rate*100:.1f}%  n={r.n:,}")

    body = ""
    for r in grp.itertuples():
        body += f"D{int(r.dec)+1} & {r.x:+.3f} & {r.n:,} & {r.rate*100:.1f}\\% \\\\\n"
    loc_tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "$\\Delta KL$ decile & Mean $\\Delta KL$ & N & Status-700 share \\\\\n\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "loc_by_decile.tex").write_text(loc_tex)

    metrics = {
        "blockchain": {"n": int(blk_n), "mean_dkl": blk_dkl, "reach": blk_reach, "surv5": blk_surv5,
                       "all_reach": all_reach, "all_surv5": all_surv5, "all_n": int(out.height)},
        "abs_dkl": rows,
        "loc_by_decile": grp.to_dict(orient="records"),
    }
    (RESULTS / "u_shape_followup_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print("\n[done] wrote blockchain_wave.tex, abs_dkl_table.tex, loc_by_decile.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
