"""Test whether the dKL-vs-survival relationship is U-shaped (or
"M-shaped" if both extremes outperform the middle):

  (1) Per-industry decile-rate: identify which industries display the
      pattern.
  (2) Quadratic regression: Pr(survive) ~ alpha + beta1*dKL + beta2*dKL^2,
      where a positive beta2 confirms concave-up shape.
  (3) Split at zero, fit log(rate) = a +/- b*|dKL| on each side
      (exponential model).

Outputs:
  paper/results/u_shape_table.tex
  paper/results/u_shape_industries.tex
  paper/results/u_shape_curves.png
  paper/results/u_shape_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"
PROC = REPO_ROOT / "data" / "processed"


CLEAN = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1995) & (pl.col("year") <= 2020)
)


def fit_quad(d: pl.DataFrame, outcome: str):
    pdf = d.select(pl.col("dkl"), pl.col(outcome).cast(pl.Int8), pl.col("year")).drop_nulls().to_pandas()
    if len(pdf) < 1000:
        return None
    pdf["dkl_z"] = (pdf["dkl"] - pdf["dkl"].mean()) / pdf["dkl"].std()
    pdf["dkl_z_sq"] = pdf["dkl_z"] ** 2
    pdf["year_c"] = pdf["year"] - pdf["year"].mean()
    pdf["year_c2"] = pdf["year_c"] ** 2
    X = sm.add_constant(pdf[["dkl_z", "dkl_z_sq", "year_c", "year_c2"]])
    m = sm.GLM(pdf[outcome], X, family=sm.families.Binomial()).fit(disp=False, maxiter=200)
    return {
        "n": int(len(pdf)),
        "beta_dkl": float(m.params["dkl_z"]),
        "se_dkl": float(m.bse["dkl_z"]),
        "beta_dkl_sq": float(m.params["dkl_z_sq"]),
        "se_dkl_sq": float(m.bse["dkl_z_sq"]),
    }


def fit_split_exp(d: pl.DataFrame, outcome: str):
    """Fit log(rate) = a + b * |dKL| on each side of dKL=0.
    Aggregates to bins of |dKL| within each side.
    """
    pdf = d.select(pl.col("dkl"), pl.col(outcome).cast(pl.Int8)).drop_nulls().to_pandas()
    out = {}
    for side, mask in (("neg", pdf["dkl"] < 0), ("pos", pdf["dkl"] > 0)):
        sub = pdf[mask].copy()
        if len(sub) < 1000:
            out[side] = None
            continue
        sub["abs_dkl"] = sub["dkl"].abs()
        edges = np.quantile(sub["abs_dkl"], np.linspace(0, 1, 11))
        sub["bin"] = np.digitize(sub["abs_dkl"], edges[1:-1])
        agg = sub.groupby("bin").agg(x=("abs_dkl", "mean"), y=(outcome, "mean"), n=(outcome, "size")).reset_index(drop=True)
        agg = agg[agg["n"] >= 100]
        if len(agg) < 4:
            out[side] = None
            continue
        # log y = a + b * x
        Y = np.log(agg["y"].values + 1e-9)
        X = sm.add_constant(agg["x"].values)
        m = sm.OLS(Y, X).fit()
        out[side] = {
            "n": int(sub.shape[0]),
            "n_bins": int(len(agg)),
            "intercept_a": float(m.params[0]),
            "slope_b": float(m.params[1]),
            "se_b": float(m.bse[1]),
            "rsq": float(m.rsquared),
            "bins_xy": agg.to_dict(orient="records"),
        }
    return out


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    classes = sorted(p.stem.replace("outcomes_class", "")
                     for p in PROC.glob("outcomes_class*.parquet")
                     if p.stem.replace("outcomes_class", "").isdigit())

    # Aggregate quadratic fits
    quad_rows = []
    decile_rows = []
    for cls in classes:
        df = pl.read_parquet(PROC / f"outcomes_class{cls}.parquet").filter(CLEAN)
        if df.height < 5000:
            continue
        for outcome, cap in (("reached_registration", 2025), ("survived_5y", 2020), ("survived_10y", 2015)):
            d = df.filter(pl.col("year") <= cap)
            res = fit_quad(d, outcome)
            if res is None:
                continue
            quad_rows.append({
                "cls": cls, "industry": industry_name(cls), "outcome": outcome,
                **res,
            })
        # Decile rates of survived_5y per dKL decile (for plot)
        d5 = df.filter(pl.col("year") <= 2020).select("dkl", "survived_5y").drop_nulls().to_pandas()
        edges = np.quantile(d5["dkl"], np.linspace(0, 1, 11))
        d5["dec"] = np.digitize(d5["dkl"], edges[1:-1])
        agg = d5.groupby("dec").agg(rate=("survived_5y", "mean"), x=("dkl", "mean"), n=("survived_5y", "size")).reset_index()
        for r in agg.itertuples():
            decile_rows.append({
                "cls": cls, "industry": industry_name(cls),
                "decile": int(r.dec), "x": float(r.x), "rate": float(r.rate), "n": int(r.n),
            })

    # ----- Per-outcome aggregate quadratic across all industries -----
    print("\n[quadratic fits, aggregate]")
    headline = {}
    for outcome in ("reached_registration", "survived_5y", "survived_10y"):
        all_pdfs = []
        for cls in classes:
            df = pl.read_parquet(PROC / f"outcomes_class{cls}.parquet").filter(CLEAN)
            if df.height < 5000:
                continue
            cap = {"reached_registration": 2025, "survived_5y": 2020, "survived_10y": 2015}[outcome]
            d = df.filter(pl.col("year") <= cap)
            sub = d.select(pl.col("dkl"), pl.col(outcome).cast(pl.Int8), pl.col("year")).drop_nulls().to_pandas()
            sub["cls"] = cls
            all_pdfs.append(sub)
        pdf = __import__("pandas").concat(all_pdfs, ignore_index=True)
        pdf["dkl_z"] = (pdf["dkl"] - pdf["dkl"].mean()) / pdf["dkl"].std()
        pdf["dkl_z_sq"] = pdf["dkl_z"] ** 2
        pdf["year_c"] = pdf["year"] - pdf["year"].mean()
        pdf["year_c2"] = pdf["year_c"] ** 2
        X = sm.add_constant(pdf[["dkl_z", "dkl_z_sq", "year_c", "year_c2"]])
        m = sm.GLM(pdf[outcome], X, family=sm.families.Binomial()).fit(disp=False, maxiter=200)
        headline[outcome] = {
            "n": int(len(pdf)),
            "beta_dkl": float(m.params["dkl_z"]),
            "se_dkl": float(m.bse["dkl_z"]),
            "beta_dkl_sq": float(m.params["dkl_z_sq"]),
            "se_dkl_sq": float(m.bse["dkl_z_sq"]),
        }
        print(f"  {outcome:<25}  beta_dKL = {headline[outcome]['beta_dkl']:+.4f} (SE {headline[outcome]['se_dkl']:.4f})  "
              f"beta_dKL^2 = {headline[outcome]['beta_dkl_sq']:+.4f} (SE {headline[outcome]['se_dkl_sq']:.4f})  n={headline[outcome]['n']:,}")

        # Split-exponential
        df_all = pl.from_pandas(pdf.drop(columns=["cls"]).rename(columns={outcome: outcome}))
        split_res = fit_split_exp(df_all, outcome)
        headline[outcome]["split_exp"] = split_res

    # Per-industry quadratic counts
    pos_count = sum(1 for r in quad_rows if r["outcome"] == "survived_5y" and r["beta_dkl_sq"] > 0)
    sig_pos = sum(1 for r in quad_rows if r["outcome"] == "survived_5y" and r["beta_dkl_sq"] > 0 and abs(r["beta_dkl_sq"] / r["se_dkl_sq"]) > 1.96)
    n_indus = sum(1 for r in quad_rows if r["outcome"] == "survived_5y")
    print(f"\n[per-industry quadratic on survived_5y]: {n_indus} industries; {pos_count} with positive beta_dKL^2 ({sig_pos} significant)")

    # ----- Plot: decile-rate by industry, normalized -----
    fig, ax = plt.subplots(figsize=(9, 5.4))
    # Aggregate decile rates across ALL industries
    decile_pdf = __import__("pandas").DataFrame(decile_rows)
    agg_dec = decile_pdf.groupby("decile").agg(x=("x", "mean"), y=("rate", "mean"), n=("n", "sum")).reset_index()
    ax.plot(agg_dec["decile"], agg_dec["y"] * 100, "o-", color="#d62728", linewidth=2.0, markersize=6,
            label="Cross-industry mean (all 45 NICE classes)", zorder=3)
    # Each industry as a thin grey line
    for cls in decile_pdf["cls"].unique():
        sub = decile_pdf[decile_pdf["cls"] == cls].sort_values("decile")
        ax.plot(sub["decile"], sub["rate"] * 100, "-", color="#888", alpha=0.18, linewidth=0.6, zorder=1)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"D{i+1}" for i in range(10)])
    ax.set_xlabel(r"$\Delta KL$ decile (D1 = least innovative, D10 = most)")
    ax.set_ylabel(r"5-year registration-renewal rate (\%)")
    ax.set_title(r"Per-industry decile rates of 5y survival vs.\ $\Delta KL$" + "\n(grey lines = individual industries; red = cross-industry mean)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "u_shape_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] saved u_shape_curves.png")

    # ----- LaTeX tables -----
    body = ""
    label = {"reached_registration": "Completed registration",
             "survived_5y": "Renewed 5y registration",
             "survived_10y": "Renewed 10y registration"}
    for outcome in ("reached_registration", "survived_5y", "survived_10y"):
        h = headline[outcome]
        body += (
            f"{label[outcome]} & {h['n']:,} & "
            f"{h['beta_dkl']:+.4f} ({h['se_dkl']:.4f}) & "
            f"{h['beta_dkl_sq']:+.4f} ({h['se_dkl_sq']:.4f}) \\\\\n"
        )
    tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Outcome & N & Linear $\\beta_{\\Delta KL}^z$ & Quadratic $\\beta_{(\\Delta KL)^2}^z$ \\\\\n"
        "\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "u_shape_table.tex").write_text(tex)

    # Per-industry table: top 10 industries by quadratic coefficient (positive = U-shape)
    s5 = [r for r in quad_rows if r["outcome"] == "survived_5y"]
    s5.sort(key=lambda r: -r["beta_dkl_sq"])
    body = ""
    for r in s5[:15]:
        ind = r["industry"].replace("&", r"\&")
        sig = "*" if abs(r["beta_dkl_sq"] / r["se_dkl_sq"]) > 1.96 else ""
        body += f"{ind} ({r['cls']}) & {r['n']:,} & {r['beta_dkl']:+.4f} & {r['beta_dkl_sq']:+.4f}{sig} \\\\\n"
    body += "\\midrule\n"
    body += "\\multicolumn{4}{l}{\\textit{...bottom 5 (most negative quadratic = inverted-U):}} \\\\\n\\midrule\n"
    for r in s5[-5:]:
        ind = r["industry"].replace("&", r"\&")
        sig = "*" if abs(r["beta_dkl_sq"] / r["se_dkl_sq"]) > 1.96 else ""
        body += f"{ind} ({r['cls']}) & {r['n']:,} & {r['beta_dkl']:+.4f} & {r['beta_dkl_sq']:+.4f}{sig} \\\\\n"
    tex_ind = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Industry (NICE) & N & Linear $\\beta_{\\Delta KL}^z$ & Quadratic $\\beta_{(\\Delta KL)^2}^z$ \\\\\n"
        "\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}%\n}\n"
    )
    (RESULTS / "u_shape_industries.tex").write_text(tex_ind)

    (RESULTS / "u_shape_metrics.json").write_text(json.dumps({
        "headline": headline,
        "per_industry_5y_count_positive_quadratic": pos_count,
        "per_industry_5y_count_total": n_indus,
        "per_industry_5y_sig_positive": sig_pos,
    }, indent=2, default=str))
    print(f"\n[done] wrote u_shape_table.tex / u_shape_industries.tex / u_shape_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
