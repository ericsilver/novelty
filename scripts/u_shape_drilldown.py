"""Two diagnostics on the U-shape:

  1. Rank all 44 industries by U-shape strength (the quadratic
     coefficient on z(dKL)^2 from a logistic regression on 5y
     survival). Industries without a U either show a flat or an
     inverted-U; for those we report the LINEAR coefficient
     instead.

  2. The U-shape might be driven by the most-extreme filings on
     each KL axis. We test by re-fitting the pooled quadratic
     after dropping:
        (a) the bottom 5th percentile of kl_vs_past (= most
            similar to past, "templated copies of last decade"),
        (b) the bottom 5th percentile of kl_vs_future
            (= most similar to future, "happened to land on
            what later became canonical"),
        (c) both,
        (d) for completeness: dropping dKL extremes
            (the tails of the U itself), to confirm it's the
            shape rather than a few outliers.

Outputs:
  paper/results/u_shape_industries_ranked.tex
  paper/results/u_shape_drilldown.tex
  paper/results/u_shape_drilldown_metrics.json
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
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1990) & (pl.col("year") <= 2020)
)


def load_all() -> pl.DataFrame:
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        parts.append(pl.read_parquet(path).filter(CLEAN).with_columns(pl.lit(cls).alias("nice_class")))
    return pl.concat(parts)


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
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    universe = load_all().filter(pl.col("year") <= 2020)
    print(f"[load] {universe.height:,} clean filings", flush=True)

    # ---- (1) Per-industry U-shape ranking ----
    rows: list[dict] = []
    for cls in sorted(set(universe["nice_class"].to_list())):
        sub = universe.filter(pl.col("nice_class") == cls).select("dkl", "survived_5y", "year").drop_nulls().to_pandas()
        if len(sub) < 5000:
            continue
        sub["survived_5y"] = sub["survived_5y"].astype(int)
        res = fit_quad(sub, "survived_5y")
        if res is None:
            continue
        t_quad = res["beta_dkl_sq"] / res["se_dkl_sq"] if res["se_dkl_sq"] > 0 else float("nan")
        rows.append({
            "cls": cls,
            "industry": industry_name(cls),
            "n": res["n"],
            "beta_dkl": res["beta_dkl"],
            "beta_dkl_sq": res["beta_dkl_sq"],
            "t_quad": t_quad,
        })

    rows.sort(key=lambda r: -r["t_quad"])

    # Tabulate top 15 (strongest U) and bottom 5 (weakest / inverted)
    L = r">{\raggedright\arraybackslash}"
    body = ""
    body += "\\midrule\n\\multicolumn{5}{l}{\\textit{Industries with the strongest U-shape (top 15 by $t$ on $\\beta_{dKL^2}$)}} \\\\\n\\midrule\n"
    for r in rows[:15]:
        ind = r["industry"].replace("&", r"\&")
        body += (
            f"{ind} ({r['cls']}) & {r['n']:,} & "
            f"{r['beta_dkl']:+.4f} & {r['beta_dkl_sq']:+.4f} & {r['t_quad']:+.1f} \\\\\n"
        )
    body += "\\midrule\n\\multicolumn{5}{l}{\\textit{Industries with the weakest or inverted U (bottom 5)}} \\\\\n\\midrule\n"
    for r in rows[-5:]:
        ind = r["industry"].replace("&", r"\&")
        body += (
            f"{ind} ({r['cls']}) & {r['n']:,} & "
            f"{r['beta_dkl']:+.4f} & {r['beta_dkl_sq']:+.4f} & {r['t_quad']:+.1f} \\\\\n"
        )
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Industry (NICE) & N & $\\beta_{dKL}^z$ (linear) & $\\beta_{dKL^2}^z$ (quadratic) & $t$ on quadratic \\\\\n"
        + body + "\\bottomrule\n\\end{tabular}%\n}\n"
    )
    (RESULTS / "u_shape_industries_ranked.tex").write_text(tex)

    # ---- Stats on weak-U industries: do they have stronger LINEAR coefficient? ----
    weak = [r for r in rows if r["t_quad"] < 1.96]
    strong = [r for r in rows if r["t_quad"] >= 1.96]
    if weak and strong:
        weak_lin = np.mean([abs(r["beta_dkl"]) for r in weak])
        strong_lin = np.mean([abs(r["beta_dkl"]) for r in strong])
        print(f"\n[U-strength vs linear strength]")
        print(f"  industries with U (t_quad >= 1.96):  n={len(strong)}  mean |beta_linear| = {strong_lin:.4f}")
        print(f"  industries without U (t_quad <  1.96): n={len(weak)}  mean |beta_linear| = {weak_lin:.4f}")

    # ---- (2) Drop-extremes test on pooled data ----
    print("\n[drop-extremes test]", flush=True)
    base = universe.select("dkl", "kl_vs_past", "kl_vs_future", "survived_5y", "year").drop_nulls().to_pandas()
    base["survived_5y"] = base["survived_5y"].astype(int)
    p5_pros = base["kl_vs_past"].quantile(0.05)
    p5_retr = base["kl_vs_future"].quantile(0.05)
    print(f"  baseline KL distributions:  prospective p5 = {p5_pros:.3f},  retrospective p5 = {p5_retr:.3f}")

    drop_specs = [
        ("Full sample (none dropped)", base),
        (f"Drop bottom 5\\% of prospective KL, vs. past (< {p5_pros:.2f})", base[base["kl_vs_past"] >= p5_pros]),
        (f"Drop bottom 5\\% of retrospective KL, vs. future (< {p5_retr:.2f})", base[base["kl_vs_future"] >= p5_retr]),
        (f"Drop bottom 5\\% of both axes", base[(base["kl_vs_past"] >= p5_pros) & (base["kl_vs_future"] >= p5_retr)]),
        ("Drop bottom + top 10\\% of $\\Delta KL$ (test extreme tails)",
         base[(base["dkl"] >= base["dkl"].quantile(0.10)) & (base["dkl"] <= base["dkl"].quantile(0.90))]),
    ]

    body = ""
    drop_results = []
    for label, d in drop_specs:
        res = fit_quad(d, "survived_5y")
        if res is None:
            continue
        t_q = res["beta_dkl_sq"] / res["se_dkl_sq"] if res["se_dkl_sq"] > 0 else float("nan")
        body += (
            f"{label} & {res['n']:,} & "
            f"{res['beta_dkl']:+.4f} & {res['beta_dkl_sq']:+.4f} & {t_q:+.1f} \\\\\n"
        )
        drop_results.append({"label": label, **res, "t_quad": t_q})
        print(f"  {label}: n={res['n']:,}  beta_lin={res['beta_dkl']:+.4f}  beta_quad={res['beta_dkl_sq']:+.4f}  t_quad={t_q:+.1f}")

    tex = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Sample & N & $\\beta_{dKL}^z$ & $\\beta_{dKL^2}^z$ & $t$ on quad. \\\\\n\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "u_shape_drilldown.tex").write_text(tex)

    (RESULTS / "u_shape_drilldown_metrics.json").write_text(json.dumps({
        "industry_ranking": rows,
        "drop_specs": drop_results,
        "weak_lin_mean": weak_lin if weak and strong else None,
        "strong_lin_mean": strong_lin if weak and strong else None,
    }, indent=2, default=str))
    print(f"\n[done] wrote u_shape_industries_ranked.tex, u_shape_drilldown.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
