"""Compare patent counts and trademark KL scores as return predictors
on the SAME panel. Report R^2 for each predictor in isolation and in
combination so we can see the marginal R^2 added by each.

Predictors (all z-scored, on the common panel):
  P = log(1 + n_patents)        -- patent activity
  I = mean_dkl                  -- innovativeness (composite)
  S = mean_pros                 -- novelty (surprise at filing time)

Specifications (all share year, year^2 controls):
  (1) controls only (baseline)
  (2) + P
  (3) + I
  (4) + S
  (5) + P + I
  (6) + P + S
  (7) + I + S
  (8) + P + I + S

For each horizon k = 1..5y of excess return (firm cumulative price
relative to S&P 500), winsorised at [p1, p99].

Output:
  paper/results/r2_decomposition.tex
  paper/results/r2_decomposition_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"


def load_panel() -> pl.DataFrame:
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        parts.append(pl.read_parquet(path))
    fil = pl.concat(parts).filter(
        pl.col("owner_name").is_not_null()
        & (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= 1990) & (pl.col("year") <= 2020)
    )
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name", "cik")
    matched = fil.join(cw, on="owner_name", how="inner")
    fy_tm = matched.group_by(["cik", "year"]).agg(
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.col("prospective_kl").mean().alias("mean_pros"),
    ).rename({"year": "base_year"})

    pat = pl.read_parquet(PROC / "patent_firm_year.parquet")
    cw_full = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("norm", "cik")
    pat_with_cik = pat.join(cw_full, on="norm", how="inner").select("cik", "year", "n_patents").rename({"year": "base_year"})
    fy_pat = pat_with_cik.group_by(["cik", "base_year"]).agg(pl.col("n_patents").sum())

    rets = pl.read_parquet(PROC / "stock_returns.parquet")
    panel = (
        fy_tm.join(fy_pat, on=["cik", "base_year"], how="inner")
        .join(rets, on=["cik", "base_year"], how="inner")
    )
    for k in (1, 2, 3, 4, 5):
        panel = panel.with_columns(
            (pl.col(f"ret_{k}y") - pl.col(f"sp500_ret_{k}y")).alias(f"excess_{k}y")
        )
    return panel


def fit_with_r2(d, predictors, outcome):
    y = d[outcome]
    cols = list(predictors) + ["year_c", "year_c2"]
    X = sm.add_constant(d[cols])
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["cik"]})
    return float(m.rsquared), {p: (float(m.params[p]), float(m.bse[p])) for p in predictors}


def main() -> int:
    panel = load_panel()
    print(f"[panel] {panel.height:,} firm-years on common (cik, year) basis", file=sys.stderr)

    SPECS = {
        "Controls only": [],
        "Patents (P)": ["pat_z"],
        "Innovator dKL (I)": ["dkl_z"],
        "Novelty pros (S)": ["pros_z"],
        "Patents + Innovator (P+I)": ["pat_z", "dkl_z"],
        "Patents + Novelty (P+S)": ["pat_z", "pros_z"],
        "Innovator + Novelty (I+S)": ["dkl_z", "pros_z"],
        "All three (P+I+S)": ["pat_z", "dkl_z", "pros_z"],
    }

    horizons = (1, 2, 3, 4, 5)
    results = {}
    for k in horizons:
        outcome = f"excess_{k}y"
        d = (
            panel.select(
                pl.col("cik"), pl.col("base_year"),
                pl.col("n_patents"), pl.col("mean_dkl"), pl.col("mean_pros"),
                pl.col(outcome),
            )
            .drop_nulls()
            .to_pandas()
        )
        if len(d) < 200:
            continue
        # Winsorize outcome
        lo, hi = d[outcome].quantile([0.01, 0.99])
        d[outcome] = d[outcome].clip(lo, hi)
        d["pat_z"] = (np.log1p(d["n_patents"]) - np.log1p(d["n_patents"]).mean()) / np.log1p(d["n_patents"]).std()
        d["dkl_z"] = (d["mean_dkl"] - d["mean_dkl"].mean()) / d["mean_dkl"].std()
        d["pros_z"] = (d["mean_pros"] - d["mean_pros"].mean()) / d["mean_pros"].std()
        d["year_c"] = d["base_year"] - d["base_year"].mean()
        d["year_c2"] = d["year_c"] ** 2
        results[k] = {"n": int(len(d)), "specs": {}}
        for label, preds in SPECS.items():
            r2, coefs = fit_with_r2(d, preds, outcome)
            results[k]["specs"][label] = {"r2": r2, "coefs": coefs}

    # LaTeX table: rows = specifications, columns = horizons (R^2 only)
    body = ""
    for label in SPECS:
        body += f"{label} & "
        body += " & ".join(f"{results[k]['specs'][label]['r2']:.4f}" for k in horizons)
        body += " \\\\\n"
    n_row = "N (firm-years) & " + " & ".join(f"{results[k]['n']:,}" for k in horizons) + r" \\" + "\n"
    tex = (
        "\\begin{tabular}{l" + "r" * len(horizons) + "}\n\\toprule\n"
        "Specification (predictors, all $z$-scored) & "
        + " & ".join(f"{k}y $R^2$" for k in horizons)
        + " \\\\\n\\midrule\n"
        + body
        + "\\midrule\n" + n_row
        + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "r2_decomposition.tex").write_text(tex)

    # Console table
    print(f"\n{'spec':<32}" + "".join(f"{k}y R^2 (Δ from baseline)" .ljust(24) for k in horizons))
    base_r2 = {k: results[k]["specs"]["Controls only"]["r2"] for k in horizons}
    for label in SPECS:
        line = f"{label:<32}"
        for k in horizons:
            r2 = results[k]["specs"][label]["r2"]
            d = (r2 - base_r2[k]) * 100
            line += f"{r2:.4f} ({d:+.2f}pp)".ljust(24)
        print(line)
    print()
    print("(Δ pp = R^2 - controls-only R^2, in percentage points)")

    (RESULTS / "r2_decomposition_metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] wrote {RESULTS}/r2_decomposition.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
