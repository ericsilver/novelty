"""Full ranked list of all 44 NICE industries by strength of the
survival U-shape on ΔKL. For each industry we fit a logistic regression
with z(ΔKL) and z(ΔKL)^2 as predictors of 5y survival (with log(n_terms)
and year quadratic controls); the t-statistic on the quadratic
coefficient is the rank key (positive = U-shape, negative = inverted-U).

Outputs:
  paper/results/u_shape_full_ranking.csv
  paper/results/u_shape_full_ranking.tex
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

from novelty.industries import name as industry_name

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def fit_one(cls: str) -> dict | None:
    sp = PROC / f"surprise_class{cls}.parquet"
    op = PROC / f"outcomes_class{cls}.parquet"
    if not (sp.exists() and op.exists()):
        return None
    s = pl.read_parquet(
        sp,
        columns=["serial_number", "year", "prospective_kl", "retrospective_kl",
                 "n_ref_prospective", "n_ref_retrospective", "n_terms"],
    ).filter(
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
        & pl.col("year").is_between(1990, 2021)
    ).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
    )
    o = pl.read_parquet(op, columns=["serial_number", "survived_5y"])
    j = s.join(o, on="serial_number", how="inner")
    n = j.height
    if n < 5_000:
        return None

    pdf = j.to_pandas()
    pdf["log_n_terms"] = np.log(pdf["n_terms"].clip(lower=1))
    pdf["year_c"] = pdf["year"] - pdf["year"].mean()
    pdf["year_c2"] = pdf["year_c"] ** 2
    dkl = pdf["dkl"].to_numpy()
    z = (dkl - dkl.mean()) / dkl.std() if dkl.std() > 0 else np.zeros_like(dkl)
    z2 = z * z
    y = pdf["survived_5y"].astype(int).to_numpy()
    X = sm.add_constant(np.column_stack([z, z2, pdf["log_n_terms"].to_numpy(),
                                          pdf["year_c"].to_numpy(),
                                          pdf["year_c2"].to_numpy()]))
    try:
        m = sm.Logit(y, X).fit(disp=False, maxiter=80)
    except Exception as e:
        print(f"  class {cls}: fit failed: {e}")
        return None
    beta_lin = float(m.params[1])
    beta_quad = float(m.params[2])
    se_quad = float(m.bse[2])
    t_quad = beta_quad / se_quad if se_quad > 0 else 0.0
    del s, o, j, pdf
    gc.collect()
    return {
        "cls": cls, "name": industry_name(cls), "n": n,
        "beta_lin": beta_lin, "beta_quad": beta_quad,
        "se_quad": se_quad, "t_quad": t_quad,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in class_list():
        print(f"[fit] class {cls}…", flush=True)
        r = fit_one(cls)
        if r:
            rows.append(r)
            print(f"  n={r['n']:>8,}  beta_lin={r['beta_lin']:+.4f}  "
                  f"beta_quad={r['beta_quad']:+.4f}  t_quad={r['t_quad']:+.1f}",
                  flush=True)
    df = pl.from_dicts(rows).sort("t_quad", descending=True)
    df.write_csv(OUT / "u_shape_full_ranking.csv")

    lines = [r"\begin{tabular}{rlrrrr}",
             r"\toprule",
             r"Rank & Industry (NICE) & $n$ & $\beta_{\Delta KL}^z$ & "
             r"$\beta_{(\Delta KL)^2}^z$ & $t$ on quadratic \\",
             r"\midrule"]
    for i, r in enumerate(df.iter_rows(named=True), start=1):
        name = r["name"].replace("&", r"\&")
        star = "*" if abs(r["t_quad"]) >= 1.96 else ""
        lines.append(f"{i} & {name} ({r['cls']}) & {r['n']:,} & "
                     f"{r['beta_lin']:+.4f} & {r['beta_quad']:+.4f}{star} & "
                     f"{r['t_quad']:+.1f} \\\\")
        if i == sum(1 for x in rows if x["t_quad"] > 1.96):
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{6}{l}{\textit{below: $|t| < 1.96$ or quadratic negative (U is noise or reverses)}} \\")
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "u_shape_full_ranking.tex").write_text("\n".join(lines))
    print(f"\n[done] wrote .csv + .tex with {len(rows)} industries", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
