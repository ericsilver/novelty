"""For each (industry, year) cell, fit a logistic regression
  P(survived_5y) = logit^-1(b0 + b1*z(dKL) + b2*z(dKL)^2 + g*log(n_terms))
and classify the cell by:
  - has +U  (b2 > 0 and z-stat > +1.96)
  - has inverted-U (b2 < 0 and z-stat < -1.96)
  - indeterminate (|z-stat| <= 1.96)
  - advantages +dKL (b1 > 0, ignoring the quadratic)
  - advantages -dKL (b1 < 0)
Also at the industry level: count years per industry meeting each criterion.

Outputs:
  paper/results/u_shape_industry_year_matrix.csv     (44 industries x 32 years)
  paper/results/u_shape_industry_year_summary.csv    (one row per industry)
  paper/results/u_shape_industry_year_table.tex      (the headline table)
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
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def fit_cell(sub_pd) -> dict | None:
    if len(sub_pd) < 500: return None
    dkl = sub_pd["dkl"].to_numpy()
    if dkl.std() == 0: return None
    z = (dkl - dkl.mean()) / dkl.std()
    log_n = np.log(np.clip(sub_pd["n_terms"].to_numpy(), 1, None))
    X = sm.add_constant(np.column_stack([z, z*z, log_n]))
    y = sub_pd["survived_5y"].astype(int).to_numpy()
    if y.sum() < 20 or (y.sum() == len(y)) or y.sum() == 0:
        return None
    try:
        m = sm.Logit(y, X).fit(disp=False, maxiter=50)
    except Exception:
        return None
    b1, b2 = float(m.params[1]), float(m.params[2])
    z1 = b1 / float(m.bse[1]) if m.bse[1] > 0 else 0.0
    z2 = b2 / float(m.bse[2]) if m.bse[2] > 0 else 0.0
    return {"n": len(sub_pd), "beta_lin": b1, "beta_quad": b2,
            "z_lin": z1, "z_quad": z2}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        op = PROC / f"outcomes_class{cls}.parquet"
        if not (sp.exists() and op.exists()): continue
        s = pl.read_parquet(sp, columns=["serial_number","year","prospective_kl","retrospective_kl",
                                          "n_ref_prospective","n_ref_retrospective","n_terms"]).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
            & pl.col("year").is_between(1990, 2021)
        ).with_columns(
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"))
        o = pl.read_parquet(op, columns=["serial_number","survived_5y"])
        j = s.join(o, on="serial_number", how="inner")
        if j.height < 5000:
            continue
        print(f"[fit] class {cls}: {j.height:,} filings", flush=True)
        for yr in range(1990, 2022):
            sub = j.filter(pl.col("year") == yr).to_pandas()
            r = fit_cell(sub)
            if r is None: continue
            rows.append({"cls": cls, "name": industry_name(cls), "year": yr, **r})
        del s, o, j
        gc.collect()

    df = pl.from_dicts(rows)
    df.write_csv(OUT / "u_shape_industry_year_matrix.csv")
    print(f"\n[done] {df.height:,} (industry x year) cells fit", flush=True)

    # Classify each cell
    df = df.with_columns(
        pl.when((pl.col("z_quad") > 1.96)).then(pl.lit("+U"))
          .when((pl.col("z_quad") < -1.96)).then(pl.lit("-U"))
          .otherwise(pl.lit("flat")).alias("u_class"),
        pl.when((pl.col("z_lin") > 1.96)).then(pl.lit("+dKL"))
          .when((pl.col("z_lin") < -1.96)).then(pl.lit("-dKL"))
          .otherwise(pl.lit("none")).alias("lin_adv"),
    )

    # Per-industry summary: how many years of each type?
    summary = (df.group_by(["cls","name"]).agg([
        pl.len().alias("n_years_fit"),
        (pl.col("u_class") == "+U").sum().alias("n_years_pos_U"),
        (pl.col("u_class") == "-U").sum().alias("n_years_inv_U"),
        (pl.col("u_class") == "flat").sum().alias("n_years_flat"),
        (pl.col("lin_adv") == "+dKL").sum().alias("n_years_adv_pos_dKL"),
        (pl.col("lin_adv") == "-dKL").sum().alias("n_years_adv_neg_dKL"),
        (pl.col("lin_adv") == "none").sum().alias("n_years_no_lin_adv"),
        pl.col("beta_quad").mean().alias("mean_beta_quad"),
        pl.col("beta_lin").mean().alias("mean_beta_lin"),
    ]).sort("mean_beta_quad", descending=True))
    summary.write_csv(OUT / "u_shape_industry_year_summary.csv")
    print(summary.to_pandas().to_string(index=False))

    # Aggregate counts: industry-level classification (majority of years)
    summary = summary.with_columns(
        pl.when(pl.col("n_years_pos_U") > pl.col("n_years_inv_U") + 2)
            .then(pl.lit("U"))
          .when(pl.col("n_years_inv_U") > pl.col("n_years_pos_U") + 2)
            .then(pl.lit("invU"))
          .otherwise(pl.lit("indet")).alias("industry_class")
    )
    counts = summary.group_by("industry_class").agg(pl.len().alias("n"))
    print()
    print("Industry-level classification (majority-of-years rule, margin of 2):")
    print(counts.to_pandas().to_string(index=False))

    # Total cell counts
    cell_summary = df.group_by("u_class").agg(pl.len().alias("n_cells"))
    print()
    print("Cell-level (industry x year) counts by U classification:")
    print(cell_summary.to_pandas().to_string(index=False))
    lin_summary = df.group_by("lin_adv").agg(pl.len().alias("n_cells"))
    print()
    print("Cell-level counts by linear direction:")
    print(lin_summary.to_pandas().to_string(index=False))

    # Build the headline table
    n_industries = summary.height
    n_cells_total = df.height
    pos_U_industries = int(summary.filter(pl.col("industry_class") == "U").height)
    inv_U_industries = int(summary.filter(pl.col("industry_class") == "invU").height)
    indet_industries = int(summary.filter(pl.col("industry_class") == "indet").height)
    pos_U_cells  = int(df.filter(pl.col("u_class") == "+U").height)
    inv_U_cells  = int(df.filter(pl.col("u_class") == "-U").height)
    flat_cells   = int(df.filter(pl.col("u_class") == "flat").height)
    adv_pos_cells = int(df.filter(pl.col("lin_adv") == "+dKL").height)
    adv_neg_cells = int(df.filter(pl.col("lin_adv") == "-dKL").height)
    no_adv_cells  = int(df.filter(pl.col("lin_adv") == "none").height)

    tex = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Statistic & count & \% of total \\",
        r"\midrule",
        r"\textbf{Industries (of " + f"{n_industries}" + r" with sufficient data)} & & \\",
        f"  with significant $+U$ on $\\Delta KL$ & {pos_U_industries} & {100*pos_U_industries/n_industries:.0f}\\% \\\\",
        f"  with significant inverted-$U$ on $\\Delta KL$ & {inv_U_industries} & {100*inv_U_industries/n_industries:.0f}\\% \\\\",
        f"  indeterminate & {indet_industries} & {100*indet_industries/n_industries:.0f}\\% \\\\",
        r"\midrule",
        r"\textbf{Industry $\times$ year cells (of " + f"{n_cells_total:,}" + r" total)} & & \\",
        f"  significant $+U$ on $\\Delta KL$ ($z > +1.96$) & {pos_U_cells:,} & {100*pos_U_cells/n_cells_total:.0f}\\% \\\\",
        f"  significant inverted-$U$ ($z < -1.96$) & {inv_U_cells:,} & {100*inv_U_cells/n_cells_total:.0f}\\% \\\\",
        f"  indeterminate quadratic & {flat_cells:,} & {100*flat_cells/n_cells_total:.0f}\\% \\\\",
        f"  linear advantage $+\\Delta KL$ ($z > +1.96$) & {adv_pos_cells:,} & {100*adv_pos_cells/n_cells_total:.0f}\\% \\\\",
        f"  linear advantage $-\\Delta KL$ ($z < -1.96$) & {adv_neg_cells:,} & {100*adv_neg_cells/n_cells_total:.0f}\\% \\\\",
        f"  neither (linear $|z| \\leq 1.96$) & {no_adv_cells:,} & {100*no_adv_cells/n_cells_total:.0f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUT / "u_shape_industry_year_table.tex").write_text("\n".join(tex))
    print("\nHEADLINE TABLE:\n" + "\n".join(tex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
