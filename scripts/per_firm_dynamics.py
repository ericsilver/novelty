"""Per-firm temporal dynamics + debut-quartile analysis.

Three questions:
  (a) Do surprising trademarks predict future surprising trademarks
      (within-firm autocorrelation of prospective KL)?
  (b) Do innovative trademarks predict future innovative trademarks
      (within-firm autocorrelation of dKL)?
  (c) Does a firm's innovative capacity tend to increase or decrease
      with experience (slope of dKL on filing-order within firm)?

Then split firms into quartiles by debut-filing dKL and re-run (a)-(c)
within each quartile. Hypothesis: firms that self-identify as innovative
on their first filing keep doing so.

Outputs (in paper/results/):
  per_firm_dynamics.tex
  per_firm_dynamics_metrics.json
  debut_quartile_dynamics.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"


def load_universe() -> pl.DataFrame:
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
    universe = load_universe().filter(
        pl.col("owner_name").is_not_null()
        & (pl.col("n_ref_past") >= 1000)
        & (pl.col("n_ref_future") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= 1990) & (pl.col("year") <= 2020)
    )
    print(f"       {universe.height:,} clean filings (1995-2020)")

    # Per-firm filing sequence: keep firms with >=2 filings
    multi = universe.with_columns(
        pl.len().over("owner_name").alias("n_filings_total")
    ).filter(pl.col("n_filings_total") >= 2)
    print(f"       {multi.height:,} filings from {multi['owner_name'].n_unique():,} multi-filing firms")

    # Per-firm AR(1) and trend
    pdf = multi.select(
        "owner_name", "filing_date", "year",
        "kl_vs_past", "kl_vs_future", "dkl",
        "n_filings_total",
    ).sort(["owner_name", "filing_date"]).to_pandas()
    pdf["pros_lag"] = pdf.groupby("owner_name")["kl_vs_past"].shift(1)
    pdf["dkl_lag"] = pdf.groupby("owner_name")["dkl"].shift(1)
    pdf["filing_idx"] = pdf.groupby("owner_name").cumcount()  # 0,1,2,... per firm
    pdf["year_centered"] = pdf["year"] - pdf.groupby("owner_name")["year"].transform("mean")

    # Pearson AR(1) (across all consecutive filing pairs in all firms)
    valid = pdf.dropna(subset=["pros_lag", "dkl_lag"])
    rho_pros = float(np.corrcoef(valid["pros_lag"], valid["kl_vs_past"])[0, 1])
    rho_dkl = float(np.corrcoef(valid["dkl_lag"], valid["dkl"])[0, 1])
    n_pairs = len(valid)
    print(f"\n[AR(1)] across {n_pairs:,} consecutive filing pairs:")
    print(f"  Cor(prospective_KL_t, prospective_KL_{{t-1}}) within firm = {rho_pros:+.3f}")
    print(f"  Cor(dKL_t,            dKL_{{t-1}})            within firm = {rho_dkl:+.3f}")

    # Per-firm trend: slope of dKL on filing index (so "rises with experience" if positive)
    # Compute per-firm slope only for firms with >=4 filings
    print(f"\n[trend] per-firm slope of dKL on filing-index:", flush=True)
    big = pdf.groupby("owner_name").filter(lambda g: len(g) >= 4)
    print(f"  among {big['owner_name'].nunique():,} firms with >=4 filings")
    def slope(g):
        x = g["filing_idx"].values.astype(float)
        y = g["dkl"].values.astype(float)
        if x.std() == 0 or len(x) < 2:
            return np.nan
        return float(np.cov(x, y, bias=False)[0, 1] / x.var(ddof=1))
    firm_slopes = big.groupby("owner_name").apply(slope, include_groups=False)
    firm_slopes = firm_slopes.dropna()
    print(f"  median slope: {firm_slopes.median():+.4f}, mean slope: {firm_slopes.mean():+.4f}")
    print(f"  share of firms with positive slope: {(firm_slopes > 0).mean()*100:.1f}%")
    n_pos = int((firm_slopes > 0).sum())
    n_neg = int((firm_slopes < 0).sum())
    binom_p = float(0.5)  # null hypothesis is 50/50
    n_total = n_pos + n_neg
    z = (n_pos - n_total/2) / np.sqrt(n_total/4)
    print(f"  +slopes / total: {n_pos}/{n_total} (binomial z = {z:+.2f})")

    # ---- Quartile analysis by debut dKL ----
    debut = pdf.sort_values(["owner_name", "filing_date"]).drop_duplicates("owner_name", keep="first")
    debut["debut_q"] = pd.qcut(debut["dkl"], 4, labels=["Q1 (least novel)", "Q2", "Q3", "Q4 (most novel)"])
    quartile_map = debut.set_index("owner_name")["debut_q"]
    pdf["debut_q"] = pdf["owner_name"].map(quartile_map)

    rows = []
    for q in ["Q1 (least novel)", "Q2", "Q3", "Q4 (most novel)"]:
        sub = pdf[pdf["debut_q"] == q]
        if len(sub) == 0:
            continue
        # Subsequent filings (excluding debut)
        sub_seq = sub[sub["filing_idx"] >= 1]
        # AR(1) within quartile
        v_q = sub.dropna(subset=["pros_lag", "dkl_lag"])
        rho_p = float(np.corrcoef(v_q["pros_lag"], v_q["kl_vs_past"])[0, 1]) if len(v_q) >= 50 else np.nan
        rho_d = float(np.corrcoef(v_q["dkl_lag"], v_q["dkl"])[0, 1]) if len(v_q) >= 50 else np.nan
        # Mean dKL of subsequent filings
        mean_dkl_sub = float(sub_seq["dkl"].mean()) if len(sub_seq) > 0 else np.nan
        debut_dkl = float(debut[debut["debut_q"] == q]["dkl"].mean())
        rows.append({
            "quartile": q,
            "n_firms": int(debut[debut["debut_q"] == q]["owner_name"].nunique()),
            "mean_debut_dkl": debut_dkl,
            "mean_subsequent_dkl": mean_dkl_sub,
            "ar1_pros": rho_p,
            "ar1_dkl": rho_d,
        })
    print("\n[quartiles] by debut dKL:")
    for r in rows:
        print(f"  {r['quartile']:<22}  n={r['n_firms']:>7,}  debut_dKL={r['mean_debut_dkl']:+.3f}  "
              f"subseq_dKL={r['mean_subsequent_dkl']:+.3f}  AR1_pros={r['ar1_pros']:+.3f}  AR1_dkl={r['ar1_dkl']:+.3f}")

    # LaTeX outputs
    main_tex = (
        "\\begin{tabular}{lr}\n\\toprule\n"
        f"Within-firm AR(1) of prospective KL & {rho_pros:+.3f} \\\\\n"
        f"Within-firm AR(1) of $\\Delta KL$ (innovator score) & {rho_dkl:+.3f} \\\\\n"
        f"\\midrule\n"
        f"N consecutive filing-pairs & {n_pairs:,} \\\\\n"
        f"N multi-filing firms & {pdf['owner_name'].nunique():,} \\\\\n"
        f"\\midrule\n"
        f"Median per-firm slope of $\\Delta KL$ on filing index & {firm_slopes.median():+.4f} \\\\\n"
        f"Mean per-firm slope of $\\Delta KL$ on filing index & {firm_slopes.mean():+.4f} \\\\\n"
        f"Share of firms with positive slope & {(firm_slopes > 0).mean()*100:.1f}\\% \\\\\n"
        f"N firms with $\\geq 4$ filings used for slopes & {len(firm_slopes):,} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "per_firm_dynamics.tex").write_text(main_tex)

    qtex = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Quartile (by debut $\\Delta KL$) & N firms & Debut $\\Delta KL$ & "
        "Subseq.\\ mean $\\Delta KL$ & AR(1) prospective & AR(1) $\\Delta KL$ \\\\\n\\midrule\n"
    )
    for r in rows:
        qname = r["quartile"].replace("(", "\\,(").replace(")", ")")
        qtex += (
            f"{qname} & {r['n_firms']:,} & "
            f"{r['mean_debut_dkl']:+.3f} & {r['mean_subsequent_dkl']:+.3f} & "
            f"{r['ar1_pros']:+.3f} & {r['ar1_dkl']:+.3f} \\\\\n"
        )
    qtex += "\\bottomrule\n\\end{tabular}\n"
    (RESULTS / "debut_quartile_dynamics.tex").write_text(qtex)

    metrics = {
        "n_pairs": n_pairs,
        "ar1_pros": rho_pros,
        "ar1_dkl": rho_dkl,
        "median_per_firm_slope": float(firm_slopes.median()),
        "mean_per_firm_slope": float(firm_slopes.mean()),
        "share_positive_slope": float((firm_slopes > 0).mean()),
        "n_firms_for_slopes": int(len(firm_slopes)),
        "quartiles": rows,
    }
    (RESULTS / "per_firm_dynamics_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    print(f"\n[done] wrote {RESULTS}/per_firm_dynamics.tex + debut_quartile_dynamics.tex")
    return 0


if __name__ == "__main__":
    import pandas as pd  # noqa: needed for qcut/groupby
    raise SystemExit(main())
