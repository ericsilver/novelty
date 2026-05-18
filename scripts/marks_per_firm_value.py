"""Conditional on being publicly listed, does the number of trademark
filings a firm has predict firm value?

Hypothesis under test: in patents, count and value move together
(prolific patenters tend to have valuable patents). In trademarks, the
opposite might hold: a smaller portfolio could be evidence of focus,
and focus might be associated with higher per-firm value. Tested on the
SEC-crosswalked firms.

Per CIK:
  n_marks           total clean-panel trademark filings linked to this CIK
  max_revenue       firm size proxy (max over observed years)
  mean_gross_margin operating quality proxy
  mean_op_margin    operating-income / revenue proxy
  mean_rd_intensity rd_expense / revenue proxy (when reported)
  mean_excess_4y    excess returns over S&P at 4y horizon (if available)

Outputs:
  paper/results/marks_per_firm_value.png
  paper/results/marks_per_firm_value.csv
  paper/results/marks_per_firm_value.tex     (paper-ready summary table)
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def owner_filing_counts() -> pl.DataFrame:
    """Count clean H=2y trademark filings per owner_name."""
    parts = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        tm = PROC / f"tm_class{cls}.parquet"
        if not (sp.exists() and tm.exists()):
            continue
        s = pl.read_parquet(
            sp,
            columns=["serial_number","n_ref_prospective","n_ref_retrospective","n_terms",
                     "prospective_kl","retrospective_kl"],
        ).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
        ).select("serial_number")
        t = pl.read_parquet(tm, columns=["serial_number","owner_name"]).filter(
            pl.col("owner_name").is_not_null()
        )
        j = s.join(t, on="serial_number", how="inner").select("owner_name")
        parts.append(j)
        del s, t, j
        gc.collect()
    df = pl.concat(parts).group_by("owner_name").agg(pl.len().alias("n_marks"))
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[step 1] owner -> n_marks across the clean H=2y corpus", flush=True)
    owner_counts = owner_filing_counts()
    print(f"         {owner_counts.height:,} unique owners with >=1 clean filing", flush=True)

    print("[step 2] join to SEC crosswalk", flush=True)
    xw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name", "cik")
    # Roll-up: one CIK may have multiple owner_name strings -> sum n_marks
    linked = owner_counts.join(xw, on="owner_name", how="inner").group_by("cik").agg(
        pl.col("n_marks").sum().alias("n_marks")
    )
    print(f"         {linked.height:,} CIKs with at least one clean trademark filing", flush=True)

    print("[step 3] join to SEC financials (per-CIK rollup)", flush=True)
    fin = pl.read_parquet(PROC / "sec_firm_year.parquet").select(
        "cik","fy","revenue","gross_margin","operating_income","rd_expense","net_income"
    )
    rollup = (fin.group_by("cik")
                 .agg([
                     pl.col("revenue").max().alias("max_revenue"),
                     pl.col("revenue").median().alias("med_revenue"),
                     pl.col("gross_margin").mean().alias("mean_gross_margin"),
                     (pl.col("operating_income") / pl.col("revenue")).mean().alias("mean_op_margin"),
                     (pl.col("rd_expense") / pl.col("revenue")).mean().alias("mean_rd_intensity"),
                     (pl.col("net_income") / pl.col("revenue")).mean().alias("mean_net_margin"),
                     pl.col("fy").min().alias("first_year_observed"),
                     pl.col("fy").max().alias("last_year_observed"),
                     pl.len().alias("n_firm_years"),
                 ]))
    panel = linked.join(rollup, on="cik", how="inner")
    # Add log mark count + bin
    panel = panel.with_columns(
        pl.col("n_marks").log10().alias("log10_n_marks"),
    )
    print(f"         {panel.height:,} CIKs with both n_marks and financials", flush=True)

    panel.write_csv(OUT / "marks_per_firm_value.csv")

    # Correlations on log-marks vs each value proxy
    pdf = panel.to_pandas()
    pdf["log10_max_revenue"]  = np.log10(pdf["max_revenue"].clip(lower=1))
    pdf["log10_med_revenue"]  = np.log10(pdf["med_revenue"].clip(lower=1))

    corrs = []
    for col, label in [("log10_max_revenue", "log10(max revenue)"),
                       ("log10_med_revenue", "log10(median revenue)"),
                       ("mean_gross_margin", "mean gross margin"),
                       ("mean_op_margin",    "mean operating margin"),
                       ("mean_net_margin",   "mean net margin"),
                       ("mean_rd_intensity", "mean R&D intensity")]:
        sub = pdf[["log10_n_marks", col]].dropna()
        sub = sub[~np.isinf(sub[col])]
        if len(sub) < 30:
            continue
        r_p = sub.corr(method="pearson").iloc[0,1]
        r_s = sub.corr(method="spearman").iloc[0,1]
        corrs.append({"value_metric": label, "n": len(sub),
                      "pearson_r": r_p, "spearman_r": r_s})
        print(f"  corr(log10 n_marks, {label}):  Pearson={r_p:+.3f}  Spearman={r_s:+.3f}  n={len(sub):,}",
              flush=True)

    cdf = pl.from_dicts(corrs)
    cdf.write_csv(OUT / "marks_per_firm_value_correlations.csv")

    # Bin by log10(n_marks): show mean value metric per bin (deciles)
    print("\n[step 4] decile binning on log10(n_marks)", flush=True)
    pdf["bin"] = pd.qcut(pdf["log10_n_marks"], 10, duplicates="drop", labels=False) if False else None
    import pandas as pd
    pdf["bin"] = pd.qcut(pdf["log10_n_marks"], 10, duplicates="drop", labels=False)
    bins = (pdf.groupby("bin")
                .agg(n_marks_lo=("n_marks","min"),
                     n_marks_hi=("n_marks","max"),
                     n_firms=("cik","count"),
                     mean_log_max_revenue=("log10_max_revenue","mean"),
                     mean_gm=("mean_gross_margin","mean"),
                     mean_op=("mean_op_margin","mean"),
                     mean_rd=("mean_rd_intensity","mean"))
                .reset_index())
    print(bins.round(3).to_string(index=False))
    bins.to_csv(OUT / "marks_per_firm_value_bins.csv", index=False)

    # Plot: 4-panel scatter+binned
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    cfg = [
        ("log10_max_revenue", "$\\log_{10}$(max revenue, USD)", axes[0,0]),
        ("mean_gross_margin", "mean gross margin",              axes[0,1]),
        ("mean_op_margin",    "mean operating margin",          axes[1,0]),
        ("mean_rd_intensity", "mean R\\&D intensity",           axes[1,1]),
    ]
    for col, label, ax in cfg:
        sub = pdf[["log10_n_marks", col]].dropna()
        sub = sub[~np.isinf(sub[col])]
        if col != "log10_max_revenue":
            sub = sub[(sub[col] > -1) & (sub[col] < 2)]
        ax.scatter(sub["log10_n_marks"], sub[col], s=4, alpha=0.18, color="#888")
        # Binned mean line
        try:
            sub["xb"] = pd.qcut(sub["log10_n_marks"], 15, duplicates="drop", labels=False)
            b = sub.groupby("xb").agg(x=("log10_n_marks","mean"), y=(col, "mean")).reset_index()
            ax.plot(b["x"], b["y"], "-o", color="#cc4444", lw=2, ms=5)
        except ValueError:
            pass
        r_p = sub[["log10_n_marks", col]].corr(method="pearson").iloc[0,1]
        ax.set_xlabel("$\\log_{10}$(number of clean trademark filings)")
        ax.set_ylabel(label)
        ax.set_title(f"{label}  (Pearson r = {r_p:+.3f}, n = {len(sub):,})")
        ax.grid(alpha=0.3)
    fig.suptitle("Conditional on public listing: trademark filing count vs.\\ firm-value proxies\n"
                 f"$n = {panel.height:,}$ SEC-crosswalked firms", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "marks_per_firm_value.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {OUT/'marks_per_firm_value.png'}", flush=True)

    # LaTeX summary table
    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Value metric & $n$ firms & Pearson $r$ & Spearman $\rho$ \\",
           r"\midrule"]
    for r in corrs:
        label = r["value_metric"].replace("&", r"\&")
        tex.append(f"{label} & {r['n']:,} & "
                   f"{r['pearson_r']:+.3f} & {r['spearman_r']:+.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "marks_per_firm_value.tex").write_text("\n".join(tex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
