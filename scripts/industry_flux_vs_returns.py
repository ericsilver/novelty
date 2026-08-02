"""Per-industry comparison: do industries 'in flux' (high std(ΔKL),
high mean |ΔKL|, high mean prospective KL) have supernumerary returns
relative to the S&P 500?

For each of the 44 NICE classes with sufficient data we compute:
  Flux measures (all clean H=2y filings in the class):
    mean_pros_kl       — mean prospective KL (how unusual vs recent past)
    mean_dkl           — mean ΔKL (signed directional flux)
    std_dkl            — std ΔKL across filings (flux intensity)
    mean_abs_dkl       — mean |ΔKL| (flux magnitude)
  Financial measures (SEC-crosswalked firms whose modal class is the focal class):
    n_firms            — number of public firms
    median_gross_margin
    median_op_margin
    mean_excess_4y     — mean (ret_4y - sp500_ret_4y) across firm-years
    mean_excess_5y     — mean (ret_5y - sp500_ret_5y) across firm-years
    median_revenue_growth_5y — median (rev(t+5) / rev(t)) within firm-pairs

Then test:
  corr(flux measure, excess return) across classes
  corr(flux measure, margin) across classes

Outputs:
  paper/results/industry_flux_vs_returns.csv
  paper/results/industry_flux_vs_returns.png
  paper/results/industry_flux_vs_returns.tex
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. Per-class flux statistics + owner -> class filings count ----
    print("[step 1] per-class flux stats + owner-class filings …", flush=True)
    class_flux = []
    owner_cls_counts = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        tm = PROC / f"tm_class{cls}.parquet"
        if not sp.exists():
            continue
        s = pl.read_parquet(
            sp,
            columns=["serial_number","owner_name","year","kl_vs_past",
                     "kl_vs_future","n_ref_past","n_ref_future","n_terms"],
        ).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
            & pl.col("year").is_between(1990, 2021)
        ).with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        )
        if s.height < 5_000:
            continue
        class_flux.append({
            "cls": cls,
            "name": industry_name(cls),
            "n_filings": s.height,
            "mean_pros_kl": float(s["kl_vs_past"].mean()),
            "mean_dkl": float(s["dkl"].mean()),
            "std_dkl": float(s["dkl"].std()),
            "mean_abs_dkl": float(s["dkl"].abs().mean()),
        })
        owner_cls = (s.filter(pl.col("owner_name").is_not_null())
                       .group_by("owner_name").agg(pl.len().alias("n"))
                       .with_columns(pl.lit(cls).alias("cls")))
        owner_cls_counts.append(owner_cls)
        del s
        gc.collect()
        print(f"  {cls}: {class_flux[-1]['n_filings']:,} filings", flush=True)

    flux_df = pl.from_dicts(class_flux)
    print(f"[done] {flux_df.height} classes with >=5,000 clean filings", flush=True)

    # ---- 2. Owner -> primary class (the class where this owner files most) ----
    print("[step 2] determining each owner's primary class …", flush=True)
    own = pl.concat(owner_cls_counts)
    primary = (own.sort(["owner_name", "n"], descending=[False, True])
                  .group_by("owner_name", maintain_order=True)
                  .first()
                  .rename({"cls": "primary_cls", "n": "n_in_primary_cls"}))
    print(f"  {primary.height:,} unique owners assigned a primary class", flush=True)

    # ---- 3. Join SEC crosswalk and financials ----
    print("[step 3] joining SEC crosswalk + financials + returns …", flush=True)
    xw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name","cik")
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet").select(
        "cik","fy","revenue","gross_margin","operating_income","net_income"
    )
    ret = pl.read_parquet(PROC / "stock_returns.parquet").select(
        "cik","base_year","ret_4y","sp500_ret_4y","ret_5y","sp500_ret_5y"
    ).with_columns([
        (pl.col("ret_4y") - pl.col("sp500_ret_4y")).alias("excess_4y"),
        (pl.col("ret_5y") - pl.col("sp500_ret_5y")).alias("excess_5y"),
    ])

    sec_per_cik = sec.with_columns(
        (pl.col("operating_income") / pl.col("revenue")).alias("op_margin"),
        (pl.col("net_income") / pl.col("revenue")).alias("net_margin"),
    ).group_by("cik").agg([
        pl.col("gross_margin").median().alias("median_gross_margin"),
        pl.col("op_margin").median().alias("median_op_margin"),
        pl.col("net_margin").median().alias("median_net_margin"),
        pl.col("revenue").max().alias("max_revenue"),
    ])
    ret_per_cik = ret.group_by("cik").agg([
        pl.col("excess_4y").mean().alias("mean_excess_4y"),
        pl.col("excess_5y").mean().alias("mean_excess_5y"),
        pl.col("ret_4y").mean().alias("mean_ret_4y"),
        pl.col("ret_5y").mean().alias("mean_ret_5y"),
    ])
    firms = (primary.join(xw, on="owner_name", how="inner")
                    .join(sec_per_cik, on="cik", how="left")
                    .join(ret_per_cik, on="cik", how="left"))
    print(f"  {firms.height:,} crosswalked firms with a primary class", flush=True)

    # ---- 4. Aggregate per class ----
    print("[step 4] aggregating per class …", flush=True)
    agg = (firms.group_by("primary_cls").agg([
        pl.len().alias("n_firms_sec"),
        pl.col("median_gross_margin").median().alias("class_median_gross_margin"),
        pl.col("median_op_margin").median().alias("class_median_op_margin"),
        pl.col("median_net_margin").median().alias("class_median_net_margin"),
        pl.col("mean_excess_4y").median().alias("class_median_excess_4y"),
        pl.col("mean_excess_5y").median().alias("class_median_excess_5y"),
        pl.col("max_revenue").median().alias("class_median_max_revenue"),
    ]).rename({"primary_cls": "cls"}))
    out = flux_df.join(agg, on="cls", how="left")
    out = out.sort("mean_abs_dkl", descending=True)
    out.write_csv(OUT / "industry_flux_vs_returns.csv")

    pdf = out.to_pandas()
    # Show core table
    print("\n=== PER-INDUSTRY FLUX vs FINANCIAL OUTCOMES ===")
    cols = ["cls","name","n_filings","mean_pros_kl","mean_dkl","std_dkl","mean_abs_dkl",
            "n_firms_sec","class_median_gross_margin","class_median_op_margin",
            "class_median_excess_4y","class_median_excess_5y"]
    print(pdf[cols].round(4).to_string(index=False))

    # ---- 5. Correlations across classes ----
    print("\n=== CROSS-INDUSTRY CORRELATIONS ===")
    pairs = [
        ("mean_pros_kl",  "class_median_gross_margin"),
        ("mean_pros_kl",  "class_median_op_margin"),
        ("mean_pros_kl",  "class_median_excess_4y"),
        ("mean_pros_kl",  "class_median_excess_5y"),
        ("mean_dkl",      "class_median_gross_margin"),
        ("mean_dkl",      "class_median_excess_4y"),
        ("mean_dkl",      "class_median_excess_5y"),
        ("std_dkl",       "class_median_gross_margin"),
        ("std_dkl",       "class_median_excess_4y"),
        ("std_dkl",       "class_median_excess_5y"),
        ("mean_abs_dkl",  "class_median_gross_margin"),
        ("mean_abs_dkl",  "class_median_excess_4y"),
        ("mean_abs_dkl",  "class_median_excess_5y"),
    ]
    corr_rows = []
    for x, y in pairs:
        sub = pdf[[x, y]].dropna()
        if len(sub) < 5: continue
        r_p = sub.corr(method="pearson").iloc[0,1]
        r_s = sub.corr(method="spearman").iloc[0,1]
        corr_rows.append({"x": x, "y": y, "n": len(sub),
                          "pearson_r": r_p, "spearman_r": r_s})
        print(f"  {x:<20s} vs {y:<28s}:  Pearson={r_p:+.3f}  Spearman={r_s:+.3f}  n={len(sub)}")
    pl.from_dicts(corr_rows).write_csv(OUT / "industry_flux_vs_returns_correlations.csv")

    # ---- 6. Plot 2x2: pros vs margin/return, abs_dkl vs margin/return ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    cfg = [
        ("mean_pros_kl", "mean prospective KL (industry-mean)",
         "class_median_gross_margin", "median gross margin (industry-median across firms)", axes[0,0]),
        ("mean_pros_kl", "mean prospective KL (industry-mean)",
         "class_median_excess_4y", "median excess 4y return vs S\\&P (industry-mean across firms)", axes[0,1]),
        ("mean_abs_dkl", r"mean $|\Delta KL|$ — flux intensity (industry-mean)",
         "class_median_gross_margin", "median gross margin", axes[1,0]),
        ("mean_abs_dkl", r"mean $|\Delta KL|$ — flux intensity (industry-mean)",
         "class_median_excess_4y", "median excess 4y return vs S\\&P", axes[1,1]),
    ]
    for x, xlbl, y, ylbl, ax in cfg:
        sub = pdf[["cls","name",x,y,"n_firms_sec"]].dropna()
        ax.scatter(sub[x], sub[y], s=40 + sub["n_firms_sec"].clip(upper=400)/8,
                   alpha=0.7, color="#2b6cb0", edgecolor="black")
        # label outliers
        if len(sub) >= 5:
            # Top 3 and bottom 3 by y
            for _, r in sub.nlargest(3, y).iterrows():
                ax.annotate(f"{r['cls']}", (r[x], r[y]), fontsize=8, alpha=0.8,
                            xytext=(4,4), textcoords="offset points")
            for _, r in sub.nsmallest(3, y).iterrows():
                ax.annotate(f"{r['cls']}", (r[x], r[y]), fontsize=8, alpha=0.8,
                            xytext=(4,-10), textcoords="offset points")
            m, b = np.polyfit(sub[x], sub[y], 1)
            xs = np.linspace(sub[x].min(), sub[x].max(), 50)
            ax.plot(xs, m*xs + b, ls="--", color="#cc4444", lw=1.5)
            r_p = sub[[x,y]].corr(method="pearson").iloc[0,1]
            ax.set_title(f"{ylbl}  (Pearson r = {r_p:+.2f}, n = {len(sub)} industries)",
                         fontsize=10)
        else:
            ax.set_title(ylbl)
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.grid(alpha=0.3)
        ax.axhline(0, ls=":", color="#999", lw=0.8)
    fig.suptitle("Per-industry flux vs financial outcomes (44 NICE classes)\n"
                 "Point size proportional to number of SEC-listed firms", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "industry_flux_vs_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {OUT/'industry_flux_vs_returns.png'}", flush=True)

    # LaTeX summary
    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Flux measure (industry mean) & Outcome & Pearson $r$ & $n$ \\",
           r"\midrule"]
    for r in corr_rows:
        x_pretty = {"mean_pros_kl": r"mean $KL_{\text{pros}}$",
                    "mean_dkl": r"mean $\Delta KL$",
                    "std_dkl": r"std $\Delta KL$",
                    "mean_abs_dkl": r"mean $|\Delta KL|$"}.get(r["x"], r["x"])
        y_pretty = {"class_median_gross_margin": "median gross margin",
                    "class_median_op_margin": "median operating margin",
                    "class_median_excess_4y": "median excess 4y return",
                    "class_median_excess_5y": "median excess 5y return"}.get(r["y"], r["y"])
        tex.append(f"{x_pretty} & {y_pretty} & {r['pearson_r']:+.3f} & {r['n']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "industry_flux_vs_returns.tex").write_text("\n".join(tex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
