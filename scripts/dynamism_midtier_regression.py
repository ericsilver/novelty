"""C2 regression (reframed): does mid-tier multi-filer proliferation
predict falling debut share?

C1 showed that top-10 owner share is *falling* in every class. So the
"FAANG crowding" story does not hold. The C2 reframe asks whether the
within-class debut-share decline in tech classes is instead driven by
mid-tier multi-filer proliferation: owners with 2-9 filings per year
who are not first-time filers but also not top-10 incumbents.

Per (class, year) we compute:
  - debut_share          = debuts / total filings
  - single_filer_share   = filings by owners with exactly 1 filing in this class-year
  - midtier_share        = filings by owners with 2-9 filings in this class-year
  - heavy_share          = filings by owners with 10+ filings in this class-year
  - top10_share          = (already computed in C1) share to top-10 owners

Regression: ΔDebutShare[t] ~ ΔMidTierShare[t-1] + ΔHeavyShare[t-1]
                            + ΔTop10Share[t-1] + class FE + year FE.

We expect: in tech classes (009, 038, 042, 016) midtier_share rises and
debut_share falls together. Across the full panel the predictions are
ambiguous because the consumer-goods classes show the opposite pattern.

Outputs:
  paper/results/dynamism_midtier_panel.csv        (class x year x predictors)
  paper/results/dynamism_midtier_regression.tex   (regression table)
  paper/results/dynamism_midtier_regression.txt   (interpretation)
  paper/results/dynamism_midtier_scatter.png      (scatter of slopes)
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from novelty.industries import name as industry_name

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES  = REPO / "paper" / "results"

YEAR_MIN = 1985
YEAR_MAX = 2025


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)

    # Recompute debut years (owner -> earliest filing across all classes)
    print("[pass 1] owner debut dates …", flush=True)
    parts = []
    for cls in class_list():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                              columns=["owner_name","filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8))
        d = tm.group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        parts.append(d)
        del tm
        gc.collect()
    debut = (pl.concat(parts).group_by("owner_name")
                .agg(pl.col("debut_date").min().alias("debut_date")))
    print(f"  {debut.height:,} owners", flush=True)

    # Pass 2: per (class, year) panel of multi-filer shares + debut share
    print("[pass 2] per-class panel …", flush=True)
    rows = []
    for cls in class_list():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                              columns=["owner_name","filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(pl.col("filing_date").str.slice(0,4).cast(pl.Int64).alias("year"))
        tm = tm.filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        tm = tm.join(debut, on="owner_name", how="left").with_columns(
            (pl.col("filing_date") == pl.col("debut_date")).alias("is_debut"))
        # Per year, owner counts
        owner_year = tm.group_by(["year","owner_name"]).agg(pl.len().alias("k"))
        for yr in range(YEAR_MIN, YEAR_MAX+1):
            sy = owner_year.filter(pl.col("year") == yr)
            if sy.height == 0:
                continue
            n_total = int(sy["k"].sum())
            n_single = int(sy.filter(pl.col("k") == 1)["k"].sum())
            n_midtier = int(sy.filter(pl.col("k").is_between(2,9))["k"].sum())
            n_heavy = int(sy.filter(pl.col("k") >= 10)["k"].sum())
            top10 = sy.sort("k", descending=True).head(10)
            n_top10 = int(top10["k"].sum())
            n_debuts = int(tm.filter((pl.col("year") == yr) & pl.col("is_debut")).height)
            rows.append({
                "cls": cls, "class_name": industry_name(cls), "year": yr,
                "n_total": n_total, "n_debuts": n_debuts,
                "debut_share": n_debuts / n_total,
                "single_share": n_single / n_total,
                "midtier_share": n_midtier / n_total,
                "heavy_share": n_heavy / n_total,
                "top10_share": n_top10 / n_total,
            })
        del tm, owner_year
        gc.collect()

    panel = pl.from_dicts(rows).sort(["cls","year"])
    panel.write_csv(RES / "dynamism_midtier_panel.csv")
    print(f"\nPanel built: {panel.height:,} (class x year) cells", flush=True)

    # Cross-section descriptive: slope of midtier_share vs slope of debut_share per class
    slopes = []
    for cls in sorted(panel["cls"].unique().to_list()):
        sub = panel.filter(pl.col("cls") == cls).sort("year").to_pandas()
        x = sub["year"].to_numpy().astype(float)
        slopes.append({
            "cls": cls, "class_name": sub["class_name"].iloc[0],
            "slope_debut": np.polyfit(x, sub["debut_share"], 1)[0],
            "slope_midtier": np.polyfit(x, sub["midtier_share"], 1)[0],
            "slope_heavy": np.polyfit(x, sub["heavy_share"], 1)[0],
            "slope_top10": np.polyfit(x, sub["top10_share"], 1)[0],
        })
    sl = pd.DataFrame(slopes)
    r_midtier = sl[["slope_midtier","slope_debut"]].corr().iloc[0,1]
    r_heavy = sl[["slope_heavy","slope_debut"]].corr().iloc[0,1]
    r_top10 = sl[["slope_top10","slope_debut"]].corr().iloc[0,1]

    print(f"\nCross-class correlations of long-run slopes:")
    print(f"  corr(slope_midtier, slope_debut) = {r_midtier:+.3f}")
    print(f"  corr(slope_heavy,   slope_debut) = {r_heavy:+.3f}")
    print(f"  corr(slope_top10,   slope_debut) = {r_top10:+.3f}")

    # Panel regression: ΔDebut[t] ~ ΔMidTier[t-1] + ΔHeavy[t-1] + ΔTop10[t-1] + FE
    df = panel.to_pandas().sort_values(["cls","year"]).reset_index(drop=True)
    for col in ["debut_share","midtier_share","heavy_share","top10_share"]:
        df[f"d_{col}"] = df.groupby("cls")[col].diff()
        df[f"d_{col}_lag1"] = df.groupby("cls")[f"d_{col}"].shift(1)
    reg_df = df.dropna(subset=["d_debut_share","d_midtier_share_lag1",
                                 "d_heavy_share_lag1","d_top10_share_lag1"]).copy()
    # Class and year FE via dummies
    reg_df = pd.get_dummies(reg_df, columns=["cls","year"], drop_first=True, dtype=float)
    Xcols = ["d_midtier_share_lag1","d_heavy_share_lag1","d_top10_share_lag1"] \
            + [c for c in reg_df.columns if c.startswith("cls_") or c.startswith("year_")]
    X = sm.add_constant(reg_df[Xcols].astype(float))
    y = reg_df["d_debut_share"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="cluster",
                              cov_kwds={"groups": reg_df.index // 1})  # placeholder; we'll use class
    # Cluster on class
    cls_dummies = [c for c in reg_df.columns if c.startswith("cls_")]
    if cls_dummies:
        # Reconstruct class labels for clustering
        class_index = reg_df[cls_dummies].idxmax(axis=1).fillna("cls_BASE").values
    else:
        class_index = np.array(["all"] * len(reg_df))
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": class_index})

    # Pull just the predictors of interest
    pred_summary = []
    for name in ["d_midtier_share_lag1","d_heavy_share_lag1","d_top10_share_lag1"]:
        b = model.params[name]; se = model.bse[name]; t = b/se
        pred_summary.append((name, b, se, t, model.pvalues[name]))

    txt = []
    txt.append("C2 reframed regression: does mid-tier multi-filer proliferation predict")
    txt.append("a fall in within-class debut share, with class and year fixed effects?")
    txt.append("=" * 70)
    txt.append("")
    txt.append(f"Panel cells used:    {len(reg_df):,}  ({len(reg_df['cls' if 'cls' in reg_df else cls_dummies[0]].unique() if cls_dummies else [None]):,} classes x ~years)")
    txt.append(f"R² (within):         {model.rsquared:.4f}")
    txt.append(f"R² adjusted:         {model.rsquared_adj:.4f}")
    txt.append("")
    txt.append("Coefficients on lagged-Δ share predictors (class FE + year FE; cluster-by-class SE):")
    txt.append("")
    txt.append(f"  {'predictor':30s} {'coef':>10s}  {'SE':>10s}  {'t':>7s}  {'p':>8s}")
    for name, b, se, t, p in pred_summary:
        txt.append(f"  {name:30s} {b:+10.4f}  {se:10.4f}  {t:+7.2f}  {p:8.4f}")
    txt.append("")
    txt.append("Interpretation:")
    txt.append(f"  - corr(Δslope midtier, Δslope debut)  across classes: {r_midtier:+.3f}")
    txt.append(f"  - corr(Δslope heavy,   Δslope debut)  across classes: {r_heavy:+.3f}")
    txt.append(f"  - corr(Δslope top10,   Δslope debut)  across classes: {r_top10:+.3f}")
    txt.append("")
    txt.append("Causal-vs-common-shock caveat: this is a panel of partial correlations with")
    txt.append("class FE + year FE. It removes class-level constants and secular trends, but")
    txt.append("a within-class shock that simultaneously raises mid-tier filing and lowers")
    txt.append("debut filing (e.g. an IP-thicket regime in software) would generate the same")
    txt.append("signature. Read as 'consistent with', not 'caused by'.")
    text = "\n".join(txt)
    (RES / "dynamism_midtier_regression.txt").write_text(text)
    print("\n" + text)

    # Tex table for paper
    tex = [r"\begin{tabular}{lrrr}",
           r"\toprule",
           r"Predictor (lagged $\Delta$) & coef & cluster SE & $t$ \\",
           r"\midrule",
           f"  $\\Delta$ mid-tier share (2--9 filings/year)   & {pred_summary[0][1]:+.4f} & {pred_summary[0][2]:.4f} & {pred_summary[0][3]:+.2f} \\\\",
           f"  $\\Delta$ heavy share (10+ filings/year)       & {pred_summary[1][1]:+.4f} & {pred_summary[1][2]:.4f} & {pred_summary[1][3]:+.2f} \\\\",
           f"  $\\Delta$ top-10 share                         & {pred_summary[2][1]:+.4f} & {pred_summary[2][2]:.4f} & {pred_summary[2][3]:+.2f} \\\\",
           r"\midrule",
           r"  Class fixed effects & \multicolumn{3}{c}{Yes} \\",
           r"  Year fixed effects  & \multicolumn{3}{c}{Yes} \\",
           f"  $N$ (class $\\times$ year cells) & \\multicolumn{{3}}{{c}}{{{len(reg_df):,}}} \\\\",
           f"  $R^2$  & \\multicolumn{{3}}{{c}}{{{model.rsquared:.3f}}} \\\\",
           r"\bottomrule",
           r"\end{tabular}",]
    (RES / "dynamism_midtier_regression.tex").write_text("\n".join(tex))

    # Scatter of class-level slopes: midtier rise vs debut fall
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, key, label in [(axes[0], "slope_midtier", "Mid-tier (2-9 filings/yr)"),
                            (axes[1], "slope_top10", "Top-10 owners")]:
        ax.scatter(sl[key]*100, sl["slope_debut"]*100, s=40, alpha=0.7,
                   edgecolor="#333", linewidth=0.5)
        for _, r in sl.iterrows():
            ax.annotate(r["cls"], (r[key]*100, r["slope_debut"]*100),
                        fontsize=7, alpha=0.7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        rr = sl[[key,"slope_debut"]].corr().iloc[0,1]
        ax.set_xlabel(f"Slope of {label} share (pp/year)")
        ax.set_ylabel("Slope of within-class debut share (pp/year)")
        ax.set_title(f"{label}: cross-class r = {rr:+.3f}")
        ax.grid(alpha=0.3)
    fig.suptitle("Long-run per-class slopes 1985-2025: does mid-tier share rise where debut share falls?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "dynamism_midtier_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] wrote midtier regression outputs to {RES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
