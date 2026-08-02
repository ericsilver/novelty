"""Temporal stability of the survival U-shape: how does the per-year
quadratic on ΔKL look, and is the pooled result driven by a few years?

For each filing-year cohort 1990-2021, fit a logistic regression
survived_5y ~ z(ΔKL) + z(ΔKL)^2 + log(n_terms) (no year controls,
since we are conditioning on year). Repeat:
  - on the pooled corpus across all 44 industries (the headline)
  - on five representative classes:
      009 Software & Electronics      (was inverted, now positive U)
      005 Pharmaceuticals             (strong positive U)
      016 Paper, Print & Office       (strongest positive U)
      032 Beer & Soft Drinks          (inverted-U)
      038 Telecommunications          (strongest inverted-U)

Then overlay U.S. recession bars (NBER 2001, 2008-09, 2020) on the
plot to see if U-strength tracks the macro cycle.

Outputs:
  paper/results/u_shape_temporal_stability.png
  paper/results/u_shape_temporal_stability.csv
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

YEAR_LO, YEAR_HI = 1990, 2021
RECESSIONS = [(2001, 2001), (2008, 2009), (2020, 2020)]
CLASSES_OF_INTEREST = [
    ("pooled",  None,  "Pooled (all 44 classes)"),
    ("009",     "009", "Software & Electronics (009)"),
    ("005",     "005", "Pharmaceuticals (005)"),
    ("016",     "016", "Paper, Print & Office (016)"),
    ("032",     "032", "Beer & Soft Drinks (032)"),
    ("038",     "038", "Telecommunications (038)"),
]


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def load_panel(cls_subset: list[str] | None = None) -> pl.DataFrame:
    parts = []
    classes = class_list() if cls_subset is None else cls_subset
    for cls in classes:
        sp = PROC / f"surprise_class{cls}.parquet"
        op = PROC / f"outcomes_class{cls}.parquet"
        if not (sp.exists() and op.exists()):
            continue
        s = pl.read_parquet(
            sp,
            columns=["serial_number", "year", "kl_vs_past", "kl_vs_future",
                     "n_ref_past", "n_ref_future", "n_terms"],
        ).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
            & pl.col("year").is_between(YEAR_LO, YEAR_HI)
        ).with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        )
        o = pl.read_parquet(op, columns=["serial_number", "survived_5y"])
        j = s.join(o, on="serial_number", how="inner").select(
            "year","dkl","n_terms","survived_5y"
        )
        if j.height:
            parts.append(j)
        del s, o, j
        gc.collect()
    return pl.concat(parts)


def fit_year(sub_pd) -> dict | None:
    if len(sub_pd) < 1000:
        return None
    dkl = sub_pd["dkl"].to_numpy()
    sd = dkl.std()
    if sd == 0:
        return None
    z = (dkl - dkl.mean()) / sd
    log_n = np.log(np.clip(sub_pd["n_terms"].to_numpy(), 1, None))
    X = sm.add_constant(np.column_stack([z, z*z, log_n]))
    y = sub_pd["survived_5y"].astype(int).to_numpy()
    try:
        m = sm.Logit(y, X).fit(disp=False, maxiter=80)
    except Exception:
        return None
    return {
        "n": len(sub_pd),
        "beta_lin": float(m.params[1]),
        "beta_quad": float(m.params[2]),
        "se_quad": float(m.bse[2]),
        "t_quad": float(m.params[2] / m.bse[2]) if m.bse[2] > 0 else 0.0,
        "mean_survival": float(y.mean()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    panels = {}
    for label, cls, _ in CLASSES_OF_INTEREST:
        print(f"[load] {label} …", flush=True)
        panels[label] = load_panel(None if cls is None else [cls])
        print(f"   n = {panels[label].height:,}", flush=True)
    for label, cls, title in CLASSES_OF_INTEREST:
        df = panels[label]
        for yr in range(YEAR_LO, YEAR_HI + 1):
            sub = df.filter(pl.col("year") == yr).to_pandas()
            r = fit_year(sub)
            if r is None: continue
            all_rows.append({"series": label, "year": yr, **r})
    out_df = pl.from_dicts(all_rows)
    out_df.write_csv(OUT / "u_shape_temporal_stability.csv")
    print(f"[done] {out_df.height} year-series rows", flush=True)

    # Plot per-year beta_quad over time for each series
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    palette = {
        "pooled":  ("#222222", 2.5, "-"),
        "009":     ("#2b6cb0", 1.5, "-"),
        "005":     ("#229922", 1.5, "-"),
        "016":     ("#aa55cc", 1.5, "-"),
        "032":     ("#cc4444", 1.5, "--"),
        "038":     ("#cc7700", 1.5, "--"),
    }
    pdf = out_df.to_pandas()

    # Top panel: beta_quad over years with CI bands
    ax = axes[0]
    for label, _, title in CLASSES_OF_INTEREST:
        sub = pdf[pdf["series"] == label].sort_values("year")
        if sub.empty: continue
        col, lw, ls = palette[label]
        ax.plot(sub["year"], sub["beta_quad"], ls + "o", color=col, lw=lw, ms=4,
                label=title)
        ax.fill_between(sub["year"], sub["beta_quad"] - 1.96 * sub["se_quad"],
                         sub["beta_quad"] + 1.96 * sub["se_quad"],
                         color=col, alpha=0.10)
    ax.axhline(0, ls=":", color="#666", lw=1)
    # Recession bars
    for r_lo, r_hi in RECESSIONS:
        ax.axvspan(r_lo - 0.5, r_hi + 0.5, color="#eecccc", alpha=0.4, lw=0)
    ax.set_ylabel(r"$\beta_{(\Delta KL)^2}$ on survived\_5y  (per-year fit)")
    ax.set_title(r"Per-year quadratic on $\Delta KL$: does the U survive year-by-year?"
                 + "\n(positive = U-shape, negative = inverted-U; pink bands = NBER recessions)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9, ncol=2)

    # Bottom panel: mean survival rate over years (sanity for macro)
    ax2 = axes[1]
    for label, _, title in CLASSES_OF_INTEREST:
        sub = pdf[pdf["series"] == label].sort_values("year")
        if sub.empty: continue
        col, lw, ls = palette[label]
        ax2.plot(sub["year"], sub["mean_survival"], ls + "o", color=col, lw=lw, ms=4)
    for r_lo, r_hi in RECESSIONS:
        ax2.axvspan(r_lo - 0.5, r_hi + 0.5, color="#eecccc", alpha=0.4, lw=0)
    ax2.set_ylabel("mean 5y survival rate")
    ax2.set_xlabel("filing year")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "u_shape_temporal_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {OUT/'u_shape_temporal_stability.png'}", flush=True)

    # Print summary table
    summary = (out_df.group_by("series").agg([
        pl.len().alias("n_years"),
        (pl.col("beta_quad") > 0).cast(pl.Float64).mean().alias("share_years_positive_U"),
        (pl.col("t_quad") > 1.96).cast(pl.Float64).mean().alias("share_years_sig_positive"),
        (pl.col("t_quad") < -1.96).cast(pl.Float64).mean().alias("share_years_sig_inverted"),
        pl.col("beta_quad").mean().alias("mean_beta_quad"),
        pl.col("beta_quad").median().alias("median_beta_quad"),
        pl.col("t_quad").mean().alias("mean_t_quad"),
    ]))
    print()
    print(summary.to_pandas().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
