"""Debut-filings-only test of the harvest (negative-ΔKL) tail.

Restrict to each owner's first-ever clean filing across all classes.
On the negative-ΔKL half, fit an exponential survival curve:

  P(survived 5y | ΔKL = -x)  =  a + b * exp(-c * x)     for x > 0

Equivalently, fit log(rate - a) ~ -c*x.

Reports:
  - corr(ΔKL, survived_5y) on the full debut panel
  - corr(|ΔKL|, survived_5y) on the negative-ΔKL half only
  - exponential R^2 on the binned (|ΔKL|, survival) curve
  - same on the positive-ΔKL half for comparison

Outputs:
  paper/results/debut_harvest_fit.csv
  paper/results/debut_harvest_fit.png
  paper/results/debut_harvest_fit.txt
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Pass 1: owner -> earliest filing_date across all classes
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
        del tm, d
        gc.collect()
    debut = (pl.concat(parts).group_by("owner_name")
                .agg(pl.col("debut_date").min().alias("debut_date")))
    print(f"  {debut.height:,} unique owners", flush=True)

    print("[pass 2] debut-only panel with ΔKL + survived_5y …", flush=True)
    parts = []
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
        )
        o = pl.read_parquet(op, columns=["serial_number","owner_name","filing_date",
                                          "survived_5y","reached_registration"]).join(
            debut, on="owner_name", how="left"
        ).filter(pl.col("filing_date") == pl.col("debut_date"))
        j = s.join(o, on="serial_number", how="inner").with_columns(
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"))
        if j.height:
            parts.append(j.select("dkl","survived_5y","reached_registration","n_terms","year"))
        del s, o, j
        gc.collect()
    df = pl.concat(parts)
    print(f"  {df.height:,} debut filings", flush=True)
    df.write_csv(OUT / "debut_harvest_fit.csv")

    pdf = df.to_pandas()
    pdf["survived"] = pdf["survived_5y"].astype(int)

    # Full-panel correlation
    r_full = pdf[["dkl","survived"]].corr(method="pearson").iloc[0,1]
    rs_full = pdf[["dkl","survived"]].corr(method="spearman").iloc[0,1]

    # Negative side
    neg = pdf[pdf["dkl"] < 0].copy()
    neg["abs_dkl"] = -neg["dkl"]
    r_neg = neg[["abs_dkl","survived"]].corr(method="pearson").iloc[0,1]
    rs_neg = neg[["abs_dkl","survived"]].corr(method="spearman").iloc[0,1]

    # Positive side
    pos = pdf[pdf["dkl"] > 0].copy()
    r_pos = pos[["dkl","survived"]].corr(method="pearson").iloc[0,1]
    rs_pos = pos[["dkl","survived"]].corr(method="spearman").iloc[0,1]

    # Bin by |dkl| on each side and fit exponential
    def bin_fit(side_df, side_label):
        n_bins = 20
        cuts = np.quantile(side_df["abs_dkl"], np.linspace(0,1,n_bins+1))
        bin_idx = np.clip(np.searchsorted(cuts[1:-1], side_df["abs_dkl"]), 0, n_bins-1)
        side_df = side_df.assign(b=bin_idx)
        g = side_df.groupby("b").agg(x=("abs_dkl","mean"),
                                     y=("survived","mean"),
                                     n=("survived","size")).reset_index()
        x = g["x"].to_numpy(); y = g["y"].to_numpy(); n_pts = g["n"].to_numpy()
        # Exponential: y = a + b*exp(-c*x), with a estimated from the rightmost point
        def expfn(xx, a, b, c):
            return a + b * np.exp(-c * xx)
        try:
            popt, pcov = curve_fit(expfn, x, y, p0=(y[-1], y[0]-y[-1], 5.0),
                                    bounds=([0, -1, 0], [1, 2, 100]), maxfev=4000)
            yhat = expfn(x, *popt)
            ss_res = np.sum((y - yhat)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        except Exception as e:
            popt = (float("nan"),)*3; r2 = float("nan")
            print(f"  {side_label} fit failed: {e}")
        return g, popt, r2

    neg["abs_dkl"] = -neg["dkl"]
    g_neg, popt_neg, r2_neg = bin_fit(neg, "negative side")
    pos_for_fit = pos.copy(); pos_for_fit["abs_dkl"] = pos_for_fit["dkl"]
    g_pos, popt_pos, r2_pos = bin_fit(pos_for_fit, "positive side")

    summary = [
        f"Debut-only panel: n = {len(pdf):,} filings.",
        f"Overall corr(ΔKL, survived_5y):     Pearson r = {r_full:+.4f}  Spearman = {rs_full:+.4f}",
        f"",
        f"Negative-ΔKL side (n = {len(neg):,}):",
        f"  corr(|ΔKL|, survived):  Pearson r = {r_neg:+.4f}  Spearman = {rs_neg:+.4f}",
        f"  Exponential fit y = a + b*exp(-c*|ΔKL|):  a={popt_neg[0]:.4f}, b={popt_neg[1]:+.4f}, c={popt_neg[2]:.3f}",
        f"  R² of exponential fit on 20 binned points: {r2_neg:.4f}",
        f"",
        f"Positive-ΔKL side (n = {len(pos):,}):",
        f"  corr(ΔKL, survived):  Pearson r = {r_pos:+.4f}  Spearman = {rs_pos:+.4f}",
        f"  Exponential fit y = a + b*exp(-c*ΔKL):  a={popt_pos[0]:.4f}, b={popt_pos[1]:+.4f}, c={popt_pos[2]:.3f}",
        f"  R² of exponential fit on 20 binned points: {r2_pos:.4f}",
    ]
    text = "\n".join(summary)
    (OUT / "debut_harvest_fit.txt").write_text(text)
    print("\n" + text)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot(g_neg["x"], g_neg["y"], "o", color="#cc4444", ms=6, label="binned mean")
    xx = np.linspace(g_neg["x"].min(), g_neg["x"].max(), 100)
    yy = popt_neg[0] + popt_neg[1]*np.exp(-popt_neg[2]*xx)
    ax.plot(xx, yy, "--", color="#cc4444", lw=1.5,
            label=f"exp fit  $R^2 = {r2_neg:.3f}$")
    ax.set_xlabel(r"$|\Delta KL|$ (negative side, nats)")
    ax.set_ylabel("5y survival rate")
    ax.set_title(r"Debut filings, harvest tail ($\Delta KL < 0$)" + "\n" +
                 f"corr$(|\\Delta KL|, \\text{{surv}})$ = {r_neg:+.3f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(g_pos["x"], g_pos["y"], "o", color="#229922", ms=6, label="binned mean")
    xx = np.linspace(g_pos["x"].min(), g_pos["x"].max(), 100)
    yy = popt_pos[0] + popt_pos[1]*np.exp(-popt_pos[2]*xx)
    ax.plot(xx, yy, "--", color="#229922", lw=1.5,
            label=f"exp fit  $R^2 = {r2_pos:.3f}$")
    ax.set_xlabel(r"$\Delta KL$ (positive side, nats)")
    ax.set_ylabel("5y survival rate")
    ax.set_title(r"Debut filings, innovation tail ($\Delta KL > 0$)" + "\n" +
                 f"corr$(\\Delta KL, \\text{{surv}})$ = {r_pos:+.3f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "debut_harvest_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote .csv .png .txt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
