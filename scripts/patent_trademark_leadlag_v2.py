"""Lead-lag v2: reads the saved firm × year panel and runs the
correlation + FE regression on the patent-active subset only
(firms with ≥1 patent across any year, application-date basis).

The v1 ran on the full 8M-row panel which was dominated by
trademark-only firms whose patent count is identically 0, attenuating
the correlation. v2 restricts to the 54,323 firms with any patent
activity, which is the population the pipeline hypothesis is about.

Also adds significance tests on each lag's correlation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats as scstats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "results"


def main() -> int:
    print("[load] saved panel …", flush=True)
    panel = pl.read_parquet(OUT / "leadlag_firm_year.parquet")
    print(f"      total rows: {panel.height:,}")

    # Identify patent-active firms (any year, application-date)
    active = panel.group_by("norm").agg(pl.col("n_patents_application").sum().alias("tot_pat"))
    active_norms = active.filter(pl.col("tot_pat") > 0).select("norm")
    print(f"      patent-active firms: {active_norms.height:,}")

    p = panel.join(active_norms, on="norm", how="inner")
    print(f"      panel rows on patent-active firms: {p.height:,}")
    pdf = p.to_pandas().sort_values(["norm","year"])

    # ----- T2: lead-lag correlations on the patent-active subset
    print("\n=== T2 (patent-active firms only): lead-lag corr(patents(t), TM-ΔKL(t+k)) ===")
    correlations = []
    for k in range(-5, 6):
        shifted = pdf[["norm","year","mean_dkl","n_positive_dkl_filings","n_filings_clean"]].copy()
        shifted["year_orig"] = shifted["year"] - k
        m = pdf[["norm","year","n_patents_grant","n_patents_application"]].merge(
            shifted, left_on=["norm","year"], right_on=["norm","year_orig"], how="inner",
            suffixes=("","_shifted"))
        # Require both sides observed at the relevant lag
        m = m.dropna(subset=["mean_dkl"])
        if len(m) < 100:
            continue
        # Restrict the correlation to rows where the firm actually filed a TM that year
        # (otherwise mean_dkl is null which dropna handled).
        n = len(m)
        r_app = m["n_patents_application"].corr(m["mean_dkl"])
        r_grant = m["n_patents_grant"].corr(m["mean_dkl"])
        r_app_pos = m["n_patents_application"].corr(m["n_positive_dkl_filings"])
        r_grant_pos = m["n_patents_grant"].corr(m["n_positive_dkl_filings"])
        # CI via Fisher z (approximate)
        def ci(r, n):
            if n < 4 or not np.isfinite(r): return (np.nan, np.nan)
            z = np.arctanh(r); se = 1/np.sqrt(n-3)
            lo, hi = np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)
            return lo, hi
        correlations.append({
            "k": k, "n": n,
            "r_dkl_grant": r_grant, "r_dkl_app": r_app,
            "r_dkl_app_lo": ci(r_app, n)[0], "r_dkl_app_hi": ci(r_app, n)[1],
            "r_pos_grant": r_grant_pos, "r_pos_app": r_app_pos,
            "r_pos_app_lo": ci(r_app_pos, n)[0], "r_pos_app_hi": ci(r_app_pos, n)[1],
        })
    cor = pd.DataFrame(correlations)
    cor.to_csv(OUT / "leadlag_correlations_v2.csv", index=False)
    print(cor[["k","n","r_dkl_app","r_dkl_app_lo","r_dkl_app_hi",
                "r_pos_app","r_pos_app_lo","r_pos_app_hi"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))

    # Peak detection
    peak_dkl_app = cor.loc[cor["r_dkl_app"].idxmax()]
    peak_pos_app = cor.loc[cor["r_pos_app"].idxmax()]
    print(f"\n  PEAK r_dkl_app at k={int(peak_dkl_app['k'])}: r={peak_dkl_app['r_dkl_app']:+.4f}  n={int(peak_dkl_app['n']):,}")
    print(f"  PEAK r_pos_app at k={int(peak_pos_app['k'])}: r={peak_pos_app['r_pos_app']:+.4f}  n={int(peak_pos_app['n']):,}")

    # ----- T4: within-firm FE regression with patent-app lags 0..3
    print("\n=== T4: Within-firm FE regression, application-date lags 0..3 ===")
    work = pdf[["norm","year","mean_dkl","n_patents_application"]].copy()
    for lag in [0, 1, 2, 3]:
        s = work[["norm","year","n_patents_application"]].copy()
        s["year"] = s["year"] + lag
        s = s.rename(columns={"n_patents_application": f"pat_lag{lag}"})
        work = work.merge(s[["norm","year",f"pat_lag{lag}"]], on=["norm","year"], how="left")
    work = work.dropna(subset=["mean_dkl"])
    cnt = work.groupby("norm")["mean_dkl"].count()
    keep = cnt[cnt >= 5].index
    work = work[work["norm"].isin(keep)].copy()
    for lag in [0,1,2,3]:
        work[f"pat_lag{lag}"] = work[f"pat_lag{lag}"].fillna(0)
    # Demean within firm
    fmean = work.groupby("norm")[["mean_dkl","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]].transform("mean")
    w = work[["mean_dkl","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]] - fmean
    w = w.dropna()
    X = sm.add_constant(w[["pat_lag0","pat_lag1","pat_lag2","pat_lag3"]])
    y = w["mean_dkl"]
    mod = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": work.loc[w.index, "norm"]})
    print(mod.summary().tables[1])
    print(f"  N firms = {work['norm'].nunique():,}, N firm-years = {len(w):,}")

    # ----- T5: focus on n_positive_dkl_filings outcome (the cleanest "new vocabulary" signal)
    print("\n=== T5: Within-firm FE regression, outcome = n_positive_dkl_filings ===")
    work2 = pdf[["norm","year","n_positive_dkl_filings","n_patents_application","n_filings_clean"]].copy()
    work2 = work2[work2["n_filings_clean"] >= 1]  # only years the firm trademarked
    for lag in [0,1,2,3]:
        s = work2[["norm","year","n_patents_application"]].copy()
        s["year"] = s["year"] + lag
        s = s.rename(columns={"n_patents_application": f"pat_lag{lag}"})
        work2 = work2.merge(s[["norm","year",f"pat_lag{lag}"]], on=["norm","year"], how="left")
    cnt = work2.groupby("norm")["n_positive_dkl_filings"].count()
    keep = cnt[cnt >= 5].index
    work2 = work2[work2["norm"].isin(keep)].copy()
    for lag in [0,1,2,3]:
        work2[f"pat_lag{lag}"] = work2[f"pat_lag{lag}"].fillna(0)
    fmean2 = work2.groupby("norm")[["n_positive_dkl_filings","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]].transform("mean")
    w2 = work2[["n_positive_dkl_filings","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]] - fmean2
    w2 = w2.dropna()
    X2 = sm.add_constant(w2[["pat_lag0","pat_lag1","pat_lag2","pat_lag3"]])
    y2 = w2["n_positive_dkl_filings"]
    mod2 = sm.OLS(y2, X2).fit(cov_type="cluster", cov_kwds={"groups": work2.loc[w2.index, "norm"]})
    print(mod2.summary().tables[1])
    print(f"  N firms = {work2['norm'].nunique():,}, N firm-years = {len(w2):,}")

    # ----- Summary figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.errorbar(cor["k"], cor["r_dkl_app"],
                 yerr=[cor["r_dkl_app"] - cor["r_dkl_app_lo"], cor["r_dkl_app_hi"] - cor["r_dkl_app"]],
                 fmt="-s", color="#2b6cb0", lw=2, ms=7, capsize=4, label="application date (clean)")
    ax.errorbar(cor["k"], cor["r_dkl_grant"],
                 yerr=None, fmt="-o", color="#cc4444", lw=1.5, ms=5, alpha=0.6,
                 label="grant date (robustness)")
    ax.axvline(0, color="#444", ls=":", lw=0.8); ax.axhline(0, color="#444", ls="--", lw=0.8)
    ax.set_xlabel("Lag k (patents at t, mean ΔKL at t+k)")
    ax.set_ylabel("Pearson correlation, patent-active firms only")
    ax.set_title("T2: patents → mean ΔKL")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.errorbar(cor["k"], cor["r_pos_app"],
                 yerr=[cor["r_pos_app"] - cor["r_pos_app_lo"], cor["r_pos_app_hi"] - cor["r_pos_app"]],
                 fmt="-s", color="#2b6cb0", lw=2, ms=7, capsize=4, label="application date (clean)")
    ax.errorbar(cor["k"], cor["r_pos_grant"],
                 yerr=None, fmt="-o", color="#cc4444", lw=1.5, ms=5, alpha=0.6,
                 label="grant date (robustness)")
    ax.axvline(0, color="#444", ls=":", lw=0.8); ax.axhline(0, color="#444", ls="--", lw=0.8)
    ax.set_xlabel("Lag k (patents at t, count of positive-ΔKL TM at t+k)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("T3: patents → count of positive-ΔKL trademarks")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.suptitle("Patents → trademark ΔKL lead-lag (patent-active firms only, n="
                 f"{active_norms.height:,} firms)", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "leadlag_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- Headline text
    text = []
    text.append("Patent → trademark lead-lag (patent-active firms only)")
    text.append("=" * 70)
    text.append(f"Population: {active_norms.height:,} firms with at least 1 patent in any year.")
    text.append("")
    text.append("T2: corr(patents(t), TM ΔKL(t+k))")
    text.append(f"  {'k':>3}  {'n':>9}  {'r_dkl_grant':>12}  {'r_dkl_app':>12}  {'r_pos_grant':>12}  {'r_pos_app':>12}")
    for _, r in cor.iterrows():
        text.append(f"  {int(r['k']):>3}  {int(r['n']):>9,}  {r['r_dkl_grant']:>+12.4f}  {r['r_dkl_app']:>+12.4f}  {r['r_pos_grant']:>+12.4f}  {r['r_pos_app']:>+12.4f}")
    text.append("")
    text.append(f"PEAK r_dkl_app at k={int(peak_dkl_app['k'])}: r={peak_dkl_app['r_dkl_app']:+.4f}")
    text.append(f"PEAK r_pos_app at k={int(peak_pos_app['k'])}: r={peak_pos_app['r_pos_app']:+.4f}")
    text.append("")
    text.append("T4: Within-firm FE regression, mean_dkl ~ pat_lag0..3 (clustered SE)")
    text.append(str(mod.summary().tables[1]))
    text.append(f"  N firms = {work['norm'].nunique():,}, N firm-years = {len(w):,}")
    text.append("")
    text.append("T5: Within-firm FE regression, n_positive_dkl_filings ~ pat_lag0..3")
    text.append(str(mod2.summary().tables[1]))
    text.append(f"  N firms = {work2['norm'].nunique():,}, N firm-years = {len(w2):,}")
    (OUT / "leadlag_v2.txt").write_text("\n".join(text))
    print()
    print("\n".join(text[-25:]))
    print(f"\n[done] {OUT/'leadlag_v2.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
