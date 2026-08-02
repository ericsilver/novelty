"""P(ever-public) as a function of ΔKL — a coarser financial-success check.

The financial regressions in scripts/financials_regression.py operate on
the matched (USPTO ↔ SEC) panel and ask: among firms successful enough to
file 10-Ks, do novel ones have higher margins?

This script asks the prior question: among trademark-filing firms,
does novelty predict the binary outcome of *ever appearing in SEC EDGAR*?
Any company filing income statements is "successful" by a coarser bar
than gross-margin level — it has gone public, registered S-1 securities,
or otherwise reached a reporting threshold.

Sample selection: the SEC crosswalk's fuzzy pass requires owners with
>=10 USPTO filings, so single-filing firms are CENSORED on the
public-detection side.  We restrict to firms with >=10 clean filings
(crosswalk-eligible) and additionally stratify by filing-volume bucket so
that "P(public) by ΔKL" within each volume band is on a level playing field.

Outputs:
  paper/results/public_listing_by_dkl.png
  paper/results/public_listing_by_dkl.txt
  paper/results/public_listing_by_dkl.tex   (table for the paper)
  paper/results/public_listing_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

CLEAN_BASE = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
    & pl.col("kl_vs_past").is_finite()
    & pl.col("kl_vs_future").is_finite()
    & pl.col("owner_name").is_not_null()
)


def load_firm_panel(year_min: int, year_max: int) -> pl.DataFrame:
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        df = pl.read_parquet(path).filter(
            CLEAN_BASE
            & (pl.col("year") >= year_min)
            & (pl.col("year") <= year_max)
        ).with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        ).select("owner_name", "kl_vs_past", "kl_vs_future", "dkl")
        parts.append(df)
    full = pl.concat(parts)
    firm = full.group_by("owner_name").agg(
        pl.len().alias("n_filings"),
        pl.col("kl_vs_past").mean().alias("mean_pros"),
        pl.col("kl_vs_future").mean().alias("mean_retr"),
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.col("dkl").max().alias("max_dkl"),
    )
    return firm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2023)
    ap.add_argument("--min-filings", type=int, default=10,
                    help="restrict to firms with >= this many clean filings "
                         "(crosswalk fuzzy-pass requires >=10)")
    args = ap.parse_args()

    print(f"[load] firm panel, years {args.year_min}-{args.year_max}…", flush=True)
    firm = load_firm_panel(args.year_min, args.year_max)
    print(f"  {firm.height:,} firms in clean window", flush=True)

    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik", "match_type"
    ).unique(subset=["owner_name"])
    print(f"  {cw.height:,} unique owner_names in SEC crosswalk", flush=True)

    panel = firm.join(cw, on="owner_name", how="left").with_columns(
        pl.col("cik").is_not_null().alias("ever_public"),
    )
    elig = panel.filter(pl.col("n_filings") >= args.min_filings)
    print(f"  {elig.height:,} firms with >= {args.min_filings} clean filings "
          f"(crosswalk-eligible)", flush=True)
    overall_rate = float(elig["ever_public"].mean())
    print(f"  base rate P(ever_public) = {overall_rate:.4f} "
          f"({int(elig['ever_public'].sum()):,} matched)", flush=True)

    # ---- P(ever_public) by mean ΔKL quintile (within crosswalk-eligible firms) ----
    elig_pd = elig.to_pandas()
    elig_pd["dkl_quintile"] = (
        elig_pd["mean_dkl"].rank(method="first", pct=True).mul(5).clip(upper=4.999).astype(int)
    )
    elig_pd["log_n_filings"] = np.log10(elig_pd["n_filings"])

    print("\n[main] P(ever_public) by mean-ΔKL quintile (>=10 filings only):")
    print(f"  {'quintile':<10s}{'n_firms':>10s}{'mean_dkl':>11s}"
          f"{'mean_n_fil':>13s}{'P(public)':>12s}{'95% CI':>20s}")
    rows_q = []
    for q in range(5):
        sub = elig_pd[elig_pd["dkl_quintile"] == q]
        n = len(sub)
        if n == 0:
            continue
        mn = float(sub["mean_dkl"].mean())
        nf = float(sub["n_filings"].mean())
        p = float(sub["ever_public"].mean())
        # Wilson 95% CI
        from math import sqrt
        z = 1.96
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        ci = (max(0, center - half), min(1, center + half))
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        rows_q.append({
            "quintile": q + 1, "n_firms": n, "mean_dkl": mn,
            "mean_n_filings": nf, "p_public": p,
            "ci_lo": ci[0], "ci_hi": ci[1],
        })
        print(f"  Q{q+1:<9d}{n:>10,}{mn:>11.4f}{nf:>13,.1f}{p:>12.4f}{ci_str:>20s}")

    # ---- Stratified by filing-count band ----
    print("\n[strata] P(ever_public) by ΔKL quintile, within filing-count band:")
    bands = [(10, 24), (25, 99), (100, 999), (1000, 1_000_000)]
    band_labels = ["10-24", "25-99", "100-999", ">=1k"]
    strat_rows = []
    for (lo, hi), lbl in zip(bands, band_labels):
        sub = elig_pd[(elig_pd["n_filings"] >= lo) & (elig_pd["n_filings"] <= hi)].copy()
        if len(sub) < 100:
            continue
        sub["dkl_quintile"] = (
            sub["mean_dkl"].rank(method="first", pct=True).mul(5).clip(upper=4.999).astype(int)
        )
        print(f"\n  band {lbl}  (n={len(sub):,}, base P(public)={sub['ever_public'].mean():.4f}):")
        print(f"    {'quintile':<10s}{'n_firms':>10s}{'mean_dkl':>11s}{'P(public)':>12s}")
        for q in range(5):
            ss = sub[sub["dkl_quintile"] == q]
            n = len(ss)
            if n == 0:
                continue
            mn = float(ss["mean_dkl"].mean())
            p = float(ss["ever_public"].mean())
            print(f"    Q{q+1:<9d}{n:>10,}{mn:>11.4f}{p:>12.4f}")
            strat_rows.append({
                "band": lbl, "quintile": q + 1, "n_firms": n,
                "mean_dkl": mn, "p_public": p,
            })

    # ---- Logistic regression: ever_public ~ ΔKL + log_n_filings ----
    import statsmodels.api as sm
    print("\n[logit] ever_public ~ z(mean_dkl) + log10(n_filings)")
    df = elig_pd[["ever_public", "mean_dkl", "log_n_filings"]].dropna().copy()
    df["z_dkl"] = (df["mean_dkl"] - df["mean_dkl"].mean()) / df["mean_dkl"].std()
    X = sm.add_constant(df[["z_dkl", "log_n_filings"]])
    y = df["ever_public"].astype(int)
    try:
        model = sm.Logit(y, X).fit(disp=False)
        coef = model.params.to_dict()
        se = model.bse.to_dict()
        pv = model.pvalues.to_dict()
        for k in ["const", "z_dkl", "log_n_filings"]:
            print(f"    {k:<18s}  beta={coef[k]:+.4f}  se={se[k]:.4f}  p={pv[k]:.3g}")
    except Exception as e:
        print(f"  logit failed: {e}")
        model = None

    # ---- Also a quadrant view: P(public) on (mean_pros, mean_retr) ----
    print("\n[quadrant] P(ever_public) by (mean_pros, mean_retr) quadrant "
          "split at panel medians (eligible firms):")
    pmed = float(elig_pd["mean_pros"].median())
    rmed = float(elig_pd["mean_retr"].median())
    quad_rows = []
    for pl_lo, pl_hi, p_label in [(0, pmed, "low_pros"), (pmed, np.inf, "high_pros")]:
        for rl_lo, rl_hi, r_label in [(0, rmed, "low_retr"), (rmed, np.inf, "high_retr")]:
            ss = elig_pd[(elig_pd["mean_pros"] >= pl_lo)
                         & (elig_pd["mean_pros"] < pl_hi)
                         & (elig_pd["mean_retr"] >= rl_lo)
                         & (elig_pd["mean_retr"] < rl_hi)]
            n = len(ss)
            p = float(ss["ever_public"].mean()) if n > 0 else float("nan")
            print(f"  {p_label} × {r_label:<10s}  n={n:>7,}  P(public)={p:.4f}")
            quad_rows.append({"pros": p_label, "retr": r_label,
                              "n_firms": n, "p_public": p})

    # ---------- Outputs ----------
    out_txt = RESULTS / "public_listing_by_dkl.txt"
    with out_txt.open("w") as f:
        f.write("P(ever-public) as a function of ΔKL\n")
        f.write(f"Window: {args.year_min}-{args.year_max}; min_filings={args.min_filings}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Crosswalk-eligible firms: {elig.height:,}\n")
        f.write(f"Base rate P(ever_public): {overall_rate:.4f}  "
                f"({int(elig['ever_public'].sum()):,} matched)\n\n")
        f.write("By mean-ΔKL quintile (eligible only):\n")
        f.write(f"  {'quintile':<10s}{'n_firms':>10s}{'mean_dkl':>11s}"
                f"{'mean_n_fil':>13s}{'P(public)':>12s}{'CI lo':>10s}{'CI hi':>10s}\n")
        for r in rows_q:
            f.write(f"  Q{r['quintile']:<9d}{r['n_firms']:>10,}"
                    f"{r['mean_dkl']:>11.4f}{r['mean_n_filings']:>13,.1f}"
                    f"{r['p_public']:>12.4f}{r['ci_lo']:>10.4f}{r['ci_hi']:>10.4f}\n")
        f.write("\nStratified by filing-count band:\n")
        for r in strat_rows:
            f.write(f"  band={r['band']:<8s}Q{r['quintile']}  n={r['n_firms']:>6,}  "
                    f"mean_dkl={r['mean_dkl']:>+8.4f}  P(public)={r['p_public']:.4f}\n")
        f.write("\nLogit ever_public ~ z(mean_dkl) + log10(n_filings):\n")
        if model is not None:
            for k in ["const", "z_dkl", "log_n_filings"]:
                f.write(f"  {k:<18s}beta={model.params[k]:+.4f}  "
                        f"se={model.bse[k]:.4f}  p={model.pvalues[k]:.3g}\n")
        f.write("\nQuadrant view (split at panel medians):\n")
        for r in quad_rows:
            f.write(f"  {r['pros']} × {r['retr']:<10s}  n={r['n_firms']:>7,}  "
                    f"P(public)={r['p_public']:.4f}\n")
    print(f"\n[done] wrote {out_txt}", flush=True)

    # ---- LaTeX table for the paper ----
    out_tex = RESULTS / "public_listing_by_dkl.tex"
    with out_tex.open("w") as f:
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("$\\overline{\\Delta KL}$ quintile & "
                "$n$ firms & mean $\\overline{\\Delta KL}$ & "
                "mean filings & $P(\\text{ever-public})$ \\\\\n")
        f.write("\\midrule\n")
        for r in rows_q:
            f.write(f"Q{r['quintile']} ({'least innov.' if r['quintile']==1 else 'most innov.' if r['quintile']==5 else ''})"
                    f" & {r['n_firms']:,} & ${r['mean_dkl']:+.3f}$ "
                    f"& {r['mean_n_filings']:,.0f} & "
                    f"${r['p_public']:.3f}$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"[done] wrote {out_tex}", flush=True)

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    qs = [r["quintile"] for r in rows_q]
    ps = [r["p_public"] for r in rows_q]
    yerr = np.array([[r["p_public"] - r["ci_lo"] for r in rows_q],
                     [r["ci_hi"] - r["p_public"] for r in rows_q]])
    ax.bar(qs, ps, yerr=yerr, capsize=4,
           color=["#cc4444", "#dd9966", "#888888", "#66aacc", "#2b6cb0"],
           edgecolor="black", linewidth=0.5)
    ax.axhline(overall_rate, color="black", linestyle="--", alpha=0.5,
               label=f"base rate {overall_rate:.3f}")
    ax.set_xticks(qs)
    ax.set_xticklabels([f"Q{q}" for q in qs])
    ax.set_xlabel("Mean ΔKL quintile (Q1=tired, Q5=innovative)")
    ax.set_ylabel("P(ever appears in SEC EDGAR)")
    ax.set_title("(a) Pooled (≥10 filings)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    band_data = {}
    for r in strat_rows:
        band_data.setdefault(r["band"], []).append((r["quintile"], r["p_public"]))
    colors = {"10-24": "#cc4444", "25-99": "#dd9966",
              "100-999": "#66aacc", ">=1k": "#2b6cb0"}
    for band, pts in band_data.items():
        pts = sorted(pts)
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                 color=colors.get(band, "gray"), label=f"{band} filings")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticklabels([f"Q{q}" for q in [1,2,3,4,5]])
    ax2.set_xlabel("Mean ΔKL quintile")
    ax2.set_ylabel("P(ever appears in SEC EDGAR)")
    ax2.set_title("(b) Stratified by filing-count band")
    ax2.legend(title="filings", loc="best")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_yscale("log")

    fig.suptitle("Going public as a function of trademark novelty "
                 f"(crosswalk-eligible firms, n={elig.height:,})", fontsize=11)
    fig.tight_layout()
    out_png = RESULTS / "public_listing_by_dkl.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_png}", flush=True)

    # ---- metrics json ----
    metrics = {
        "n_firms_eligible": elig.height,
        "base_p_public": overall_rate,
        "by_quintile": rows_q,
        "by_strata": strat_rows,
        "by_quadrant": quad_rows,
    }
    if model is not None:
        metrics["logit"] = {
            "z_dkl_beta": float(model.params["z_dkl"]),
            "z_dkl_se": float(model.bse["z_dkl"]),
            "z_dkl_p": float(model.pvalues["z_dkl"]),
            "log_n_filings_beta": float(model.params["log_n_filings"]),
            "log_n_filings_se": float(model.bse["log_n_filings"]),
            "log_n_filings_p": float(model.pvalues["log_n_filings"]),
        }
    out_json = RESULTS / "public_listing_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"[done] wrote {out_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
