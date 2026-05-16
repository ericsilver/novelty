"""Isolate the marginal effects of prospective and retrospective KL.

Pull each outcome — registration / 5y-survival, P(ever-public),
gross-margin (on matched panel) — and plot it against prospective KL
alone and retrospective KL alone, separately, with the other component
free.  Two running examples: Class 9 (Software & Electronics) and
Class 32 (Beverages, i.e. soft drinks).  Pooled all-class line shown
for context.

The point is to give the reader a sense of the \emph{shape} (linear?
exponential? monotone?) of each component's relationship with each
outcome before introducing the difference $\\Delta KL = KL_{pros} -
KL_{retr}$ as a summary scalar.

Outputs (paper/results/):
  pros_vs_retr_curves_software.png        (Class 9)
  pros_vs_retr_curves_beverages.png       (Class 32)
  pros_vs_retr_curves_all.png             (pooled)
  pros_vs_retr_curves.txt
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

CLEAN = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
    & pl.col("prospective_kl").is_finite()
    & pl.col("retrospective_kl").is_finite()
    & pl.col("owner_name").is_not_null()
)


def load_filings(classes: list[str], year_min: int, year_max: int) -> pl.DataFrame:
    parts = []
    for cls in classes:
        path = PROC / f"outcomes_class{cls}.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path).filter(
            CLEAN & (pl.col("year") >= year_min) & (pl.col("year") <= year_max)
        ).with_columns(
            (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y"),
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
            pl.lit(cls).alias("nice_class"),
        ).select("owner_name", "nice_class", "year", "prospective_kl",
                 "retrospective_kl", "dkl", "reached_registration",
                 "passed_5y", "n_terms")
        parts.append(df)
    return pl.concat(parts)


def firm_panel(filings: pl.DataFrame, min_filings: int = 3) -> pl.DataFrame:
    return (filings.group_by("owner_name")
            .agg(pl.col("prospective_kl").mean().alias("mean_pros"),
                 pl.col("retrospective_kl").mean().alias("mean_retr"),
                 pl.col("dkl").mean().alias("mean_dkl"),
                 pl.col("passed_5y").mean().alias("firm_passed5"),
                 pl.col("reached_registration").mean().alias("firm_reach"),
                 pl.len().alias("n_filings"))
            .filter(pl.col("n_filings") >= min_filings))


def bin_and_summarize(panel_pd, x_col, y_col, n_bins=10, q_lo=0.02, q_hi=0.98):
    """Return (bin_centers, mean_y, ci_lo, ci_hi, n)."""
    lo, hi = panel_pd[x_col].quantile([q_lo, q_hi])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers, ys, lows, highs, ns = [], [], [], [], []
    for i in range(n_bins):
        m = (panel_pd[x_col] >= edges[i]) & (panel_pd[x_col] < edges[i + 1])
        sub = panel_pd[m]
        if len(sub) < 30:
            continue
        n = len(sub)
        y = sub[y_col].mean()
        # bootstrap CI of mean
        rng = np.random.default_rng(42 + i)
        boots = [sub[y_col].sample(n, replace=True, random_state=rng.integers(2**31)).mean()
                 for _ in range(200)]
        lo_b, hi_b = np.percentile(boots, [2.5, 97.5])
        centers.append((edges[i] + edges[i + 1]) / 2)
        ys.append(y); lows.append(lo_b); highs.append(hi_b); ns.append(n)
    return np.array(centers), np.array(ys), np.array(lows), np.array(highs), np.array(ns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2023)
    args = ap.parse_args()

    sw_classes = ["009"]
    bv_classes = ["032"]
    all_classes = sorted(p.stem.replace("outcomes_class", "")
                         for p in PROC.glob("outcomes_class*.parquet")
                         if p.stem.replace("outcomes_class", "").isdigit())

    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik").unique(subset=["owner_name"])

    print(f"[load] software (cls 009)…", flush=True)
    sw_filings = load_filings(sw_classes, args.year_min, args.year_max)
    print(f"  {sw_filings.height:,} filings", flush=True)
    sw_firm = firm_panel(sw_filings)
    sw_firm = sw_firm.join(cw, on="owner_name", how="left").with_columns(
        pl.col("cik").is_not_null().alias("ever_public")).to_pandas()
    print(f"  {len(sw_firm):,} firms (>=3 clean filings)", flush=True)

    print(f"[load] beverages (cls 032)…", flush=True)
    bv_filings = load_filings(bv_classes, args.year_min, args.year_max)
    print(f"  {bv_filings.height:,} filings", flush=True)
    bv_firm = firm_panel(bv_filings)
    bv_firm = bv_firm.join(cw, on="owner_name", how="left").with_columns(
        pl.col("cik").is_not_null().alias("ever_public")).to_pandas()
    print(f"  {len(bv_firm):,} firms (>=3 clean filings)", flush=True)

    print(f"[load] all 45 classes…", flush=True)
    all_filings = load_filings(all_classes, args.year_min, args.year_max)
    all_firm = firm_panel(all_filings)
    all_firm = all_firm.join(cw, on="owner_name", how="left").with_columns(
        pl.col("cik").is_not_null().alias("ever_public")).to_pandas()
    print(f"  {len(all_filings):,} filings, {len(all_firm):,} firms", flush=True)

    # ---- Figure: 2x3 panels per scope; one figure per scope ----
    scopes = [("software (Class 9)", sw_firm, "software"),
              ("beverages (Class 32)", bv_firm, "beverages"),
              ("all 45 classes pooled", all_firm, "all")]

    outcomes = [
        ("firm_passed5", "Mean per-firm 5y-survival\n(P(reached registration AND still live))",
         "share of firms' filings"),
        ("ever_public", "P(ever in SEC EDGAR)",
         "share of firms"),
    ]

    summary_rows = []

    for scope_label, panel, fname in scopes:
        # Filter to firms with at least 10 filings when computing P(ever_public)
        # so the crosswalk fuzzy-pass eligibility applies
        elig = panel[panel["n_filings"] >= 10].copy()

        fig, axes = plt.subplots(len(outcomes), 2, figsize=(11, 5 * len(outcomes)),
                                 squeeze=False)
        for i, (y_col, y_label, ann) in enumerate(outcomes):
            for j, (x_col, x_label) in enumerate([
                ("mean_pros", r"Prospective KL  (firm-mean)"),
                ("mean_retr", r"Retrospective KL  (firm-mean)"),
            ]):
                ax = axes[i, j]
                src = elig if y_col == "ever_public" else panel
                if len(src) < 100:
                    ax.text(0.5, 0.5, "insufficient data",
                            ha="center", va="center", transform=ax.transAxes)
                    continue
                xc, yc, lo, hi, ns = bin_and_summarize(src, x_col, y_col, n_bins=12)
                if len(xc) == 0:
                    continue
                ax.plot(xc, yc, "o-", color="#2b6cb0", linewidth=2,
                        markersize=5, label="bin mean")
                ax.fill_between(xc, lo, hi, color="#2b6cb0", alpha=0.18,
                                label="95% bootstrap CI")
                # Linear and log fits
                if len(xc) >= 3:
                    # Linear fit
                    lin = np.polyfit(xc, yc, 1)
                    ax.plot(xc, np.polyval(lin, xc), "--", color="#cc4444",
                            alpha=0.7, label=f"linear fit (slope={lin[0]:+.3f})")
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label, fontsize=9)
                ax.grid(True, alpha=0.3)
                # annotate min/max bin
                ax.text(0.02, 0.95, f"min n/bin={ns.min():,}; max n/bin={ns.max():,}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
                if i == 0 and j == 0:
                    ax.legend(loc="lower left", fontsize=8)
                summary_rows.append({
                    "scope": scope_label, "outcome": y_col, "x": x_col,
                    "slope": float(lin[0]) if len(xc) >= 3 else float("nan"),
                    "n_bins": len(xc),
                    "lo_y": float(yc[0]), "hi_y": float(yc[-1]),
                })
        fig.suptitle(
            f"Outcomes vs prospective and retrospective KL components, "
            f"{scope_label}", fontsize=12, y=1.0,
        )
        fig.tight_layout()
        out_png = RESULTS / f"pros_vs_retr_curves_{fname}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [done] {out_png}", flush=True)

    # ---- Numerical summary ----
    out_txt = RESULTS / "pros_vs_retr_curves.txt"
    with out_txt.open("w") as f:
        f.write("Isolated marginal effects of prospective and retrospective KL.\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"{'scope':<30s}{'outcome':<15s}{'x_axis':<14s}"
                f"{'slope':>10s}{'lo_y':>10s}{'hi_y':>10s}{'n_bins':>8s}\n")
        for r in summary_rows:
            f.write(f"{r['scope'][:28]:<30s}{r['outcome']:<15s}{r['x']:<14s}"
                    f"{r['slope']:>+10.4f}{r['lo_y']:>10.4f}"
                    f"{r['hi_y']:>10.4f}{r['n_bins']:>8d}\n")
        f.write("\nInterpretation:\n")
        f.write("  slope > 0 => outcome rises with the KL component\n")
        f.write("  slope < 0 => outcome falls with the KL component\n")
        f.write("  Comparison of the two columns isolates which component\n")
        f.write("  is doing the work for each outcome.\n")
    print(f"[done] wrote {out_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
