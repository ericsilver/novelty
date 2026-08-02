"""Per-filing isolation of prospective vs retrospective KL effects.

Same idea as scripts/prospective_vs_retrospective.py but at the FILING
level rather than firm-mean.  Per-filing the two components have more
independent variation, so any underlying divergence (e.g.\ prospective
hurts registration but retrospective helps it) should show up more
clearly here than in the firm-aggregated view.

Outcomes available per-filing:
  - reached_registration  (filing cleared examination)
  - passed_5y             (registered AND currently live)

Going-public and margin outcomes are firm-level only and remain in
prospective_vs_retrospective.py.

Outputs (paper/results/):
  per_filing_pros_vs_retr_software.png
  per_filing_pros_vs_retr_beverages.png
  per_filing_pros_vs_retr_all.png
  per_filing_pros_vs_retr.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
    & pl.col("kl_vs_past").is_finite()
    & pl.col("kl_vs_future").is_finite()
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
        ).select("kl_vs_past", "kl_vs_future",
                 "reached_registration", "passed_5y", "n_terms")
        parts.append(df)
    return pl.concat(parts)


def bin_curve(arr_x, arr_y, n_bins=15, q_lo=0.02, q_hi=0.98, min_n=200):
    lo, hi = np.quantile(arr_x, [q_lo, q_hi])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers, ys, lows, highs, ns = [], [], [], [], []
    for i in range(n_bins):
        m = (arr_x >= edges[i]) & (arr_x < edges[i + 1])
        n = int(m.sum())
        if n < min_n: continue
        y = float(arr_y[m].mean())
        # Wilson 95% CI for binary
        from math import sqrt
        z = 1.96
        denom = 1 + z * z / n
        center = (y + z * z / (2 * n)) / denom
        half = (z * sqrt(y * (1 - y) / n + z * z / (4 * n * n))) / denom
        ci_lo = max(0, center - half); ci_hi = min(1, center + half)
        centers.append((edges[i] + edges[i + 1]) / 2)
        ys.append(y); lows.append(ci_lo); highs.append(ci_hi); ns.append(n)
    return np.array(centers), np.array(ys), np.array(lows), np.array(highs), np.array(ns)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2023)
    args = ap.parse_args()

    sw_classes = ["009"]
    bv_classes = ["032"]
    all_classes = sorted(p.stem.replace("outcomes_class", "")
                         for p in PROC.glob("outcomes_class*.parquet")
                         if p.stem.replace("outcomes_class", "").isdigit())

    scopes = [
        ("software (Class 9)", sw_classes, "software"),
        ("beverages (Class 32)", bv_classes, "beverages"),
        ("all 45 classes pooled", all_classes, "all"),
    ]

    summary_rows = []
    for label, cls_list, fname in scopes:
        print(f"[load] {label}…", flush=True)
        f = load_filings(cls_list, args.year_min, args.year_max)
        print(f"  {f.height:,} filings", flush=True)

        x_pros = f["kl_vs_past"].to_numpy()
        x_retr = f["kl_vs_future"].to_numpy()
        y_reg = f["reached_registration"].to_numpy().astype(float)
        y_5y = f["passed_5y"].to_numpy().astype(float)

        # Per-filing correlation between components — context
        rho = float(np.corrcoef(x_pros, x_retr)[0, 1])
        print(f"  corr(pros, retr) per-filing: {rho:+.4f}", flush=True)

        outcomes = [
            ("reached_registration", y_reg, "Reached registration"),
            ("passed_5y", y_5y, "Passed 5y (reg & still live)"),
        ]
        fig, axes = plt.subplots(len(outcomes), 2, figsize=(11, 5 * len(outcomes)),
                                 squeeze=False)
        for i, (oname, yy, ylabel) in enumerate(outcomes):
            for j, (xx, xname) in enumerate([(x_pros, "Prospective KL (per filing)"),
                                              (x_retr, "Retrospective KL (per filing)")]):
                ax = axes[i, j]
                xc, yc, lo, hi, ns = bin_curve(xx, yy, n_bins=15)
                if len(xc) == 0:
                    ax.text(0.5, 0.5, "no bins meet min_n", transform=ax.transAxes); continue
                ax.plot(xc, yc, "o-", color="#2b6cb0", lw=2, ms=5, label="bin mean")
                ax.fill_between(xc, lo, hi, color="#2b6cb0", alpha=0.18,
                                label="95% Wilson CI")
                if len(xc) >= 3:
                    lin = np.polyfit(xc, yc, 1)
                    ax.plot(xc, np.polyval(lin, xc), "--", color="#cc4444", alpha=0.7,
                            label=f"linear (slope={lin[0]:+.4f})")
                    # Find argmax to flag inverted-U
                    argmax = int(np.argmax(yc))
                    if 1 <= argmax < len(xc) - 1:
                        ax.axvline(xc[argmax], color="green", linestyle=":", alpha=0.5,
                                   label=f"peak @ {xc[argmax]:.2f}")
                ax.set_xlabel(xname)
                ax.set_ylabel(f"P({ylabel})", fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.98, f"min n/bin={ns.min():,}\nmax n/bin={ns.max():,}",
                        transform=ax.transAxes, va="top", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
                if i == 0 and j == 0:
                    ax.legend(loc="lower right", fontsize=8)
                argmax_x = float(xc[int(np.argmax(yc))]) if len(xc) > 0 else float("nan")
                summary_rows.append({
                    "scope": label, "outcome": oname, "x": xname.split()[0],
                    "slope": float(lin[0]) if len(xc) >= 3 else float("nan"),
                    "y_at_low": float(yc[0]), "y_at_high": float(yc[-1]),
                    "y_max": float(yc.max()), "x_at_max": argmax_x,
                    "n_bins": len(xc), "rho_pros_retr": rho,
                })
        fig.suptitle(f"Per-filing outcomes vs prospective and retrospective KL — {label}\n"
                     f"per-filing $\\rho$(pros, retr) = {rho:+.3f}",
                     fontsize=12, y=1.0)
        fig.tight_layout()
        out_png = RESULTS / f"per_filing_pros_vs_retr_{fname}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [done] {out_png}", flush=True)

    out_txt = RESULTS / "per_filing_pros_vs_retr.txt"
    with out_txt.open("w") as f:
        f.write("Per-filing isolation of prospective and retrospective KL effects.\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'scope':<28s}{'outcome':<22s}{'x':<7s}"
                f"{'slope':>10s}{'y_lo':>9s}{'y_hi':>9s}"
                f"{'y_max':>9s}{'x@max':>9s}{'rho':>9s}\n")
        for r in summary_rows:
            f.write(f"{r['scope'][:26]:<28s}{r['outcome']:<22s}{r['x']:<7s}"
                    f"{r['slope']:>+10.4f}{r['y_at_low']:>9.4f}"
                    f"{r['y_at_high']:>9.4f}{r['y_max']:>9.4f}"
                    f"{r['x_at_max']:>9.2f}{r['rho_pros_retr']:>+9.3f}\n")
        f.write("\nInterpretation:\n")
        f.write("  Per-filing rho(pros, retr) tells how independently the components vary.\n")
        f.write("  If much lower than firm-mean rho, per-filing analysis isolates them better.\n")
        f.write("  Compare slope and x@max columns side-by-side for the same outcome to see\n")
        f.write("  whether prospective and retrospective KL act differently.\n")
    print(f"[done] wrote {out_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
