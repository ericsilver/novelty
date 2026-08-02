"""Line plot of 5y survival rate by prospective KL, retrospective KL,
and ΔKL deciles.  Pooled across all 45 classes, CLEAN H=2y panel.

Tests the inverted-U hypothesis: do firms succeed at BOTH extremes of
prospective/retrospective KL ('catching new business models' vs
'harvesting old ones'), with a squeeze in the middle?

Outputs:
  paper/results/survival_by_kl_lines.png
  paper/results/survival_by_kl_lines.csv
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


def pool_panel() -> pl.DataFrame:
    """Stream all classes; return per-filing rows with KL + passed_5y.
    Restricts to 5y-observable cohorts (year ≤ 2021) and CLEAN cut."""
    parts = []
    for cls in class_list():
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
            & pl.col("year").is_between(1985, 2021)
        )
        o = pl.read_parquet(
            op, columns=["serial_number", "reached_registration", "currently_live"],
        )
        j = s.join(o, on="serial_number", how="inner").with_columns(
            (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y"),
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        ).select("year", "kl_vs_past", "kl_vs_future", "dkl",
                 "n_terms", "passed_5y")
        if j.height:
            parts.append(j)
        del s, o, j
        gc.collect()
    return pl.concat(parts)


def quantile_curve(df: pl.DataFrame, var: str, n_bins: int = 20) -> pl.DataFrame:
    """Bin by quantile of var, report mean survival + 95% CI."""
    qs = np.linspace(0, 1, n_bins + 1)
    cuts = df[var].quantile(qs.tolist()) if hasattr(df[var], "quantile") else None
    # Polars' quantile returns scalar — use numpy
    arr = df[var].to_numpy()
    cuts = np.quantile(arr, qs)
    bin_idx = np.clip(np.searchsorted(cuts[1:-1], arr), 0, n_bins - 1)
    df = df.with_columns(pl.Series("bin", bin_idx))
    g = df.group_by("bin").agg([
        pl.len().alias("n"),
        pl.col("passed_5y").cast(pl.Float64).mean().alias("rate"),
        pl.col(var).mean().alias(f"{var}_mid"),
    ]).sort("bin")
    # Wilson 95% CI
    p = g["rate"].to_numpy()
    n = g["n"].to_numpy().astype(np.float64)
    z = 1.96
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    g = g.with_columns(
        pl.Series("lo", centre - half),
        pl.Series("hi", centre + half),
    )
    return g


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[pool] loading panel …", flush=True)
    df = pool_panel()
    print(f"       {df.height:,} clean H=2y filings, 1985-2021", flush=True)

    curves = {}
    for var, label in [("dkl", "ΔKL"), ("kl_vs_past", "Prospective KL (vs. past)"),
                       ("kl_vs_future", "Retrospective KL (vs. future)")]:
        curves[label] = quantile_curve(df, var, n_bins=20)
        print(f"  {label}: 20 quantile bins", flush=True)

    # Save combined CSV
    rows = []
    for label, g in curves.items():
        for r in g.iter_rows(named=True):
            rows.append({"axis": label, **r})
    pl.from_dicts(rows).write_csv(OUT / "survival_by_kl_lines.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = {"ΔKL": "#2b6cb0", "Prospective KL": "#cc4444", "Retrospective KL": "#229922"}
    for label, g in curves.items():
        pdf = g.to_pandas()
        xkey = next(c for c in pdf.columns if c.endswith("_mid"))
        ax.plot(pdf[xkey], pdf["rate"], "-o", color=colors[label], lw=2, ms=5, label=label)
        ax.fill_between(pdf[xkey], pdf["lo"], pdf["hi"], color=colors[label], alpha=0.18)
    ax.axhline(df["passed_5y"].cast(pl.Float64).mean(), ls="--", color="#444", lw=1,
               label=f"corpus mean = {df['passed_5y'].cast(pl.Float64).mean():.3f}")
    ax.set_xlabel("KL value (nats)  —  axes share scale")
    ax.set_ylabel("5y survival rate  (reached registration AND currently live)")
    ax.set_title(f"5y survival vs prospective KL, retrospective KL, ΔKL\n"
                 f"pooled across 45 classes, H=2y CLEAN cut, n={df.height:,}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    out_png = OUT / "survival_by_kl_lines.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
