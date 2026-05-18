"""Three-panel companion to survival_by_kl_lines.py.

For each KL axis (pros, retr, ΔKL) plot three outcomes side-by-side:

  Panel A  P(reached_registration)        — examiner approved the mark
  Panel B  P(survived_5y) [reached & live] — 5y survival including post-reg deaths
  Panel C  P(owner_name in SEC EDGAR)     — owner ever publicly traded

Robustness on abandoned applications: the panel already INCLUDES filings
that were abandoned at exam (they enter with reached_registration=False
and survived_5y=False).  Comparing Panel A to Panel B tests whether the
U-shape we observed in survival is driven by examiner-approval
selection or by post-registration mortality.

Outputs:
  paper/results/outcome_by_kl_lines.png
  paper/results/outcome_by_kl_lines.csv
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
    """Per-filing KL + outcomes + in_sec, CLEAN H=2y cut, 1985-2021 cohorts."""
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")
    sec_owners = sec.select("owner_name").unique().with_columns(
        pl.lit(True).alias("in_sec"))

    parts = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        op = PROC / f"outcomes_class{cls}.parquet"
        if not (sp.exists() and op.exists()): continue
        s = pl.read_parquet(
            sp,
            columns=["serial_number", "year", "prospective_kl", "retrospective_kl",
                     "n_ref_prospective", "n_ref_retrospective", "n_terms"],
        ).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
            & pl.col("year").is_between(1985, 2021)
        )
        o = pl.read_parquet(
            op, columns=["serial_number", "owner_name",
                          "reached_registration", "currently_live",
                          "abandoned_at_exam", "survived_5y"],
        )
        j = s.join(o, on="serial_number", how="inner")
        j = j.join(sec_owners, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False)
        ).with_columns(
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
        ).select("year", "prospective_kl", "retrospective_kl", "dkl",
                 "reached_registration", "survived_5y", "abandoned_at_exam",
                 "in_sec")
        if j.height:
            parts.append(j)
        del s, o, j
        gc.collect()
    return pl.concat(parts)


def quantile_curve(df: pl.DataFrame, var: str, outcome: str, n_bins: int = 20) -> pl.DataFrame:
    arr = df[var].to_numpy()
    qs = np.linspace(0, 1, n_bins + 1)
    cuts = np.quantile(arr, qs)
    bin_idx = np.clip(np.searchsorted(cuts[1:-1], arr), 0, n_bins - 1)
    g = df.with_columns(pl.Series("bin", bin_idx)).group_by("bin").agg([
        pl.len().alias("n"),
        pl.col(outcome).cast(pl.Float64).mean().alias("rate"),
        pl.col(var).mean().alias("mid"),
    ]).sort("bin")
    p = g["rate"].to_numpy(); n = g["n"].to_numpy().astype(np.float64)
    z = 1.96; denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return g.with_columns(pl.Series("lo", centre - half),
                          pl.Series("hi", centre + half))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[pool] loading …", flush=True)
    df = pool_panel()
    print(f"       n = {df.height:,}", flush=True)
    print(f"       reached_registration rate: {df['reached_registration'].cast(pl.Float64).mean():.3f}", flush=True)
    print(f"       survived_5y           rate: {df['survived_5y'].cast(pl.Float64).mean():.3f}", flush=True)
    print(f"       in_sec                rate: {df['in_sec'].cast(pl.Float64).mean():.3f}", flush=True)
    print(f"       abandoned_at_exam    rate: {df['abandoned_at_exam'].cast(pl.Float64).mean():.3f}", flush=True)

    axes_vars = [("dkl", "ΔKL", "#2b6cb0"),
                 ("prospective_kl", "Prospective KL", "#cc4444"),
                 ("retrospective_kl", "Retrospective KL", "#229922")]
    outcomes = [("reached_registration", "A. P(reached registration)"),
                ("survived_5y",           "B. P(survived 5y)"),
                ("in_sec",                "C. P(owner ever in SEC EDGAR)")]

    all_rows = []
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax_idx, (outcome, title) in enumerate(outcomes):
        ax = axs[ax_idx]
        baseline = df[outcome].cast(pl.Float64).mean()
        for var, var_label, color in axes_vars:
            g = quantile_curve(df, var, outcome, n_bins=20)
            pdf = g.to_pandas()
            ax.plot(pdf["mid"], pdf["rate"], "-o", color=color, lw=2, ms=4, label=var_label)
            ax.fill_between(pdf["mid"], pdf["lo"], pdf["hi"], color=color, alpha=0.16)
            for r in g.iter_rows(named=True):
                all_rows.append({"outcome": outcome, "axis": var_label, **r})
        ax.axhline(baseline, ls="--", color="#444", lw=1, label=f"corpus mean={baseline:.3f}")
        ax.set_xlabel("KL value (nats)")
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(f"Outcomes by KL — pooled across 45 classes, H=2y CLEAN cut, "
                 f"n={df.height:,}; 5y-observable cohorts 1985-2021", fontsize=12)
    fig.tight_layout()
    out_png = OUT / "outcome_by_kl_lines.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pl.from_dicts(all_rows).write_csv(OUT / "outcome_by_kl_lines.csv")
    print(f"\n[done] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
