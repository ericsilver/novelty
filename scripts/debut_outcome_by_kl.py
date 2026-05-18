"""Companion to outcome_by_kl_lines.py — restricted to DEBUT filings
(owner's first-ever filing across all classes).

This addresses the incumbent-volume concern: full-corpus IPO-rate-by-KL
is dominated by established public firms filing many low-ΔKL line
extensions.  The debut-only cut tests whether firms whose VERY FIRST
filing has high ΔKL go on to be publicly traded more often than
debut filers with low/middle ΔKL.

Three outcomes by KL bin:
  A.  P(reached registration)        — examiner approval
  B.  P(survived 5y)                  — alive 5y later
  C.  P(owner ever in SEC EDGAR)      — went public

Outputs:
  paper/results/debut_outcome_by_kl.png
  paper/results/debut_outcome_by_kl.csv
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


def compute_owner_debut() -> pl.DataFrame:
    """Pass 1: owner_name -> earliest filing_date across all classes."""
    parts = []
    for cls in class_list():
        tm_p = PROC / f"tm_class{cls}.parquet"
        d = pl.read_parquet(tm_p, columns=["owner_name", "filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        parts.append(d)
        del d
        gc.collect()
    return pl.concat(parts).group_by("owner_name").agg(
        pl.col("debut_date").min().alias("debut_date"))


def pool_debut_panel(debut: pl.DataFrame) -> pl.DataFrame:
    """Pass 2: per-filing KL + outcomes + in_sec for DEBUTS only."""
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))

    parts = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        op = PROC / f"outcomes_class{cls}.parquet"
        if not (sp.exists() and op.exists()): continue
        s = pl.read_parquet(
            sp,
            columns=["serial_number","year","prospective_kl","retrospective_kl",
                     "n_ref_prospective","n_ref_retrospective","n_terms"],
        ).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
            & pl.col("year").is_between(1985, 2021)
        )
        o = pl.read_parquet(
            op, columns=["serial_number","owner_name","filing_date",
                          "reached_registration","survived_5y"],
        ).join(debut, on="owner_name", how="left").filter(
            pl.col("filing_date") == pl.col("debut_date")
        )
        j = s.join(o, on="serial_number", how="inner")
        j = j.join(sec, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False)
        ).with_columns(
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
        ).select("prospective_kl","retrospective_kl","dkl",
                 "reached_registration","survived_5y","in_sec")
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
    print("[pass 1] owner debut dates …", flush=True)
    debut = compute_owner_debut()
    print(f"         {debut.height:,} unique owners", flush=True)

    print("[pass 2] debut-only panel …", flush=True)
    df = pool_debut_panel(debut)
    print(f"         {df.height:,} debut filings in CLEAN H=2y panel", flush=True)
    print(f"         reached_registration: {df['reached_registration'].cast(pl.Float64).mean():.3f}", flush=True)
    print(f"         survived_5y         : {df['survived_5y'].cast(pl.Float64).mean():.3f}", flush=True)
    print(f"         in_sec              : {df['in_sec'].cast(pl.Float64).mean():.4f}", flush=True)

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
        ax.axhline(baseline, ls="--", color="#444", lw=1, label=f"debut mean={baseline:.3f}")
        ax.set_xlabel("KL value (nats)")
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(f"Outcomes by KL — DEBUT filings only (owner's first-ever filing); n={df.height:,}",
                 fontsize=12)
    fig.tight_layout()
    out_png = OUT / "debut_outcome_by_kl.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pl.from_dicts(all_rows).write_csv(OUT / "debut_outcome_by_kl.csv")
    print(f"\n[done] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
