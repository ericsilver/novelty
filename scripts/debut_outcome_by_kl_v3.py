"""Debut-filing outcome figure with CORRECT outcome semantics.

  A. P(reached registration)            registration_date populated;
                                         filing years 1990-2018 (resolved)
  B. P(owner ever in SEC EDGAR | reg.)  same denominator as A's successes

Section 8 gate survival is reported separately (s8_survival_corrected.py)
because its clean cohort is registrations 2016-2018, which barely overlaps
the debut panel's resolved-filing window.

Outputs:
  paper/results/debut_outcome_by_kl.png   (overwrites the invalidated figure)
  paper/results/debut_outcome_by_kl.csv
  paper/results/debut_outcome_metrics.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "paper" / "results"

# 1995 = first scored year whose full 5-year past window lies in the dense
# corpus; see burnin_optimization.py (1990 median |bias| ~2.0 nats, <=0.04
# from 1993).
FILE_LO, FILE_HI = 1995, 2018


def all_class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def surprise_class_list() -> list[str]:
    return sorted(p.stem.replace("surprise_class", "")
                  for p in PROC.glob("surprise_class???.parquet"))


def compute_owner_debut() -> pl.DataFrame:
    parts = []
    for cls in all_class_list():
        d = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet", columns=["owner_name", "filing_date"]
        ).filter(
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
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))
    parts = []
    for cls in surprise_class_list():
        s = pl.read_parquet(
            PROC / f"surprise_class{cls}.parquet",
            columns=["serial_number", "year", "prospective_kl", "retrospective_kl",
                     "n_ref_prospective", "n_ref_retrospective", "n_terms"],
        ).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
            & pl.col("year").is_between(FILE_LO, FILE_HI)
        )
        t = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "owner_name", "filing_date", "registration_date"],
        ).with_columns(
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8
             ).alias("reached_registration"),
        ).join(debut, on="owner_name", how="left").filter(
            pl.col("filing_date") == pl.col("debut_date"))
        j = s.join(t, on="serial_number", how="inner")
        j = j.join(sec, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False),
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
        ).select("prospective_kl", "retrospective_kl", "dkl",
                 "reached_registration", "in_sec")
        if j.height:
            parts.append(j)
        del s, t, j
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
    z = 1.96; denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return g.with_columns(pl.Series("lo", centre - half), pl.Series("hi", centre + half))


def quintile_stats(df: pl.DataFrame, var: str, outcome: str) -> dict:
    arr = df[var].to_numpy()
    cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(cuts, arr)
    g = df.with_columns(pl.Series("q", qi)).group_by("q").agg(
        pl.len().alias("n"),
        pl.col(outcome).cast(pl.Float64).mean().alias("rate")).sort("q")
    d = {f"q{int(r['q'])+1}": r["rate"] for r in g.iter_rows(named=True)}
    d["lift_q5_q1"] = d["q5"] - d["q1"]
    return d


def main() -> int:
    print("[pass 1] owner debut dates ...", file=sys.stderr, flush=True)
    debut = compute_owner_debut()
    print(f"         {debut.height:,} owners", file=sys.stderr, flush=True)
    df = pool_debut_panel(debut)
    reg = df.filter(pl.col("reached_registration"))
    print(f"[panel] {df.height:,} debut filings; {reg.height:,} registered",
          file=sys.stderr, flush=True)

    metrics = {
        "n_debut": df.height,
        "n_registered": reg.height,
        "p_registered": float(df["reached_registration"].cast(pl.Float64).mean()),
        "p_sec_given_reg": float(reg["in_sec"].cast(pl.Float64).mean()),
        "registration": {v: quintile_stats(df, v, "reached_registration")
                         for v in ("dkl", "prospective_kl", "retrospective_kl")},
        "sec_given_reg": {v: quintile_stats(reg, v, "in_sec")
                          for v in ("dkl", "prospective_kl", "retrospective_kl")},
    }
    (OUT / "debut_outcome_metrics.json").write_text(
        json.dumps(metrics, indent=1, default=float))
    print(json.dumps({k: metrics[k] for k in
                      ("n_debut", "p_registered", "p_sec_given_reg")},
                     indent=1, default=float), file=sys.stderr, flush=True)
    print("reg by dkl:", metrics["registration"]["dkl"], file=sys.stderr, flush=True)
    print("sec by dkl:", metrics["sec_given_reg"]["dkl"], file=sys.stderr, flush=True)

    axes_vars = [("dkl", "$\\Delta$KL", "#2b6cb0"),
                 ("prospective_kl", "Prospective KL", "#cc4444"),
                 ("retrospective_kl", "Retrospective KL", "#229922")]
    panels = [
        (df,  "reached_registration", "A. P(reached registration)"),
        (reg, "in_sec",               "B. P(owner ever in SEC EDGAR | registration)"),
    ]
    all_rows = []
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (frame, outcome, title) in zip(axs, panels):
        baseline = frame[outcome].cast(pl.Float64).mean()
        for var, var_label, color in axes_vars:
            g = quantile_curve(frame, var, outcome)
            pdf = g.to_pandas()
            ax.plot(pdf["mid"], pdf["rate"], "-o", color=color, lw=2, ms=4, label=var_label)
            ax.fill_between(pdf["mid"], pdf["lo"], pdf["hi"], color=color, alpha=0.16)
            for r in g.iter_rows(named=True):
                all_rows.append({"outcome": outcome, "axis": var_label, **r})
        ax.axhline(baseline, ls="--", color="#444", lw=1, label=f"mean={baseline:.3f}")
        ax.set_xlabel("KL value (nats)"); ax.set_ylabel("rate")
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Debut filings (owner's first-ever filing), {FILE_LO}-{FILE_HI}; "
        f"n={df.height:,}, registered n={reg.height:,}. Registration = "
        "registration_date populated.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "debut_outcome_by_kl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pl.from_dicts(all_rows).write_csv(OUT / "debut_outcome_by_kl.csv")
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
