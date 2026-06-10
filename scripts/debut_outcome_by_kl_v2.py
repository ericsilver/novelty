"""Merged outcome figure for the paper: debut filings (owner's first-ever
filing), three panels by KL value:

  A.  P(reached registration)                       - all debut filings
  B.  P(survived 5y | reached registration)         - renewal among registered
  C.  P(owner ever in SEC EDGAR | reached reg.)     - listing among registered

Outcomes derive from status codes on tm_class parquets (the old
outcomes_class parquets defined survived_5y so loosely that it equalled
reached_registration; see commit history). Conditional renewal here is
status bucket 6 (live, maintained) among bucket 6 or 8 (ever registered).

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
OUT = REPO / "paper" / "results"

SURPRISE_CLASSES = ["009", "035", "039", "042"]


def all_class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


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
    for clss in SURPRISE_CLASSES:
        s = pl.read_parquet(
            PROC / f"surprise_class{clss}.parquet",
            columns=["serial_number", "year", "prospective_kl", "retrospective_kl",
                     "n_ref_prospective", "n_ref_retrospective", "n_terms"],
        ).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
            & pl.col("year").is_between(1990, 2020)
        )
        t = pl.read_parquet(
            PROC / f"tm_class{clss}.parquet",
            columns=["serial_number", "owner_name", "filing_date", "status_code"],
        ).with_columns(
            pl.col("status_code").str.slice(0, 1).alias("status_bucket"),
        ).with_columns(
            ((pl.col("status_bucket") == "6") | (pl.col("status_bucket") == "8")
             ).alias("reached_registration"),
            (pl.col("status_bucket") == "6").alias("renewed"),
        ).join(debut, on="owner_name", how="left").filter(
            pl.col("filing_date") == pl.col("debut_date")
        )
        j = s.join(t, on="serial_number", how="inner")
        j = j.join(sec, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False),
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
        ).select("prospective_kl", "retrospective_kl", "dkl",
                 "reached_registration", "renewed", "in_sec")
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
    return g.with_columns(pl.Series("lo", centre - half),
                          pl.Series("hi", centre + half))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[pass 1] owner debut dates ...", flush=True)
    debut = compute_owner_debut()
    print(f"         {debut.height:,} unique owners", flush=True)

    print("[pass 2] debut-only panel ...", flush=True)
    df = pool_debut_panel(debut)
    reg = df.filter(pl.col("reached_registration"))
    print(f"         {df.height:,} debut filings; {reg.height:,} reached registration", flush=True)
    print(f"         P(reached_registration)      : {df['reached_registration'].cast(pl.Float64).mean():.3f}", flush=True)
    print(f"         P(renewed | registered)      : {reg['renewed'].cast(pl.Float64).mean():.3f}", flush=True)
    print(f"         P(in_sec | registered)       : {reg['in_sec'].cast(pl.Float64).mean():.4f}", flush=True)

    axes_vars = [("dkl", "$\\Delta$KL", "#2b6cb0"),
                 ("prospective_kl", "Prospective KL", "#cc4444"),
                 ("retrospective_kl", "Retrospective KL", "#229922")]
    panels = [
        (df,  "reached_registration", "A. P(reached registration)"),
        (reg, "renewed",              "B. P(survived 5y | registration)"),
        (reg, "in_sec",               "C. P(owner ever in SEC EDGAR | registration)"),
    ]

    all_rows = []
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (frame, outcome, title) in zip(axs, panels):
        baseline = frame[outcome].cast(pl.Float64).mean()
        for var, var_label, color in axes_vars:
            g = quantile_curve(frame, var, outcome, n_bins=20)
            pdf = g.to_pandas()
            ax.plot(pdf["mid"], pdf["rate"], "-o", color=color, lw=2, ms=4, label=var_label)
            ax.fill_between(pdf["mid"], pdf["lo"], pdf["hi"], color=color, alpha=0.16)
            for r in g.iter_rows(named=True):
                all_rows.append({"outcome": outcome, "axis": var_label, **r})
        ax.axhline(baseline, ls="--", color="#444", lw=1, label=f"mean={baseline:.3f}")
        ax.set_xlabel("KL value (nats)")
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"Outcomes by KL, debut filings (owner's first-ever filing); "
        f"n={df.height:,}, registered subset n={reg.height:,}",
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
