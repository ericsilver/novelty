"""V2 of outcome_by_kl_lines: distinguishes P(survived 5y | reached
registration) from the unconditional P(reached registration). The v1
version had survived_5y = (reached AND currently live), so panel B
was effectively a strict superset of panel A and the two looked
nearly identical for prospective/retrospective KL. V2 correctly
conditions panel B on registration so the difference is meaningful.

Adds Wilson 95% CI error bars on every binned point.

Outputs:
  paper/results/outcome_by_kl_lines_v2.png
  paper/results/outcome_by_kl_lines_v2.csv
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
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def pool_panel() -> pl.DataFrame:
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))
    parts = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        op = PROC / f"outcomes_class{cls}.parquet"
        if not (sp.exists() and op.exists()): continue
        s = pl.read_parquet(sp, columns=["serial_number","year","kl_vs_past","kl_vs_future",
                                          "n_ref_past","n_ref_future","n_terms"]).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
            & pl.col("year").is_between(1990, 2021)
        )
        o = pl.read_parquet(op, columns=["serial_number","owner_name","reached_registration","currently_live","survived_5y"])
        j = s.join(o, on="serial_number", how="inner")
        j = j.join(sec, on="owner_name", how="left").with_columns(pl.col("in_sec").fill_null(False))
        j = j.with_columns((pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"))
        # survived_5y_cond = currently_live, restricted to reached_registration=True
        # We'll compute this in the binning logic
        if j.height: parts.append(j.select("dkl","kl_vs_past","kl_vs_future",
                                            "reached_registration","currently_live","survived_5y","in_sec"))
        del s, o, j
        gc.collect()
    return pl.concat(parts)


def wilson_ci(p, n, z=1.96):
    n = np.where(n == 0, 1, n)
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return centre - half, centre + half


def bin_curve(df: pl.DataFrame, var: str, outcome_col: str, n_bins: int = 20,
              cond_col: str | None = None) -> pl.DataFrame:
    """Bin by quantile of var. If cond_col given, restrict to rows where cond_col==True."""
    if cond_col is not None:
        df = df.filter(pl.col(cond_col))
    arr = df[var].to_numpy()
    if len(arr) == 0: return pl.DataFrame()
    cuts = np.quantile(arr, np.linspace(0,1,n_bins+1))
    bin_idx = np.clip(np.searchsorted(cuts[1:-1], arr), 0, n_bins-1)
    g = df.with_columns(pl.Series("bin", bin_idx)).group_by("bin").agg([
        pl.len().alias("n"),
        pl.col(outcome_col).cast(pl.Float64).mean().alias("rate"),
        pl.col(var).mean().alias("mid"),
    ]).sort("bin")
    p = g["rate"].to_numpy(); n = g["n"].to_numpy().astype(np.float64)
    lo, hi = wilson_ci(p, n)
    return g.with_columns(pl.Series("lo", lo), pl.Series("hi", hi))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[load] panel …", flush=True)
    df = pool_panel()
    print(f"       n = {df.height:,}", flush=True)

    axes_vars = [("dkl", r"$\Delta KL$", "#2b6cb0"),
                 ("kl_vs_past", r"$KL_{\mathrm{pros}}$", "#cc4444"),
                 ("kl_vs_future", r"$KL_{\mathrm{retr}}$", "#229922")]
    outcomes = [
        ("reached_registration", None, "A. P(reached registration)"),
        ("currently_live", "reached_registration",
         "B. P(survived 5y | reached registration)"),
        ("in_sec", None, "C. P(owner ever in SEC EDGAR)"),
    ]

    all_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_idx, (outcome_col, cond_col, title) in enumerate(outcomes):
        ax = axes[ax_idx]
        # Baseline
        sub = df if cond_col is None else df.filter(pl.col(cond_col))
        baseline = float(sub[outcome_col].cast(pl.Float64).mean())
        for var, var_label, color in axes_vars:
            g = bin_curve(df, var, outcome_col, n_bins=20, cond_col=cond_col)
            if g.height == 0: continue
            pdf = g.to_pandas()
            ax.errorbar(pdf["mid"], pdf["rate"],
                        yerr=[pdf["rate"] - pdf["lo"], pdf["hi"] - pdf["rate"]],
                        fmt="-o", color=color, lw=2, ms=4, capsize=2, capthick=0.8,
                        elinewidth=0.8, label=var_label)
            for r in g.iter_rows(named=True):
                all_rows.append({"outcome": outcome_col, "axis": var_label, **r})
        ax.axhline(baseline, ls="--", color="#444", lw=1, label=f"corpus mean={baseline:.3f}")
        ax.set_xlabel("KL value (nats)")
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(f"Outcomes by KL bin (with Wilson 95\\% CIs), pooled across 44 NICE classes, "
                 f"H=2y CLEAN cut, n={df.height:,}; 1990--2021 cohorts.\n"
                 f"Panel B is conditioned on \\texttt{{reached\\_registration = True}} so the "
                 f"P(survived) effect is separable from examiner approval.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "outcome_by_kl_lines_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pl.from_dicts(all_rows).write_csv(OUT / "outcome_by_kl_lines_v2.csv")
    print(f"[done] wrote {OUT/'outcome_by_kl_lines_v2.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
