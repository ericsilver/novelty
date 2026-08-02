"""Compare halflife sweep outputs and pick the signal-maximising H.

For each surprise_decay_class009_h{H}*.parquet on disk, plus the
flat-window baseline surprise_class009.parquet, compute several
``signal'' metrics on the (filing -> outcome) relationship:

  S1 = corr(ΔKL, passed_5y)              -- main downstream signal
  S2 = corr(kl_vs_past, reached_reg) -- examination signal
  S3 = corr(kl_vs_future, currently_live)
  S4 = std(ΔKL)                          -- spread / discrimination
  S5 = pseudo-R² of logit passed_5y ~ z(ΔKL) + log(n_terms)
  S6 = same logit's |coef on z(dkl)| -- effect size after length control
  S7 = within-firm AR(1) on ΔKL         -- temporal coherence

Then pick H that maximises a chosen blend, defaulting to S1.

Outputs:
  paper/results/halflife_signal_class009.png
  paper/results/halflife_signal_class009.txt
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"


def parse_h(stem: str) -> float | None:
    m = re.search(r"_h([0-9p]+)(?:y)?$", stem)
    if not m: return None
    return float(m.group(1).replace("p", "."))


def load_outcomes() -> pl.DataFrame:
    return pl.read_parquet(PROC / "outcomes_class009.parquet").select(
        "serial_number", "year", "n_terms", "reached_registration",
        "currently_live"
    ).with_columns(
        (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y"),
    )


def load_baseline() -> pl.DataFrame:
    """Flat 5y window baseline."""
    return pl.read_parquet(PROC / "surprise_class009.parquet").select(
        "serial_number", "kl_vs_past", "kl_vs_future",
        "n_ref_past", "n_ref_future", "n_terms"
    )


def load_decay(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).select(
        "serial_number", "kl_vs_past", "kl_vs_future",
        "n_eff_prospective", "n_eff_retrospective", "n_terms"
    )


def signal_metrics(label: str, df: pl.DataFrame) -> dict:
    # Apply CLEAN-equivalent restriction
    has_ref = (
        ("n_ref_past" in df.columns and (df["n_ref_past"] >= 1000).all())
        or ("n_eff_prospective" in df.columns)
    )
    df = df.with_columns((pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"))
    # base filter: finite KLs, year window, n_terms >= 3
    if "n_ref_past" in df.columns:
        df = df.filter((pl.col("n_ref_past") >= 1000) & (pl.col("n_ref_future") >= 1000))
    elif "n_eff_prospective" in df.columns:
        df = df.filter((pl.col("n_eff_prospective") >= 1000) & (pl.col("n_eff_retrospective") >= 1000))
    df = df.filter(
        (pl.col("n_terms") >= 3)
        & pl.col("kl_vs_past").is_finite()
        & pl.col("kl_vs_future").is_finite()
        & pl.col("dkl").is_finite()
    )
    n = df.height
    if n == 0:
        return {"label": label, "n": 0}
    pdf = df.to_pandas()

    s1 = float(np.corrcoef(pdf["dkl"], pdf["passed_5y"])[0, 1])
    s2 = float(np.corrcoef(pdf["kl_vs_past"], pdf["reached_registration"])[0, 1])
    s3 = float(np.corrcoef(pdf["kl_vs_future"], pdf["currently_live"])[0, 1])
    s4 = float(pdf["dkl"].std())

    # Logit pseudo-R² and z-scored coef
    z = (pdf["dkl"] - pdf["dkl"].mean()) / pdf["dkl"].std()
    log_n_terms = np.log(pdf["n_terms"].clip(lower=1))
    X = sm.add_constant(np.column_stack([z, log_n_terms]))
    y = pdf["passed_5y"].astype(int).to_numpy()
    try:
        m = sm.Logit(y, X).fit(disp=False, maxiter=80)
        s5 = float(m.prsquared)
        s6 = float(abs(m.params[1]))  # coef on z(dkl)
    except Exception:
        s5 = float("nan"); s6 = float("nan")

    return {
        "label": label, "n": n,
        "S1_corr_dkl_pass5": s1,
        "S2_corr_pros_reg": s2,
        "S3_corr_retr_live": s3,
        "S4_std_dkl": s4,
        "S5_pseudoR2": s5,
        "S6_z_dkl_coef": s6,
    }


def main() -> int:
    outcomes = load_outcomes()
    print(f"[outcomes] {outcomes.height:,} rows", flush=True)

    rows = []

    # Baseline (flat 5y window)
    base = load_baseline().join(outcomes, on="serial_number", how="inner")
    print(f"[base ] joined {base.height:,} rows; computing signal…", flush=True)
    rows.append({"H_pros": 5.0, "H_retr": 5.0, "model": "flat-5y", **signal_metrics("flat-5y", base)})

    # Decay variants
    for f in sorted(PROC.glob("surprise_decay_class009_h*.parquet")):
        # Try to recover halflife from filename
        stem = f.stem
        m = re.search(r"_h([0-9p]+)y?$", stem)
        if not m: continue
        h = float(m.group(1).replace("p", "."))
        decay = load_decay(f).join(outcomes, on="serial_number", how="inner")
        print(f"[H={h:>4}] joined {decay.height:,} rows; computing signal…", flush=True)
        rows.append({"H_pros": h, "H_retr": h, "model": f"decay H={h}",
                     **signal_metrics(f"decay H={h}", decay)})

    # Sort by halflife
    rows.sort(key=lambda r: r["H_pros"])

    out_txt = RESULTS / "halflife_signal_class009.txt"
    with out_txt.open("w") as f:
        f.write("Halflife sweep on Class 9 — signal metrics on the same outcome panel.\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"{'model':<14s}{'n':>10s}{'corr(ΔKL, pass5)':>20s}"
                f"{'corr(vs-past, reg)':>19s}{'corr(vs-future, live)':>23s}"
                f"{'std(ΔKL)':>11s}{'pseudoR²':>11s}{'|z(dkl) β|':>13s}\n")
        for r in rows:
            f.write(f"{r['model']:<14s}{r.get('n',0):>10,}"
                    f"{r.get('S1_corr_dkl_pass5',float('nan')):>+20.4f}"
                    f"{r.get('S2_corr_pros_reg',float('nan')):>+19.4f}"
                    f"{r.get('S3_corr_retr_live',float('nan')):>+23.4f}"
                    f"{r.get('S4_std_dkl',float('nan')):>11.4f}"
                    f"{r.get('S5_pseudoR2',float('nan')):>11.4f}"
                    f"{r.get('S6_z_dkl_coef',float('nan')):>13.4f}\n")

        # Ranked by primary signal
        valid = [r for r in rows if not math.isnan(r.get("S1_corr_dkl_pass5", float("nan")))]
        ranked = sorted(valid, key=lambda r: r["S1_corr_dkl_pass5"], reverse=True)
        f.write("\nRanked by S1 = corr(ΔKL, passed_5y) (primary signal):\n")
        for r in ranked:
            f.write(f"  {r['model']:<14s}  S1={r['S1_corr_dkl_pass5']:+.4f}\n")
        ranked2 = sorted(valid, key=lambda r: r["S5_pseudoR2"], reverse=True)
        f.write("\nRanked by S5 = pseudo-R² of logit passed_5y ~ z(ΔKL) + log(n_terms):\n")
        for r in ranked2:
            f.write(f"  {r['model']:<14s}  R²={r['S5_pseudoR2']:.5f}\n")
    print(f"\n[done] wrote {out_txt}", flush=True)
    print(f"\nResults:")
    with out_txt.open() as f:
        print(f.read())

    # ---- Figure ----
    if rows:
        hs = [r["H_pros"] for r in rows]
        s1s = [r.get("S1_corr_dkl_pass5", float("nan")) for r in rows]
        s5s = [r.get("S5_pseudoR2", float("nan")) for r in rows]
        s6s = [r.get("S6_z_dkl_coef", float("nan")) for r in rows]
        s4s = [r.get("S4_std_dkl", float("nan")) for r in rows]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0,0].plot(hs, s1s, "o-", color="#2b6cb0", lw=2)
        axes[0,0].set_xlabel("Halflife (years; flat-5y plotted at H=5)")
        axes[0,0].set_ylabel("corr(ΔKL, passed_5y)")
        axes[0,0].set_title("S1 — primary signal")
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(hs, s5s, "o-", color="#cc4444", lw=2)
        axes[0,1].set_xlabel("Halflife (years)")
        axes[0,1].set_ylabel("Logit pseudo-R²")
        axes[0,1].set_title("S5 — model fit (logit pass5 ~ z(ΔKL)+log(n_terms))")
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot(hs, s6s, "o-", color="#888", lw=2)
        axes[1,0].set_xlabel("Halflife (years)")
        axes[1,0].set_ylabel("|coef on z(ΔKL)|")
        axes[1,0].set_title("S6 — effect size after length control")
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].plot(hs, s4s, "o-", color="#229922", lw=2)
        axes[1,1].set_xlabel("Halflife (years)")
        axes[1,1].set_ylabel("std(ΔKL)")
        axes[1,1].set_title("S4 — distribution spread")
        axes[1,1].grid(True, alpha=0.3)

        fig.suptitle(f"Halflife sweep on Class 9 — pick H to maximise signal", fontsize=12)
        fig.tight_layout()
        out_png = RESULTS / "halflife_signal_class009.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[done] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
