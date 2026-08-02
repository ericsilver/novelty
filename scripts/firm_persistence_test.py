"""Tests whether the survival U-shape on ΔKL is driven by firm survival
or by trademark-level value.

Hypothesis to falsify: trademarks that are 'on trend' (ΔKL ≈ 0) might
themselves not be valuable, even though the firm survives. If true, the
U-shape on filing survival should largely vanish when we restrict to
filings from firms that we KNOW persisted --- firms that abandoned
SOME of their trademarks (showing the firm survived past the lapse) but
also kept some alive (showing the firm continued operating).

For each owner with ≥3 filings in the clean H=2y panel, classify:
  - all_survived:  every filing survived 5y → firm clearly persistent
  - all_failed:    no filings survived 5y → firm probably did not persist
  - mixed:         some survived, some did not → firm persisted, individual
                   filings were dropped on their merits

If the U-shape persists within the MIXED cohort, the survival signal is
trademark-level (the value of the specific filing) not firm-level (whether
the firm survives). If the U-shape disappears within the MIXED cohort,
the signal is firm-level (the firm sometimes survives the lapse).

Outputs:
  paper/results/firm_persistence_test.png
  paper/results/firm_persistence_test.csv
  paper/results/firm_persistence_test.txt
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
    parts = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        op = PROC / f"outcomes_class{cls}.parquet"
        if not (sp.exists() and op.exists()):
            continue
        s = pl.read_parquet(
            sp,
            columns=["serial_number","owner_name","year","kl_vs_past",
                     "kl_vs_future","n_ref_past","n_ref_future","n_terms"],
        ).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
            & pl.col("year").is_between(1985, 2021)
        )
        o = pl.read_parquet(op, columns=["serial_number", "survived_5y"])
        j = s.join(o, on="serial_number", how="inner").with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        ).select("owner_name","dkl","survived_5y")
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
    print("[pool] loading panel …", flush=True)
    df = pool_panel()
    print(f"       {df.height:,} filings; {df['owner_name'].n_unique():,} unique owners",
          flush=True)

    firm = df.group_by("owner_name").agg([
        pl.len().alias("n_total"),
        pl.col("survived_5y").cast(pl.Int32).sum().alias("n_survived"),
    ]).with_columns(
        (pl.col("n_total") - pl.col("n_survived")).alias("n_failed"),
    ).filter(pl.col("n_total") >= 3)
    print(f"       firms with >=3 clean filings: {firm.height:,}", flush=True)

    firm = firm.with_columns(
        pl.when((pl.col("n_failed") == 0)).then(pl.lit("all_survived"))
          .when((pl.col("n_survived") == 0)).then(pl.lit("all_failed"))
          .otherwise(pl.lit("mixed")).alias("cohort"),
    )
    coh = firm.group_by("cohort").agg([
        pl.len().alias("n_firms"),
        pl.col("n_total").sum().alias("n_filings"),
    ])
    print("[classify]", flush=True)
    print(coh.to_pandas().to_string(index=False), flush=True)

    df_with_cohort = df.join(firm.select("owner_name", "cohort"),
                              on="owner_name", how="inner")
    curves = {}
    for cohort in ["all_survived", "mixed", "all_failed"]:
        sub = df_with_cohort.filter(pl.col("cohort") == cohort)
        if sub.height == 0:
            continue
        curves[cohort] = quantile_curve(sub, "dkl", "survived_5y", n_bins=20)
        print(f"  {cohort}: n={sub.height:,}", flush=True)

    rows = []
    for cohort, g in curves.items():
        for r in g.iter_rows(named=True):
            rows.append({"cohort": cohort, **r})
    pl.from_dicts(rows).write_csv(OUT / "firm_persistence_test.csv")

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = {"all_survived": "#229922", "mixed": "#2b6cb0", "all_failed": "#cc4444"}
    for cohort, g in curves.items():
        pdf = g.to_pandas()
        n_total = int(g["n"].sum())
        ax.plot(pdf["mid"], pdf["rate"], "-o", color=colors[cohort], lw=2, ms=5,
                label=f"{cohort.replace('_', ' ')}  (n={n_total:,})")
        ax.fill_between(pdf["mid"], pdf["lo"], pdf["hi"], color=colors[cohort], alpha=0.16)
    ax.set_xlabel("$\\Delta KL$ (nats)")
    ax.set_ylabel("Filing-level 5y survival rate")
    ax.set_title("Does the U-shape persist within firms that demonstrably survived?\n"
                 "Per-firm-cohort survival-by-$\\Delta KL$ on the clean H=2y panel")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "firm_persistence_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines = ["Firm-persistence test: does the U-shape on filing-level survival\n"
             "persist within firms that demonstrably survived past one of their\n"
             "trademarks' Section-8 lapse?\n"]
    lines.append(f"{'cohort':<15s}  {'n_firms':>10s}  {'n_filings':>12s}  "
                 f"{'bot ΔKL':>9s}  {'mid ΔKL':>9s}  {'top ΔKL':>9s}  {'U-depth':>8s}")
    for cohort, g in curves.items():
        pdf = g.to_pandas()
        n = pdf.shape[0]
        bot = pdf.iloc[0]["rate"]; mid = pdf.iloc[n//2]["rate"]; top = pdf.iloc[-1]["rate"]
        depth = max(bot, top) - mid
        firm_row = coh.filter(pl.col("cohort")==cohort)
        nf = firm_row["n_firms"][0]
        nfilings = firm_row["n_filings"][0]
        lines.append(f"  {cohort:<13s}  {nf:>10,}  {nfilings:>12,}  "
                     f"{bot:>9.3f}  {mid:>9.3f}  {top:>9.3f}  {depth:>+8.3f}")

    text = "\n".join(lines)
    (OUT / "firm_persistence_test.txt").write_text(text)
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
