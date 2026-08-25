"""P(owner ever in SEC EDGAR | registered debut) as fine-grained line curves.

Same frame as topic_debut.py (registered debut filings 1995-2018, production
rolling scoring, n_terms >= 3 floor), but ventiles (20 bins, pooled cuts)
of lead L, atypicality A, and lead magnitude |L|. Binomial 95% bands.

Outputs: paper/results/fig_success_ventiles.png
         paper/results/fig_success_ventiles.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
FILE_LO, FILE_HI = 1995, 2018
NB = 20


def log(m): print(m, file=sys.stderr, flush=True)


def all_classes():
    return sorted(p.stem.replace("tm_class", "") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def main() -> int:
    parts = []
    for cls in all_classes():
        d = pl.read_parquet(PROC / f"tm_class{cls}.parquet", columns=["owner_name", "filing_date"]).filter(
            pl.col("owner_name").is_not_null() & (pl.col("filing_date").str.len_chars() == 8)
        ).group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        parts.append(d)
    debut = pl.concat(parts).group_by("owner_name").agg(pl.col("debut_date").min().alias("debut_date"))
    del parts; gc.collect()
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name").unique().with_columns(
        pl.lit(True).alias("in_sec"))
    parts = []
    for cls in all_classes():
        tp = PROC / f"rolling_surprise_class{cls}.parquet"
        if not tp.exists():
            continue
        topic = pl.read_parquet(tp).filter(pl.col("topic_dkl").is_finite()
                                           & pl.col("year").is_between(FILE_LO, FILE_HI))
        tok = pl.read_parquet(PROC / f"surprise_class{cls}.parquet",
                              columns=["serial_number", "n_terms"]).filter(pl.col("n_terms") >= 3)
        t = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                            columns=["serial_number", "owner_name", "filing_date", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8
        ).join(debut, on="owner_name", how="left").filter(pl.col("filing_date") == pl.col("debut_date"))
        j = topic.join(tok, on="serial_number", how="inner").join(t, on="serial_number", how="inner").join(
            sec, on="owner_name", how="left").with_columns(pl.col("in_sec").fill_null(False)).select(
            "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl", "in_sec")
        if j.height:
            parts.append(j)
        del topic, tok, t, j
        gc.collect()
    df = pl.concat(parts).with_columns(
        pl.col("topic_dkl").alias("L"), pl.col("topic_dkl").abs().alias("absL"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    log(f"[frame] {df.height:,} registered debuts; base {float(df['in_sec'].mean()):.5f}")

    out = {"n": int(df.height), "base": float(df["in_sec"].cast(pl.Float64).mean()), "curves": {}}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, var, lab, color in ((axes[0], "L", "lead $L$", "#2b6cb0"),
                                (axes[1], "absL", "lead magnitude $|L|$", "#7d3c98"),
                                (axes[2], "A", "atypicality $A$", "#c0392b")):
        arr = df[var].to_numpy()
        cuts = np.quantile(arr, np.linspace(0, 1, NB + 1))
        bins = np.clip(np.searchsorted(cuts[1:-1], arr), 0, NB - 1)
        g = df.with_columns(pl.Series("bin", bins)).group_by("bin").agg(
            pl.len().alias("n"), pl.col("in_sec").cast(pl.Float64).mean().alias("p"),
            pl.col(var).mean().alias("mid")).sort("bin")
        p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
        mid = [float(v) for v in g["mid"]]
        se = [(a * (1 - a) / b) ** 0.5 for a, b in zip(p, n)]
        out["curves"][var] = {"mid": mid, "p": p, "n": n, "se": se}
        ax.fill_between(mid, [100 * (a - 1.96 * b) for a, b in zip(p, se)],
                        [100 * (a + 1.96 * b) for a, b in zip(p, se)], color=color, alpha=0.15, lw=0)
        ax.plot(mid, [100 * v for v in p], "o-", color=color, lw=1.8, ms=3.5)
        ax.axhline(100 * out["base"], ls="--", color="#555", lw=1)
        ax.set_xlabel(f"{lab} (nats), ventile means")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("% of owners ever in SEC EDGAR")
    fig.tight_layout()
    fig.savefig(RES / "fig_success_ventiles.png", dpi=150, bbox_inches="tight")
    (RES / "fig_success_ventiles.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
