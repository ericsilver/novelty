"""Debut-filing outcomes under topic scoring (the method-grade figure).

  A. P(reached registration)            debut filings 1995-2018
  B. P(owner ever in SEC EDGAR | reg.)

Axes: topic_dkl / topic_pros / topic_retr.

Outputs:
  paper/results/debut_outcome_topic.png
  paper/results/debut_outcome_topic.json
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

FILE_LO, FILE_HI = 1995, 2018


def all_classes() -> list[str]:
    return sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def compute_owner_debut() -> pl.DataFrame:
    parts = []
    for cls in all_classes():
        d = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet", columns=["owner_name", "filing_date"]
        ).filter(pl.col("owner_name").is_not_null()
                 & (pl.col("filing_date").str.len_chars() == 8)
        ).group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        parts.append(d)
        del d
        gc.collect()
    return pl.concat(parts).group_by("owner_name").agg(
        pl.col("debut_date").min().alias("debut_date"))


def main() -> int:
    debut = compute_owner_debut()
    print(f"[debut] {debut.height:,} owners", file=sys.stderr, flush=True)
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))

    parts = []
    for cls in all_classes():
        tp = PROC / f"topic_surprise_class{cls}.parquet"
        if not tp.exists():
            continue
        topic = pl.read_parquet(tp).filter(
            pl.col("topic_dkl").is_finite()
            & pl.col("year").is_between(FILE_LO, FILE_HI))
        tok = pl.read_parquet(
            PROC / f"surprise_class{cls}.parquet",
            columns=["serial_number", "n_terms"]).filter(pl.col("n_terms") >= 3)
        t = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "owner_name", "filing_date", "registration_date"],
        ).with_columns(
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8
             ).alias("reached_registration"),
        ).join(debut, on="owner_name", how="left").filter(
            pl.col("filing_date") == pl.col("debut_date"))
        j = topic.join(tok, on="serial_number", how="inner").join(
            t, on="serial_number", how="inner").join(
            sec, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False),
        ).select("topic_pros", "topic_retr", "topic_dkl",
                 "reached_registration", "in_sec")
        if j.height:
            parts.append(j)
        del topic, tok, t, j
        gc.collect()
    df = pl.concat(parts)
    reg = df.filter(pl.col("reached_registration"))
    print(f"[panel] {df.height:,} debut filings; {reg.height:,} registered",
          file=sys.stderr, flush=True)

    def quintiles(frame, var, outcome):
        arr = frame[var].to_numpy()
        cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(cuts, arr)
        g = frame.with_columns(pl.Series("q", qi)).group_by("q").agg(
            pl.len().alias("n"),
            pl.col(outcome).cast(pl.Float64).mean().alias("rate")).sort("q")
        d = {f"q{int(r['q'])+1}": r["rate"] for r in g.iter_rows(named=True)}
        d["lift_q5_q1"] = d["q5"] - d["q1"]
        d["inverseU_depth"] = d["q3"] - 0.5 * (d["q1"] + d["q5"])
        return d

    metrics = {
        "n_debut": df.height, "n_registered": reg.height,
        "p_registered": float(df["reached_registration"].cast(pl.Float64).mean()),
        "p_sec_given_reg": float(reg["in_sec"].cast(pl.Float64).mean()),
        "registration": {v: quintiles(df, v, "reached_registration")
                         for v in ("topic_dkl", "topic_pros", "topic_retr")},
        "sec_given_reg": {v: quintiles(reg, v, "in_sec")
                          for v in ("topic_dkl", "topic_pros", "topic_retr")},
    }
    (OUT / "debut_outcome_topic.json").write_text(
        json.dumps(metrics, indent=1, default=float))
    print("reg by topic_dkl:", {k: round(v, 4) for k, v in
                                metrics["registration"]["topic_dkl"].items()},
          file=sys.stderr, flush=True)
    print("sec by topic_dkl:", {k: round(v, 5) for k, v in
                                metrics["sec_given_reg"]["topic_dkl"].items()},
          file=sys.stderr, flush=True)

    def curve(frame, var, outcome, ax, color, label):
        arr = frame[var].to_numpy()
        cuts = np.quantile(arr, np.linspace(0, 1, 21))
        bins = np.clip(np.searchsorted(cuts[1:-1], arr), 0, 19)
        g = frame.with_columns(pl.Series("bin", bins)).group_by("bin").agg(
            pl.len().alias("n"),
            pl.col(outcome).cast(pl.Float64).mean().alias("rate"),
            pl.col(var).mean().alias("mid")).sort("bin").to_pandas()
        se = np.sqrt(g["rate"] * (1 - g["rate"]) / g["n"])
        ax.fill_between(g["mid"], g["rate"] - 1.96*se, g["rate"] + 1.96*se,
                        color=color, alpha=0.16)
        ax.plot(g["mid"], g["rate"], "-o", color=color, lw=2, ms=4, label=label)

    axes_vars = [("topic_dkl", "$\\Delta$KL (topic)", "#2b6cb0"),
                 ("topic_pros", "Prospective (topic)", "#cc4444"),
                 ("topic_retr", "Retrospective (topic)", "#229922")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (frame, outcome, title) in zip(axs, [
        (df, "reached_registration", "A. P(reached registration)"),
        (reg, "in_sec", "B. P(owner ever in SEC EDGAR | registration)"),
    ]):
        base = float(frame[outcome].cast(pl.Float64).mean())
        for var, label, color in axes_vars:
            curve(frame, var, outcome, ax, color, label)
        ax.axhline(base, ls="--", color="#444", lw=1, label=f"mean={base:.3f}")
        ax.set_xlabel("topic KL value (nats)")
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Debut filings under topic scoring, {FILE_LO}-{FILE_HI}; "
        f"n={df.height:,}, registered n={reg.height:,}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "debut_outcome_topic.png", dpi=150, bbox_inches="tight")
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
