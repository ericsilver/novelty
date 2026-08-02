"""Single-class (009) exploration once the corpus is built.

Computes, on real Class-009 (Software & Electronics) filings:
  - signed dKL = pros - retr (resonance) and abs dKL = |pros - retr| (diffusion §2.2)
  - descriptives + per-year mean signed dKL (is the class drifting ahead/behind?)
  - FACE VALIDITY: most-resonant (signed+) and most-tired (signed-) marks
  - F0b reliability: does |dKL| inflate for short filings (few terms)?
  - F0c half-life sensitivity: sign-flip rate of signed dKL across windows
    W in {3,5,7}y, if those surprise files exist.

Inputs (build first):
  data/processed/surprise_class009.parquet         (W=5, default)
  data/processed/surprise_class009_W3.parquet      (optional)
  data/processed/surprise_class009_W7.parquet      (optional)
Output: paper/results/wsG_class009.json (+ stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT = REPO / "paper" / "results" / "wsG_class009.json"


def load(window_tag: str = "") -> pl.DataFrame | None:
    p = DATA / f"surprise_class009{window_tag}.parquet"
    if not p.exists():
        return None
    return pl.read_parquet(p).with_columns(
        (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        (pl.col("kl_vs_past") - pl.col("kl_vs_future")).abs().alias("dkl_abs"),
    ).filter(
        pl.col("kl_vs_past").is_finite() & pl.col("kl_vs_future").is_finite()
        & pl.col("dkl").is_finite()
    )


def main() -> int:
    df = load()
    if df is None:
        print("MISSING data/processed/surprise_class009.parquet -- build it first:")
        print("  python -m novelty.dictionary --input data/processed/tm_class009.parquet "
              "--out data/processed/vocab_class009.parquet")
        print("  python -m novelty.surprise --records data/processed/tm_class009.parquet "
              "--vocab data/processed/vocab_class009.parquet --out data/processed/surprise_class009.parquet")
        raise SystemExit(2)

    res = {}
    n = df.height
    print(f"=== Class 009 surprise: {n:,} scored filings, "
          f"years {df['year'].min()}-{df['year'].max()} ===")
    s = df["dkl"]; a = df["dkl_abs"]
    res["n"] = n
    res["dkl_signed"] = {"mean": float(s.mean()), "sd": float(s.std()),
                         "p10": float(s.quantile(.1)), "median": float(s.median()),
                         "p90": float(s.quantile(.9)),
                         "share_positive": float((s > 0).mean())}
    res["dkl_abs"] = {"mean": float(a.mean()), "median": float(a.median())}
    res["corr_pros_retr"] = float(np.corrcoef(df["kl_vs_past"], df["kl_vs_future"])[0, 1])
    print(f"signed dKL: mean={s.mean():+.4f} sd={s.std():.4f} "
          f"share>0={(s>0).mean():.1%}  | abs dKL mean={a.mean():.4f}")
    print(f"corr(pros,retr)={res['corr_pros_retr']:+.3f}  "
          f"(high corr => signed dKL is a small difference of large numbers)")

    # per-year mean signed dKL
    yr = (df.group_by("year").agg(pl.col("dkl").mean().alias("mean_dkl"),
                                  pl.len().alias("n")).sort("year"))
    res["per_year"] = yr.to_dicts()
    print("\nper-year mean signed dKL (selected):")
    for r in yr.filter(pl.col("year") % 5 == 0).iter_rows(named=True):
        print(f"  {r['year']}: {r['mean_dkl']:+.4f}  (n={r['n']:,})")

    # face validity: most resonant / most tired. Require >=10 terms to limit the
    # length artifact (F0b) and restrict to the post-burn-in, full-window era
    # 1996-2020 (pre-1995 has thin past => spurious high resonance; post-2020 has
    # truncated future). BURN_IN visible above: 1990 mean dKL = +2.05.
    BURN_IN, EDGE = 1996, 2020
    res["faceval_window"] = [BURN_IN, EDGE]
    stable = df.filter((pl.col("n_terms") >= 10)
                       & (pl.col("year") >= BURN_IN) & (pl.col("year") <= EDGE))
    top = stable.sort("dkl", descending=True).head(10)
    bot = stable.sort("dkl").head(10)
    res["top_resonant"] = top.select("year", "mark_identification", "dkl").to_dicts()
    res["top_tired"] = bot.select("year", "mark_identification", "dkl").to_dicts()
    print("\nMost RESONANT (novel-then-adopted) marks:")
    for r in top.iter_rows(named=True):
        print(f"  {r['year']} dKL={r['dkl']:+.3f}  {str(r['mark_identification'])[:45]}")
    print("Most TIRED (obvious-then-dated) marks:")
    for r in bot.iter_rows(named=True):
        print(f"  {r['year']} dKL={r['dkl']:+.3f}  {str(r['mark_identification'])[:45]}")

    # F0b reliability: |dKL| vs n_terms
    bins = df.with_columns(
        pl.when(pl.col("n_terms") < 5).then(pl.lit("1:<5"))
        .when(pl.col("n_terms") < 15).then(pl.lit("2:5-14"))
        .when(pl.col("n_terms") < 40).then(pl.lit("3:15-39"))
        .otherwise(pl.lit("4:40+")).alias("tbin"))
    rel = bins.group_by("tbin").agg(
        pl.col("dkl_abs").mean().alias("mean_abs_dkl"),
        pl.len().alias("n")).sort("tbin")
    res["reliability_by_nterms"] = rel.to_dicts()
    print("\nF0b reliability -- mean |dKL| by filing length:")
    for r in rel.iter_rows(named=True):
        print(f"  {r['tbin']:8s} mean|dKL|={r['mean_abs_dkl']:.4f}  (n={r['n']:,})")

    # F0c half-life sensitivity: sign flips across windows
    flips = {}
    base = df.select("serial_number", pl.col("dkl").alias("dkl_W5"))
    for tag, name in [("_W3", "W3"), ("_W7", "W7")]:
        alt = load(tag)
        if alt is None:
            continue
        j = base.join(alt.select("serial_number", pl.col("dkl").alias(f"dkl_{name}")),
                      on="serial_number", how="inner")
        flip = float((np.sign(j["dkl_W5"]) != np.sign(j[f"dkl_{name}"])).mean())
        flips[f"W5_vs_{name}"] = {"flip_rate": flip, "n": j.height}
        print(f"\nF0c sign-flip W5 vs {name}: {flip:.1%}  (n={j.height:,})")
    res["halflife_signflip"] = flips
    if not flips:
        print("\nF0c: build W3/W7 surprise files to compute sign-flip rate.")

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[done] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
