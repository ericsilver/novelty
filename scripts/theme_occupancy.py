"""How many themes does a single Nice class actually use, and at what T?

Two questions about the theme count that the gate contrast cannot answer.

First, whether raising T decouples the two KL levels. Past- and future-facing
surprise correlate at about 0.99, which is what makes lead a small residual of
two large quantities, and a plausible hope is that a finer partition separates
them: with 2,500 themes rather than 50, a filing's past and future references
would differ in more coordinates. This measures it directly.

Second, whether raising T gives a class more themes to work with. A theme
fitted across all 45 classes may be empty inside any one of them, so the
relevant quantity is not T but how many themes carry real mass in the class
being scored, and how many filings each is the dominant theme for. That is the
occupancy criterion a reader would want before believing a finer partition
resolves "minor themes" rather than splitting the same mass into slivers.

Reported per resolution, on one class:

  themes >= 1% mass   themes carrying at least a hundredth of the class's
                      total topic mass -- the themes the class actually uses
  docs as top         how many filings each theme is the argmax for; the
                      minimum and median across themes
  effective themes    exp(entropy of the filing's theme distribution),
                      averaged: how many themes a typical filing spreads over

Usage:  python scripts/theme_occupancy.py [CLASS] [N_DOCS]
Output: paper/results/theme_occupancy.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

CLS = sys.argv[1] if len(sys.argv) > 1 else "009"
NDOCS = int(sys.argv[2]) if len(sys.argv) > 2 else 120_000
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
MODELS = [(50, "topic_model.joblib"), (100, "topic_model_T100.joblib"),
          (200, "topic_model_T200.joblib"), (500, "topic_model_T500.joblib"),
          (1000, "topic_model_T1000.joblib"), (2500, "topic_model_T2500.joblib")]


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def score_path(T: int) -> Path:
    return PROC / ("rolling_surprise_class%s.parquet" % CLS if T == 50
                   else f"rolling_surprise_class{CLS}_T{T}.parquet")


def main() -> int:
    raw = pl.read_parquet(
        PROC / f"tm_class{CLS}.parquet",
        columns=["serial_number", "filing_date", "goods_services"]).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
    texts = raw["goods_services"].head(NDOCS).to_list()
    log(f"[occupancy] class {CLS}, {len(texts):,} filings")

    out = {"class": CLS, "n_docs": len(texts), "by_T": {}}
    log(f"\n{'T':>5} {'used':>6} {'min':>6} {'med':>7} {'eff':>7} "
        f"{'corr(K-,K+)':>12} {'sd(L)':>8} {'mean(A)':>8}")
    for T, mp in MODELS:
        f = PROC / mp
        if not f.exists():
            continue
        m = joblib.load(f)
        vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                              token_pattern=TOKEN, ngram_range=(1, 2))
        th = m["lda"].transform(vec.transform(texts))
        np.clip(th, 1e-12, None, out=th)
        th /= th.sum(axis=1, keepdims=True)
        cnt = np.bincount(th.argmax(axis=1), minlength=T)
        mass = th.mean(axis=0)
        eff = float(np.exp(-(th * np.log(th)).sum(axis=1)).mean())
        row = {"themes_ge_1pct_mass": int((mass >= 0.01).sum()),
               "min_docs_as_top": int(cnt.min()),
               "median_docs_as_top": int(np.median(cnt)),
               "themes_with_zero_docs_as_top": int((cnt == 0).sum()),
               "effective_themes_per_filing": eff}
        del m, th, vec
        gc.collect()

        sp = score_path(T)
        if sp.exists():
            d = pl.read_parquet(sp, columns=["topic_kl_vs_past",
                                             "topic_kl_vs_future"]).filter(
                pl.col("topic_kl_vs_past").is_finite()
                & pl.col("topic_kl_vs_future").is_finite())
            kp = d["topic_kl_vs_past"].to_numpy()
            kf = d["topic_kl_vs_future"].to_numpy()
            A, L = 0.5 * (kp + kf), kp - kf
            row.update({
                "n_scored": int(len(kp)),
                "corr_Kpast_Kfuture": float(np.corrcoef(kp, kf)[0, 1]),
                "sd_L": float(L.std()), "sd_A": float(A.std()),
                "mean_A": float(A.mean()),
                "var_share_in_L": float(L.var() / (kp.var() + kf.var()))})
            del d, kp, kf
            gc.collect()
        out["by_T"][str(T)] = row
        log(f"{T:>5} {row['themes_ge_1pct_mass']:>6} {row['min_docs_as_top']:>6} "
            f"{row['median_docs_as_top']:>7} {row['effective_themes_per_filing']:>7.2f} "
            f"{row.get('corr_Kpast_Kfuture', float('nan')):>12.4f} "
            f"{row.get('sd_L', float('nan')):>8.4f} "
            f"{row.get('mean_A', float('nan')):>8.3f}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "theme_occupancy.json").write_text(json.dumps(out, indent=1))
    log("\n[done] theme_occupancy.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
