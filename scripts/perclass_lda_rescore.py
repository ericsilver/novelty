"""Fifty themes fitted per Nice class, then scored on per-filing windows.

The production scoring fits one 50-theme model on a stratified sample of all
45 classes and takes each class's reference from its own filings. That gives
three things a reader might want to compare:

  global 50   one model, references per class        (production)
  global 500  one model at ten times the resolution  (rolling_rescore_all.py 500)
  per-class 50  a separate 50-theme model per class, fitted on that class alone

This script builds the third. For each class it draws up to SAMPLE filings from
that class, fits a 50-theme LDA with the same vectorizer and passes as the
production fit, transforms every filing in the class, and scores it against
per-filing windows exactly as rolling_rescore_all.py does. The themes are not
comparable across classes -- theme 7 in class 009 has nothing to do with theme
7 in class 025 -- which is the point: the question is whether a class-specific
partition changes what atypicality and lead measure inside a class.

Writes  data/processed/perclass_surprise_class{CLS}.parquet   (production column names)
        data/processed/perclass_model_class{CLS}.joblib        (vocabulary + LDA + top words)

Usage:  python scripts/perclass_lda_rescore.py [CLASS ...]    default: all 45
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
sys.path.insert(0, str(REPO / "scripts"))
from rolling_rescore_all import window_means, kl_rows, WDAYS, MIN_REF, YEAR_LO, YEAR_HI, EPS, CH  # noqa: E402

CLASSES = sys.argv[1:] or [f"{i:03d}" for i in range(1, 46)]
T = 50
SAMPLE = 60_000
MIN_DF = 20
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
SEED = 42


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def main() -> int:
    for cls in CLASSES:
        out = PROC / f"perclass_surprise_class{cls}.parquet"
        if out.exists():
            log(f"  [{cls}] exists, skipped")
            continue
        src = PROC / f"tm_class{cls}.parquet"
        if not src.exists():
            continue
        t0 = time.time()
        df = pl.read_parquet(src, columns=["serial_number", "filing_date", "goods_services"]).filter(
            pl.col("goods_services").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        ).with_columns(pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd")
                       ).drop_nulls("fd").sort("fd")
        n = df.height
        if n < 2 * MIN_REF:
            log(f"  [{cls}] {n:,} filings -- too few, skipped")
            continue
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(n, size=min(SAMPLE, n), replace=False))
        texts = df["goods_services"].to_list()
        vec = CountVectorizer(lowercase=True, token_pattern=TOKEN, ngram_range=(1, 2),
                              min_df=MIN_DF, max_features=60_000)
        Xs = vec.fit_transform([texts[i] for i in idx])
        lda = LatentDirichletAllocation(n_components=T, max_iter=8, learning_method="online",
                                        batch_size=4096, random_state=SEED, n_jobs=4)
        lda.fit(Xs)
        vocab = np.array(vec.get_feature_names_out())
        top_words = {str(k): [str(w) for w in vocab[np.argsort(-lda.components_[k])[:12]]]
                     for k in range(T)}
        joblib.dump({"vocabulary": vec.vocabulary_, "lda": lda, "top_words": top_words,
                     "n_sample": int(len(idx)), "V": int(len(vocab))},
                    PROC / f"perclass_model_class{cls}.joblib")
        del Xs

        theta = np.empty((n, T), dtype=np.float64)
        for s in range(0, n, CH):
            e = min(s + CH, n)
            theta[s:e] = lda.transform(vec.transform(texts[s:e]))
        del texts
        np.clip(theta, EPS, None, out=theta)
        theta /= theta.sum(axis=1, keepdims=True)

        days = df["fd"].to_numpy().astype("datetime64[D]").astype(np.int64)
        lo_p = np.searchsorted(days, days - WDAYS, side="left")
        hi_p = np.searchsorted(days, days, side="left")
        lo_f = np.searchsorted(days, days, side="right")
        hi_f = np.searchsorted(days, days + WDAYS, side="right")
        q_p, n_p = window_means(theta, lo_p, hi_p, T)
        k_past = kl_rows(theta, q_p); del q_p; gc.collect()
        q_f, n_f = window_means(theta, lo_f, hi_f, T)
        k_fut = kl_rows(theta, q_f); del q_f; gc.collect()
        years = df["fd"].dt.year().to_numpy()
        ok = ((n_p >= MIN_REF) & (n_f >= MIN_REF) & (days - WDAYS >= days[0])
              & (days + WDAYS <= days[-1]) & (years >= YEAR_LO) & (years <= YEAR_HI))
        k_past = np.where(ok, k_past, np.nan); k_fut = np.where(ok, k_fut, np.nan)
        df.select("serial_number").with_columns(
            pl.Series("year", years),
            pl.Series("topic_kl_vs_past", k_past), pl.Series("topic_kl_vs_future", k_fut),
            pl.Series("topic_dkl", k_past - k_fut),
            pl.Series("n_ref_past", n_p), pl.Series("n_ref_future", n_f),
        ).write_parquet(out)
        log(f"  [{cls}] {n:,} filings, V={len(vocab):,}, {int(ok.sum()):,} scored, {time.time()-t0:.0f}s")
        del theta, df, k_past, k_fut
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
