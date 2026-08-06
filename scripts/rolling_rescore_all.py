"""Re-score every class with per-filing reference windows.

Same measure as topic_p_scorer_all.py, but the reference is computed at each
filing's own date rather than once per calendar year. See Appendix "Annual
reference buckets inflate the signed axis" for why this matters: annual
bucketing imprints a spurious within-year gradient on the signed axis and
inflates the gate penalty by roughly a third in the software class.

Output columns deliberately carry the SAME names as the production topic files
(topic_kl_vs_past / topic_kl_vs_future / topic_dkl) so that every downstream
script can consume them by switching one path.

    data/processed/rolling_surprise_class{CLS}{SUFFIX}.parquet

Usage:  python scripts/rolling_rescore_all.py [T ...]      (default: 50 200)
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"

CLASSES = [f"{i:03d}" for i in range(1, 46)]
WDAYS = 1826
MIN_REF = 500
YEAR_LO, YEAR_HI = 1995, 2019
EPS = 1e-12
CH = 200_000


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def window_means(theta: np.ndarray, lo: np.ndarray, hi: np.ndarray, T: int):
    """Sliding sum over monotone index bounds -> per-row window mean and count."""
    n = theta.shape[0]
    out = np.empty((n, T), dtype=np.float64)
    cnt = (hi - lo).astype(np.int64)
    run = np.zeros(T, dtype=np.float64)
    cl = ch = 0
    for i in range(n):
        while ch < hi[i]:
            run += theta[ch]; ch += 1
        while cl < lo[i]:
            run -= theta[cl]; cl += 1
        if cnt[i] > 0:
            out[i] = run / cnt[i]
        else:
            out[i] = np.nan
    return out, cnt


def kl_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Q = np.clip(Q, EPS, None)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return np.einsum("ij,ij->i", P, np.log(P) - np.log(Q))


def score_class(cls: str, T: int, vec: CountVectorizer, lda, suffix: str) -> bool:
    src = PROC / f"tm_class{cls}.parquet"
    if not src.exists():
        return False
    df = pl.read_parquet(src, columns=["serial_number", "filing_date", "goods_services"]
                         ).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd")
    ).drop_nulls("fd").sort("fd")
    n = df.height
    if n < 2 * MIN_REF:
        log(f"  [{cls}] {n:,} filings -- too few, skipped")
        return False

    theta = np.empty((n, T), dtype=np.float64)
    texts = df["goods_services"].to_list()
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
    k_past = kl_rows(theta, q_p)
    del q_p; gc.collect()
    q_f, n_f = window_means(theta, lo_f, hi_f, T)
    k_fut = kl_rows(theta, q_f)
    del q_f; gc.collect()

    years = df["fd"].dt.year().to_numpy()
    ok = ((n_p >= MIN_REF) & (n_f >= MIN_REF)
          & (days - WDAYS >= days[0]) & (days + WDAYS <= days[-1])
          & (years >= YEAR_LO) & (years <= YEAR_HI))
    k_past = np.where(ok, k_past, np.nan)
    k_fut = np.where(ok, k_fut, np.nan)

    df.select("serial_number").with_columns(
        pl.Series("topic_kl_vs_past", k_past),
        pl.Series("topic_kl_vs_future", k_fut),
        pl.Series("topic_dkl", k_past - k_fut),
        pl.Series("n_ref_past", n_p),
        pl.Series("n_ref_future", n_f),
    ).write_parquet(PROC / f"rolling_surprise_class{cls}{suffix}.parquet")

    log(f"  [{cls}] {n:,} filings, {int(ok.sum()):,} scored ({ok.mean()*100:.0f}%)")
    del theta, df, k_past, k_fut, n_p, n_f, ok
    gc.collect()
    return True


def main() -> int:
    Ts = [int(a) for a in sys.argv[1:]] or [50, 200]
    for T in Ts:
        suffix = "" if T == 50 else f"_T{T}"
        t0 = time.time()
        m = joblib.load(PROC / f"topic_model{suffix}.joblib")
        vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                              token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b",
                              ngram_range=(1, 2))
        lda = m["lda"]
        log(f"[T={T}] model loaded, vocab={len(m['vocabulary']):,}")
        done = 0
        for cls in CLASSES:
            try:
                if score_class(cls, T, vec, lda, suffix):
                    done += 1
            except Exception as exc:
                log(f"  [{cls}] FAILED: {type(exc).__name__}: {exc}")
        log(f"[T={T}] {done}/45 classes in {(time.time()-t0)/60:.1f} min")
    log("[done] rolling rescore complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
