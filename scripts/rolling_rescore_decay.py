"""Re-score all 45 Nice classes with an exponentially weighted per-filing window.

The production scorer (rolling_rescore_all.py) treats every day inside the
five-year window as equally informative: language the class adopted five years
ago and language it adopted last month are equally familiar to the reader. That
is a real modelling choice, and an earlier build of this project used the other
one -- an exponential-decay reference with a two-year half-life -- so the
robustness question is whether the paper's claims are properties of the measure
or of that flatness.

This is the decayed variant, and it differs from the legacy decay build in one
important way: the kernel is anchored on the filing's own date, not on a
calendar year. Recent language is weighted up relative to distant language
*within* each filing's own window, so the comparison to the flat scorer isolates
the weighting and nothing else. Anchoring is never varied.

    Q_past(i)   proportional to  sum_j 2^{-(d_i - d_j)/H} theta_j
                over  d_i - W <= d_j < d_i
    Q_future(i) proportional to  sum_j 2^{-(d_j - d_i)/H} theta_j
                over  d_i < d_j <= d_i + W

with H the half-life (default two years) and W the same 1,826-day truncation
the flat scorer uses. Setting H to infinity recovers the flat scorer exactly.

Implementation. The anchor factor 2^{-d_i/H} cancels between numerator and
denominator, so each side reduces to a running weighted sum over a sliding
window -- the same two-pointer pass as the flat version, carrying weights. The
reference date for those weights is rebased every few thousand filings, because
a single global reference makes the weights span the age of the corpus (forty
orders of magnitude, since class files reach back to the 1880s) and destroys
the running sums. The forward window is computed by reversing and negating the
date series and reusing the backward routine, so both sides take the identical
numerical path.

The MIN_REF floor becomes a floor on Kish effective sample size,
(sum w)^2 / sum w^2, which is the right analogue: a window holding a thousand
filings whose weight is concentrated on ten is not a thousand-filing reference.

Reads   data/processed/tm_class{CLS}.parquet
        data/processed/topic_model.joblib
Writes  data/processed/decay_surprise_class{CLS}.parquet

Column names match the flat scorer's, so SURPRISE_SRC=decay switches every
downstream consumer over without further change.

Usage:  python scripts/rolling_rescore_decay.py [HALFLIFE_YEARS]   (default: 2)
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
MIN_EFF = 500.0
YEAR_LO, YEAR_HI = 1995, 2019
T = 50
EPS = 1e-12
CH = 200_000

HALFLIFE_YEARS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
HDAYS = HALFLIFE_YEARS * 365.25


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)




def weighted_past(theta: np.ndarray, days: np.ndarray, hdays: float,
                  lo: np.ndarray, hi: np.ndarray, T: int):
    """Decay-weighted mean of theta over the backward window [lo, hi).

    Weights fall by half every `hdays` of distance back from the anchor. The
    naive implementation -- fix one global reference date and give filing j the
    weight 2^{(d_j - base)/H} once -- is numerically unusable here: a class file
    reaches back to the 1880s, so those weights span forty orders of magnitude
    and a running sum that adds large terms and later subtracts small ones loses
    every significant digit. (It fails silently, as an effective sample size
    that goes negative and a window that scores nothing.)

    Instead the reference date is rebased whenever the anchor has drifted a
    half-life from it. Rebasing on elapsed time rather than on a filing count
    matters: a fixed number of filings spans decades in the sparse early years,
    which would make the rescale factor itself astronomically large. Between
    rebases all live weights sit within about 2^{W/H + 1} of one another, and at
    each rebase the accumulators and the stored per-item weights are scaled by a
    single common factor, exact up to one rounding. The anchor cancels out of the
    normalized mean, so its absolute value never matters.

    Returns the weighted mean and the Kish effective sample size (sum w)^2 /
    sum w^2, which is scale-invariant and so is unaffected by rebasing.
    """
    n = theta.shape[0]
    out = np.empty((n, T), dtype=np.float64)
    neff = np.zeros(n, dtype=np.float64)
    run = np.zeros(T, dtype=np.float64)
    wst = np.zeros(n, dtype=np.float64)   # weight each live row was added with
    s1 = 0.0
    s2 = 0.0
    cl = ch = 0
    base = float(days[0])
    for i in range(n):
        if abs(days[i] - base) >= hdays:
            nb = float(days[i])
            f = 2.0 ** ((base - nb) / hdays)
            run *= f
            s1 *= f
            s2 *= f * f
            if ch > cl:
                wst[cl:ch] *= f
            base = nb
        while ch < hi[i]:
            w = 2.0 ** ((days[ch] - base) / hdays)
            wst[ch] = w
            run += w * theta[ch]
            s1 += w
            s2 += w * w
            ch += 1
        while cl < lo[i]:
            w = wst[cl]
            run -= w * theta[cl]
            s1 -= w
            s2 -= w * w
            cl += 1
        if s1 > 0 and s2 > 0:
            out[i] = run / s1
            neff[i] = (s1 * s1) / s2
        else:
            out[i] = np.nan
    return out, neff


def kl_rows(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    Q = np.clip(Q, EPS, None)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return np.einsum("ij,ij->i", P, np.log(P) - np.log(Q))


def score_class(cls: str, lda, vocab) -> bool:
    tp = PROC / f"tm_class{cls}.parquet"
    if not tp.exists():
        return False
    df = pl.read_parquet(
        tp, columns=["serial_number", "filing_date", "goods_services"]).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd")
    ).drop_nulls("fd").sort("fd")
    n = df.height
    if n < 2 * int(MIN_EFF):
        log(f"  [{cls}] {n:,} filings -- too few, skipped")
        return False

    vec = CountVectorizer(vocabulary=vocab, lowercase=True,
                          token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b", ngram_range=(1, 2))
    theta = np.empty((n, T), dtype=np.float64)
    texts = df["goods_services"].to_list()
    for s in range(0, n, CH):
        e = min(s + CH, n)
        theta[s:e] = lda.transform(vec.transform(texts[s:e]))
    del texts
    np.clip(theta, EPS, None, out=theta)
    theta /= theta.sum(axis=1, keepdims=True)

    days = df["fd"].to_numpy().astype("datetime64[D]").astype(np.int64)
    years = df["fd"].dt.year().to_numpy()

    lo_p = np.searchsorted(days, days - WDAYS, side="left")
    hi_p = np.searchsorted(days, days, side="left")
    q_p, eff_p = weighted_past(theta, days, HDAYS, lo_p, hi_p, T)
    k_past = kl_rows(theta, q_p)
    del q_p
    gc.collect()

    # The forward window is the backward window of the reversed, negated series:
    # -d_i - W <= -d_j < -d_i is exactly d_i < d_j <= d_i + W. Running it through
    # the same routine keeps both sides on the identical, well-conditioned path.
    rt = np.ascontiguousarray(theta[::-1])
    rd = -days[::-1]
    lo_r = np.searchsorted(rd, rd - WDAYS, side="left")
    hi_r = np.searchsorted(rd, rd, side="left")
    q_r, eff_r = weighted_past(rt, rd, HDAYS, lo_r, hi_r, T)
    k_fut = kl_rows(rt, q_r)[::-1].copy()
    eff_f = eff_r[::-1].copy()
    del q_r, rt, eff_r
    gc.collect()

    ok = ((eff_p >= MIN_EFF) & (eff_f >= MIN_EFF)
          & (days - WDAYS >= days[0]) & (days + WDAYS <= days[-1])
          & (years >= YEAR_LO) & (years <= YEAR_HI))
    k_past = np.where(ok, k_past, np.nan)
    k_fut = np.where(ok, k_fut, np.nan)

    df.select("serial_number").with_columns(
        pl.Series("year", years),
        pl.Series("topic_kl_vs_past", k_past),
        pl.Series("topic_kl_vs_future", k_fut),
        pl.Series("topic_dkl", k_past - k_fut),
        pl.Series("n_ref_past", eff_p),
        pl.Series("n_ref_future", eff_f),
    ).write_parquet(PROC / f"decay_surprise_class{cls}.parquet")

    log(f"  [{cls}] {n:,} filings, {int(ok.sum()):,} scored ({ok.mean()*100:.0f}%)")
    del theta, df, k_past, k_fut, eff_p, eff_f, ok, days, rd
    gc.collect()
    return True


def main() -> int:
    log(f"[decay] half-life {HALFLIFE_YEARS}y ({HDAYS:.0f}d), "
        f"window +/-{WDAYS}d, T={T}")
    m = joblib.load(PROC / "topic_model.joblib")
    lda, vocab = m["lda"], m["vocabulary"]
    t0 = time.time()
    done = 0
    for c in CLASSES:
        if score_class(c, lda, vocab):
            done += 1
    log(f"[done] {done} classes in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
