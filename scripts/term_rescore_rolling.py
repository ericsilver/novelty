"""Re-score all 45 Nice classes on TERMS with per-filing reference windows.

Term scoring was the one part of the paper still computed against class-year
references: a filing made in January and one made in December were scored
against the same object, and neither could see any of the language filed during
its own year. Every topic-scored estimate moved to per-filing windows; this
brings term scoring across so that the two representations differ only in the
representation.

Why this is cheap despite the vocabulary being large. A goods/services
description runs to a few dozen words, so the filing's own distribution P is
sparse, and

    KL(P || Q) = sum_{t : P_t > 0} P_t (log P_t - log Q_t)

only needs Q at the terms the filing actually uses. The reference Q is
maintained as a dense running count vector over the class vocabulary, updated by
adding and subtracting sparse rows as two pointers sweep the date-sorted
filings, and read at a few dozen positions per filing. Each filing enters and
leaves each accumulator exactly once.

Two kernels are available and share everything else:

  flat    every day inside the +/-1,826-day window counts the same. This is
          the production convention and matches rolling_rescore_all.py.
  decay   weights fall by half every HALFLIFE years of distance from the
          filing's own date, still truncated at the same window. Anchoring is
          identical, so the flat/decay comparison isolates the weighting.

Q is Dirichlet-smoothed with alpha = 1 over the class vocabulary, matching the
earlier class-year term construction, so the only change is the anchoring.

Reads   data/processed/tm_class{CLS}.parquet
Writes  data/processed/termroll_surprise_class{CLS}.parquet          (flat)
        data/processed/termrolldecay_surprise_class{CLS}.parquet     (decay)

Columns are named topic_kl_vs_past / topic_kl_vs_future / topic_dkl despite
being term-scored, so that SURPRISE_SRC=termroll switches every downstream
consumer over without further change. The name is a wire format, not a claim
about the representation.

Usage:  python scripts/term_rescore_rolling.py [flat|decay] [HALFLIFE_YEARS]
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"

CLASSES = [f"{i:03d}" for i in range(1, 46)]
WDAYS = 1826
MIN_REF = 500          # filings required in each window (flat)
MIN_EFF = 500.0        # Kish effective size required in each window (decay)
MIN_TERM_FREQ = 50     # class-level term frequency floor
MIN_TERMS = 3          # filings with fewer in-vocabulary terms are nulled
YEAR_LO, YEAR_HI = 1995, 2019
ALPHA = 1.0            # Dirichlet smoothing on the reference
EPS = 1e-12

KERNEL = sys.argv[1] if len(sys.argv) > 1 else "flat"
HALFLIFE_YEARS = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
HDAYS = HALFLIFE_YEARS * 365.25
PREFIX = "termroll" if KERNEL == "flat" else "termrolldecay"

# The weight reference is rebased once the anchor drifts this far, so the
# rescale factor stays near unity. Rebasing on a filing count instead would
# span decades in the sparse early years and blow the accumulators up.
REBASE_DAYS = HDAYS


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def score_class(cls: str) -> bool:
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
    if n < 2 * MIN_REF:
        log(f"  [{cls}] {n:,} filings -- too few, skipped")
        return False

    vec = CountVectorizer(lowercase=True, min_df=MIN_TERM_FREQ,
                          token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b")
    X = vec.fit_transform(df["goods_services"].to_list()).tocsr().astype(np.float64)
    V = X.shape[1]
    if V < 50:
        log(f"  [{cls}] vocabulary {V} too small, skipped")
        return False

    days = df["fd"].to_numpy().astype("datetime64[D]").astype(np.int64)
    years = df["fd"].dt.year().to_numpy()

    lo_p = np.searchsorted(days, days - WDAYS, side="left")
    hi_p = np.searchsorted(days, days, side="left")
    lo_f = np.searchsorted(days, days, side="right")
    hi_f = np.searchsorted(days, days + WDAYS, side="right")

    nterms = np.diff(X.indptr)

    def one_side(X_, days_, lo, hi):
        """KL of each filing against its BACKWARD window, and that window's size.

        Only the backward direction is implemented, because it is the numerically
        well-behaved one: weights of live rows rise toward the anchor and the
        oldest, smallest-weight rows are the ones that expire. Sweeping forward
        with mirrored weights instead makes the accumulators grow at every rebase
        and lose all precision -- an effective sample size of 1e-13 and a window
        that scores nothing. The forward window is obtained by calling this on
        the reversed, negated series, where it *is* a backward window.

        The weight reference is rebased once the anchor drifts a half-life, so
        live weights stay within about 2^(W/H + 1) of one another; a single
        global reference would span the age of the class file.
        """
        ip_, ix_, dt_ = X_.indptr, X_.indices, X_.data
        rs_ = np.asarray(X_.sum(axis=1)).ravel()
        nt_ = np.diff(ip_)
        kl = np.full(n, np.nan)
        eff = np.zeros(n)
        acc = np.zeros(V, dtype=np.float64)
        wst = np.zeros(n, dtype=np.float64)
        s1 = 0.0
        s2 = 0.0
        cl = ch = 0
        base = float(days_[0])
        decay = KERNEL == "decay"
        for i in range(n):
            if decay and days_[i] - base >= REBASE_DAYS:
                nb = float(days_[i])
                f = 2.0 ** ((base - nb) / HDAYS)
                acc *= f
                s1 *= f
                s2 *= f * f
                if ch > cl:
                    wst[cl:ch] *= f
                base = nb
            while ch < hi[i]:
                w = 2.0 ** ((days_[ch] - base) / HDAYS) if decay else 1.0
                wst[ch] = w
                sl = slice(ip_[ch], ip_[ch + 1])
                np.add.at(acc, ix_[sl], w * dt_[sl])
                s1 += w; s2 += w * w
                ch += 1
            while cl < lo[i]:
                w = wst[cl]
                sl = slice(ip_[cl], ip_[cl + 1])
                np.add.at(acc, ix_[sl], -w * dt_[sl])
                s1 -= w; s2 -= w * w
                cl += 1
            if s1 <= 0 or nt_[i] < MIN_TERMS or rs_[i] <= 0:
                continue
            eff[i] = (s1 * s1) / s2 if s2 > 0 else 0.0
            sl = slice(ip_[i], ip_[i + 1])
            idx = ix_[sl]
            p = dt_[sl] / rs_[i]
            tot = acc.sum() + ALPHA * V
            q = (acc[idx] + ALPHA) / tot
            kl[i] = float(np.dot(p, np.log(np.clip(p, EPS, None)) - np.log(q)))
        return kl, eff

    k_past, eff_p = one_side(X, days, lo_p, hi_p)

    # The forward window is the backward window of the reversed, negated series:
    # -d_i - W <= -d_j < -d_i is exactly d_i < d_j <= d_i + W.
    Xr = X[::-1]
    dr = -days[::-1]
    lo_r = np.searchsorted(dr, dr - WDAYS, side="left")
    hi_r = np.searchsorted(dr, dr, side="left")
    k_fut_r, eff_f_r = one_side(Xr, dr, lo_r, hi_r)
    k_fut = k_fut_r[::-1].copy()
    eff_f = eff_f_r[::-1].copy()
    del Xr, k_fut_r, eff_f_r
    gc.collect()

    floor = MIN_EFF if KERNEL == "decay" else float(MIN_REF)
    ok = ((eff_p >= floor) & (eff_f >= floor)
          & (days - WDAYS >= days[0]) & (days + WDAYS <= days[-1])
          & (years >= YEAR_LO) & (years <= YEAR_HI)
          & np.isfinite(k_past) & np.isfinite(k_fut))
    k_past = np.where(ok, k_past, np.nan)
    k_fut = np.where(ok, k_fut, np.nan)

    df.select("serial_number").with_columns(
        pl.Series("year", years),
        pl.Series("topic_kl_vs_past", k_past),
        pl.Series("topic_kl_vs_future", k_fut),
        pl.Series("topic_dkl", k_past - k_fut),
        pl.Series("n_ref_past", eff_p),
        pl.Series("n_ref_future", eff_f),
        pl.Series("n_terms", nterms.astype(np.int32)),
    ).write_parquet(PROC / f"{PREFIX}_surprise_class{cls}.parquet")

    log(f"  [{cls}] {n:,} filings, V={V:,}, {int(ok.sum()):,} scored ({ok.mean()*100:.0f}%)")
    del X, df, k_past, k_fut, eff_p, eff_f, ok
    gc.collect()
    return True


def main() -> int:
    log(f"[term] kernel={KERNEL}"
        + (f" halflife={HALFLIFE_YEARS}y" if KERNEL == "decay" else "")
        + f", window +/-{WDAYS}d, alpha={ALPHA}")
    t0 = time.time()
    done = 0
    for c in CLASSES:
        if score_class(c):
            done += 1
    log(f"[done] {done} classes in {(time.time()-t0)/60:.1f} min -> {PREFIX}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
