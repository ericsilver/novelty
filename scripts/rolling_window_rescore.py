"""Re-score a class with a per-filing reference window instead of a per-year one.

The production scorer builds one reference per calendar year: it averages the
class's topic distributions over the five preceding CALENDAR YEARS and scores
every filing made in year y against that single object. A filing made on
1 January and one made on 31 December of the same year are therefore scored
against exactly the same reference, and neither sees any of the language filed
during its own year.

This script replaces that with a reference computed at each filing's own date:
the past window is the 1,826 days strictly before it, the future window the
1,826 days strictly after. Windows slide continuously, so a December filer's
reference contains the eleven months of within-year language that the yearly
scorer discards, and same-day filings are excluded from both sides.

Implementation is a two-pointer sliding sum over date-sorted filings, so cost is
O(n*T) with O(T) memory rather than a full prefix-sum matrix.

Usage:  python scripts/rolling_window_rescore.py [CLASS] [T]
        (defaults: 009, 200)

Output: data/processed/rolling_surprise_class{CLS}_T{T}.parquet
        paper/results/rolling_window_comparison.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

CLS = sys.argv[1] if len(sys.argv) > 1 else "009"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WDAYS = 1826            # 5 years
MIN_REF = 500           # filings required in each window for a score to stand
YEAR_LO, YEAR_HI = 1995, 2019
EPS = 1e-12


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def main() -> int:
    t_start = time.time()

    df = pl.read_parquet(
        PROC / f"tm_class{CLS}.parquet",
        columns=["serial_number", "filing_date", "goods_services"],
    ).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd")
    ).drop_nulls("fd").sort("fd")
    n = df.height
    log(f"[load] {n:,} filings in class {CLS}, {df['fd'].min()} to {df['fd'].max()}")

    suffix = "" if T == 50 else f"_T{T}"
    m = joblib.load(PROC / f"topic_model{suffix}.joblib")
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b", ngram_range=(1, 2))
    lda = m["lda"]

    t0 = time.time()
    theta = np.empty((n, T), dtype=np.float64)
    texts = df["goods_services"].to_list()
    CH = 200_000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        theta[s:e] = lda.transform(vec.transform(texts[s:e]))
        log(f"  [transform] {e:,}/{n:,}")
    del texts
    theta = np.clip(theta, EPS, None)
    theta /= theta.sum(axis=1, keepdims=True)
    log(f"[theta] {theta.shape} in {time.time()-t0:.0f}s")

    days = df["fd"].to_numpy().astype("datetime64[D]").astype(np.int64)

    # Window bounds. Past is strictly before the filing's own date; future is
    # strictly after it. Same-date filings enter neither, so a filing is never
    # scored against itself or its exact contemporaries.
    lo_p = np.searchsorted(days, days - WDAYS, side="left")
    hi_p = np.searchsorted(days, days, side="left")
    lo_f = np.searchsorted(days, days, side="right")
    hi_f = np.searchsorted(days, days + WDAYS, side="right")

    def window_means(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sliding sum over monotone index bounds; returns means and counts."""
        out = np.empty((n, T), dtype=np.float64)
        cnt = (hi - lo).astype(np.int64)
        run = np.zeros(T, dtype=np.float64)
        cl, ch = 0, 0
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

    t0 = time.time()
    q_p, n_p = window_means(lo_p, hi_p)
    log(f"[past window] {time.time()-t0:.0f}s")
    t0 = time.time()
    q_f, n_f = window_means(lo_f, hi_f)
    log(f"[future window] {time.time()-t0:.0f}s")

    def kl(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        Q = np.clip(Q, EPS, None)
        Q = Q / Q.sum(axis=1, keepdims=True)
        return np.einsum("ij,ij->i", P, np.log(P) - np.log(Q))

    k_past = kl(theta, q_p)
    k_fut = kl(theta, q_f)

    years = (df["fd"].dt.year()).to_numpy()
    dmin, dmax = days[0], days[-1]
    ok = ((n_p >= MIN_REF) & (n_f >= MIN_REF)
          & (days - WDAYS >= dmin) & (days + WDAYS <= dmax)
          & (years >= YEAR_LO) & (years <= YEAR_HI))
    k_past = np.where(ok, k_past, np.nan)
    k_fut = np.where(ok, k_fut, np.nan)
    log(f"[scored] {int(ok.sum()):,} of {n:,} filings ({ok.mean()*100:.1f}%)")

    out = df.select("serial_number", "fd").with_columns(
        pl.Series("roll_kl_vs_past", k_past),
        pl.Series("roll_kl_vs_future", k_fut),
        pl.Series("roll_dkl", k_past - k_fut),
        pl.Series("roll_n_past", n_p),
        pl.Series("roll_n_future", n_f),
    )
    dest = PROC / f"rolling_surprise_class{CLS}_T{T}.parquet"
    out.write_parquet(dest)
    log(f"[write] {dest.name}")

    # ---- compare against the production per-year scores ---------------------
    yr = pl.read_parquet(
        PROC / f"topic_surprise_class{CLS}_T{T}.parquet",
        columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"],
    )
    j = out.join(yr, on="serial_number", how="inner").drop_nulls(
        ["roll_dkl", "topic_dkl"]).filter(
        pl.col("roll_dkl").is_finite() & pl.col("topic_dkl").is_finite())
    log(f"[compare] {j.height:,} filings scored both ways")

    def stats(a: str, b: str) -> dict:
        x = j[a].to_numpy(); y = j[b].to_numpy()
        return {"pearson": float(np.corrcoef(x, y)[0, 1]),
                "spearman": float(np.corrcoef(
                    np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]),
                "sd_rolling": float(x.std()), "sd_yearly": float(y.std()),
                "mean_rolling": float(x.mean()), "mean_yearly": float(y.mean())}

    cmp = {"class": CLS, "T": T, "window_days": WDAYS, "min_ref": MIN_REF,
           "n_compared": int(j.height),
           "dkl": stats("roll_dkl", "topic_dkl"),
           "kl_past": stats("roll_kl_vs_past", "topic_kl_vs_past"),
           "kl_future": stats("roll_kl_vs_future", "topic_kl_vs_future")}

    # Does the within-year position that the yearly scorer cannot see carry
    # anything? Compare filings by month-of-year under each scoring.
    j2 = j.with_columns(pl.col("fd").dt.month().alias("mo"))
    by_mo = j2.group_by("mo").agg(
        pl.col("roll_dkl").mean().alias("roll"),
        pl.col("topic_dkl").mean().alias("yearly"),
        pl.len().alias("n")).sort("mo")
    cmp["by_month"] = by_mo.to_dicts()

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "rolling_window_comparison.json").write_text(json.dumps(cmp, indent=1))
    log(f"[done] total {time.time()-t_start:.0f}s")
    log(f"  dKL  rolling vs yearly: r={cmp['dkl']['pearson']:.4f} "
        f"sd {cmp['dkl']['sd_rolling']:.4f} vs {cmp['dkl']['sd_yearly']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
