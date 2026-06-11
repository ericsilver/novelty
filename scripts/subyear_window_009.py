"""Sub-year reference windows, class 009: does being early by months matter?

The yearly scorer's most recent reference data ends a full calendar year
before the filing. Here references are monthly:

  pros_near = KL(P || aggregate of filings in months [t-12, t-1])
  pros_far  = KL(P || aggregate of filings in months [t-24, t-13])
  freshness = pros_far - pros_near
      > 0: the filing resembles the very recent months more than the year
           before them -- it rides a wave that emerged within the last year
      < 0: unusual even against last month -- first-mover at month resolution

Outcomes: registration completion (filings 1995-2018) and the first
Section 8 gate (registrations 2016-2018), quintile rates per variable.

Output: paper/results/subyear_window_009.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

PASS = {"701", "702", "703", "704", "705", "800"}
FAIL = {"710"}
ALPHA = 1.0
BASE_YEAR = 1990


def quintile_rates(vals: np.ndarray, outcome: np.ndarray) -> dict:
    cuts = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(cuts, vals)
    d = {}
    for q in range(5):
        m = qi == q
        d[f"q{q+1}"] = float(outcome[m].mean()) if m.any() else None
        d[f"q{q+1}_n"] = int(m.sum())
    d["lift_q5_q1"] = d["q5"] - d["q1"]
    d["inverseU_depth"] = d["q3"] - 0.5 * (d["q1"] + d["q5"])
    return d


def main() -> int:
    t0 = time.time()
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer

    vocab = pl.read_parquet(PROC / "vocab_class009.parquet")
    term_col = "term" if "term" in vocab.columns else vocab.columns[0]
    terms = vocab[term_col].to_list()
    V = len(terms)
    print(f"[vocab] {V:,} terms", file=sys.stderr, flush=True)

    tm = pl.read_parquet(
        PROC / "tm_class009.parquet",
        columns=["serial_number", "filing_date", "goods_services",
                 "registration_date", "status_code"],
    ).filter(
        (pl.col("goods_services").str.len_chars() > 0)
        & (pl.col("filing_date").str.len_chars() == 8)
    ).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        pl.col("filing_date").str.slice(4, 2).cast(pl.Int32, strict=False).alias("fm"),
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8
         ).alias("registered"),
        pl.col("registration_date").str.slice(0, 4)
          .cast(pl.Int32, strict=False).alias("reg_year"),
        pl.col("status_code").is_in(list(PASS)).alias("st_pass"),
        pl.col("status_code").is_in(list(FAIL)).alias("st_fail"),
    ).filter(pl.col("fy").is_between(BASE_YEAR, 2018)
             & pl.col("fm").is_between(1, 12))
    tm = tm.with_columns(
        ((pl.col("fy") - BASE_YEAR) * 12 + pl.col("fm") - 1).alias("midx"))
    n = tm.height
    print(f"[load] {n:,} filings {BASE_YEAR}-2018 ({time.time()-t0:.0f}s)",
          file=sys.stderr, flush=True)

    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    vec = CountVectorizer(analyzer=analyzer, vocabulary=terms)
    X = vec.transform(tm["goods_services"].to_list()).tocsr()
    print(f"[vec ] X={X.shape} nnz={X.nnz:,} ({time.time()-t0:.0f}s)",
          file=sys.stderr, flush=True)

    midx = tm["midx"].to_numpy()
    n_months = int(midx.max()) + 1

    # monthly aggregate term counts
    rowsum = np.asarray(X.sum(axis=1)).ravel()
    keep = rowsum >= 3  # n_terms >= 3 equivalent on vocab terms
    M = np.zeros((n_months, V))
    for m in range(n_months):
        rows = np.where(midx == m)[0]
        if rows.size:
            M[m] = np.asarray(X[rows].sum(axis=0)).ravel()
    csum = np.vstack([np.zeros((1, V)), np.cumsum(M, axis=0)])  # csum[m] = sum months < m
    print(f"[agg ] {n_months} months ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    # per-filing sum p log p
    Xl = X.copy()
    Xl.data = Xl.data * np.log(Xl.data)
    sum_clogc = np.asarray(Xl.sum(axis=1)).ravel()
    del Xl
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = sum_clogc / np.maximum(rowsum, 1) - np.log(np.maximum(rowsum, 1))

    pros_near = np.full(n, np.nan)
    pros_far = np.full(n, np.nan)
    first_m = 24
    for m in range(first_m, n_months):
        rows = np.where((midx == m) & keep)[0]
        if not rows.size:
            continue
        q_near = csum[m] - csum[m - 12]          # months [m-12, m-1]
        q_far = csum[m - 12] - csum[m - 24]      # months [m-24, m-13]
        for q, store in ((q_near, pros_near), (q_far, pros_far)):
            qs = (q + ALPHA) / (q.sum() + ALPHA * V)
            logq = np.log(qs)
            dot = X[rows] @ logq
            store[rows] = plogp[rows] - dot / rowsum[rows]
        if m % 60 == 0:
            print(f"[kl  ] month {m}/{n_months} ({time.time()-t0:.0f}s)",
                  file=sys.stderr, flush=True)

    fy = tm["fy"].to_numpy()
    registered = tm["registered"].to_numpy()
    reg_year = tm["reg_year"].to_numpy()
    st_pass = tm["st_pass"].to_numpy()
    st_fail = tm["st_fail"].to_numpy()

    valid = np.isfinite(pros_near) & np.isfinite(pros_far) & (fy >= 1995)
    freshness = pros_far - pros_near

    out = {"n_scored": int(valid.sum()),
           "corr_near_far": float(np.corrcoef(pros_near[valid], pros_far[valid])[0, 1])}

    # registration panel
    reg_m = valid & (fy <= 2018)
    out["registration"] = {
        "n": int(reg_m.sum()),
        "pros_near": quintile_rates(pros_near[reg_m], registered[reg_m]),
        "pros_far": quintile_rates(pros_far[reg_m], registered[reg_m]),
        "freshness": quintile_rates(freshness[reg_m], registered[reg_m]),
    }
    # gate panel
    gm = valid & (reg_year >= 2016) & (reg_year <= 2018) & (st_pass | st_fail)
    out["gate"] = {
        "n": int(gm.sum()),
        "pros_near": quintile_rates(pros_near[gm], st_pass[gm]),
        "pros_far": quintile_rates(pros_far[gm], st_pass[gm]),
        "freshness": quintile_rates(freshness[gm], st_pass[gm]),
    }
    (RES / "subyear_window_009.json").write_text(json.dumps(out, indent=1, default=float))
    for panel in ("registration", "gate"):
        print(f"[{panel}] n={out[panel]['n']:,}", file=sys.stderr, flush=True)
        for v in ("pros_near", "pros_far", "freshness"):
            d = out[panel][v]
            print(f"   {v:>10}: lift {d['lift_q5_q1']:+.4f} invU {d['inverseU_depth']:+.4f}",
                  file=sys.stderr, flush=True)
    print(f"[done] corr(near,far)={out['corr_near_far']:.3f} ({time.time()-t0:.0f}s)",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
