"""Compute DeDeo prospective + retrospective KL surprise per filing.

For each filing at filing-year t with term-frequency distribution P over the
(filtered) dictionary:

    Q_pros(y)  = aggregate TF of all filings in years [t - W, t)
    Q_retr(y)  = aggregate TF of all filings in years (t, t + W]
    KL(P || Q) = sum_{i: P_i > 0} P_i * (log P_i - log Q_i)

Q is Laplace-smoothed (+alpha per cell) so unseen terms contribute finite mass.

Usage:
    python -m novelty.surprise \\
        --records data/processed/tm_class032.parquet \\
        --vocab   data/processed/vocab_class032.parquet \\
        --out     data/processed/surprise_class032.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

from .dictionary import STOPWORDS, _make_analyzer

REPO_ROOT = Path(__file__).resolve().parents[2]


def _vectorize(docs: list[str], vocabulary: list[str]):
    vec = CountVectorizer(
        analyzer=_make_analyzer(frozenset(STOPWORDS), (1, 2)),
        vocabulary=vocabulary,
    )
    return vec.fit_transform(docs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=REPO_ROOT / "data/processed/tm_class032.parquet")
    ap.add_argument("--vocab", type=Path, default=REPO_ROOT / "data/processed/vocab_class032.parquet")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/surprise_class032.parquet")
    ap.add_argument("--min-df", type=int, default=50, help="restrict dictionary to terms with doc_freq >= min-df")
    ap.add_argument("--window", type=int, default=5, help="years of look-behind/look-ahead")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing on reference Q")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] {args.records}", file=sys.stderr)
    df = pl.read_parquet(args.records)
    df = df.filter(
        pl.col("filing_date").is_not_null()
        & (pl.col("filing_date").str.len_chars() >= 4)
        & (pl.col("goods_services").str.len_chars() > 0)
    ).with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"))
    print(f"       rows with year+text: {df.height:,}  years {df['year'].min()}..{df['year'].max()}", file=sys.stderr)

    print(f"[load] {args.vocab}  (min_df>={args.min_df})", file=sys.stderr)
    vocab = (
        pl.read_parquet(args.vocab).filter(pl.col("doc_freq") >= args.min_df)["token"].to_list()
    )
    V = len(vocab)
    print(f"       vocab: {V:,} terms", file=sys.stderr)

    print(f"[vec ] vectorizing {df.height:,} docs…", file=sys.stderr)
    X = _vectorize(df["goods_services"].to_list(), vocab).tocsr()
    print(f"       X shape={X.shape} nnz={X.nnz:,}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    years = df["year"].to_numpy()
    unique_years = sorted(set(years.tolist()))

    print(f"[agg ] per-year totals over {len(unique_years)} years…", file=sys.stderr)
    year_tf: dict[int, np.ndarray] = {}
    year_n: dict[int, int] = {}
    for y in unique_years:
        idx = np.where(years == y)[0]
        year_tf[y] = np.asarray(X[idx].sum(axis=0), dtype=np.int64).ravel()
        year_n[y] = int(len(idx))

    W = args.window
    alpha = float(args.alpha)
    log_pros: dict[int, np.ndarray] = {}
    log_retr: dict[int, np.ndarray] = {}
    n_pros_y: dict[int, int] = {}
    n_retr_y: dict[int, int] = {}
    for y in unique_years:
        p_tf = np.zeros(V, dtype=np.int64); p_n = 0
        for yy in range(y - W, y):
            if yy in year_tf:
                p_tf += year_tf[yy]; p_n += year_n[yy]
        if p_n > 0:
            p_q = (p_tf.astype(np.float64) + alpha) / (p_tf.sum() + alpha * V)
            log_pros[y] = np.log(p_q)
            n_pros_y[y] = p_n

        r_tf = np.zeros(V, dtype=np.int64); r_n = 0
        for yy in range(y + 1, y + W + 1):
            if yy in year_tf:
                r_tf += year_tf[yy]; r_n += year_n[yy]
        if r_n > 0:
            r_q = (r_tf.astype(np.float64) + alpha) / (r_tf.sum() + alpha * V)
            log_retr[y] = np.log(r_q)
            n_retr_y[y] = r_n

    print(f"[kl  ] computing per-filing KL…", file=sys.stderr)
    n_docs = X.shape[0]
    indptr = X.indptr
    indices = X.indices
    data = X.data
    pros_kl = np.full(n_docs, np.nan, dtype=np.float64)
    retr_kl = np.full(n_docs, np.nan, dtype=np.float64)
    n_pros = np.zeros(n_docs, dtype=np.int32)
    n_retr = np.zeros(n_docs, dtype=np.int32)
    n_terms = np.zeros(n_docs, dtype=np.int32)

    for i in range(n_docs):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            continue
        cols = indices[s:e]
        cnts = data[s:e].astype(np.float64)
        total = cnts.sum()
        if total <= 0:
            continue
        p = cnts / total
        log_p = np.log(p)
        n_terms[i] = e - s
        y = int(years[i])
        if y in log_pros:
            pros_kl[i] = float(np.sum(p * (log_p - log_pros[y][cols])))
            n_pros[i] = n_pros_y[y]
        if y in log_retr:
            retr_kl[i] = float(np.sum(p * (log_p - log_retr[y][cols])))
            n_retr[i] = n_retr_y[y]

    out = df.with_columns(
        pl.Series("prospective_kl", pros_kl),
        pl.Series("retrospective_kl", retr_kl),
        pl.Series("n_ref_prospective", n_pros),
        pl.Series("n_ref_retrospective", n_retr),
        pl.Series("n_terms", n_terms),
    ).select(
        "serial_number",
        "filing_date",
        "year",
        "mark_identification",
        "owner_name",
        "n_terms",
        "prospective_kl",
        "n_ref_prospective",
        "retrospective_kl",
        "n_ref_retrospective",
    )
    out.write_parquet(args.out)
    print(
        f"[done] {out.height:,} rows -> {args.out}  "
        f"(prospective non-null: {out['prospective_kl'].is_not_null().sum():,}, "
        f"retrospective: {out['retrospective_kl'].is_not_null().sum():,})  "
        f"({time.time()-t0:.1f}s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
