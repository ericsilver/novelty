"""Decay-weighted DeDeo prospective + retrospective KL surprise per filing.

Same shape as ``surprise.py`` but the reference distribution is built by
per-day exponential weighting of all prior (or future) filings — not by
flat-bloc summation over a 5-year window.

For filing $i$ at filing-date $t_i$, the prospective reference is

  Q_i^{pros}(v)  ∝  α + Σ_j  exp(-λ_pros · Δt_j)  · count(v, j)

summed over all filings $j$ with $t_j < t_i$, where Δt_j is the time
between $t_j$ and $t_i$ in years and λ_pros = ln(2) / halflife_pros.

The retrospective reference is symmetric over $t_j > t_i$ with
λ_retr = ln(2) / halflife_retr.

We aggregate per filing-day rather than per filing-record, since within
a day there is no time ordering signal and per-day batching cuts the
inner-loop cost from O(N · V) to O(D · V) with D = unique days
(typically 7–8k for a class spanning 1985–2025).

Density measure: instead of n_ref (count of in-window filings) we
report Kish effective sample size

  N_eff = (Σ w_j)^2 / Σ w_j^2,

which collapses to the raw filing count when all weights are 1 and
shrinks as the weighted mass concentrates on a few records.

Columns written: kl_vs_past (= the prospective/backward-looking reference,
t_j < t_i) and kl_vs_future (= the retrospective/forward-looking reference,
t_j > t_i), with the Kish sizes kept under their original names
n_eff_prospective / n_eff_retrospective (n_eff_prospective belongs to
kl_vs_past, n_eff_retrospective to kl_vs_future).  scripts/recompute_h2.py
maps those to n_ref_past / n_ref_future when converting to surprise schema.

Usage:

    python -m novelty.surprise_decay \\
        --records  data/processed/tm_class009.parquet \\
        --vocab    data/processed/vocab_class009.parquet \\
        --out      data/processed/surprise_decay_class009.parquet \\
        --halflife-pros 2.0 \\
        --halflife-retr 2.0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .dictionary import STOPWORDS, _make_analyzer

REPO_ROOT = Path(__file__).resolve().parents[2]


def _vectorize(docs: list[str], vocabulary: list[str]):
    vec = CountVectorizer(
        analyzer=_make_analyzer(frozenset(STOPWORDS), (1, 2)),
        vocabulary=vocabulary,
    )
    return vec.fit_transform(docs)


def _per_day_aggregates(X: sparse.csr_matrix, day_idx: np.ndarray, n_days: int):
    """Return:
        day_tf: dense (D, V) per-day term counts (float64)
        day_n:  per-day filing counts (int64)
    Memory: D * V * 8 bytes; for D~10k, V~10k → 800MB.  We instead keep
    a sparse CSR matrix indexed by day, row d = sum of all filings on
    that day.
    """
    V = X.shape[1]
    # Build a (D x N_filings) selector matrix; row d has 1.0 in cols of
    # filings filed on day d.  S @ X gives the per-day aggregate.
    rows = day_idx
    cols = np.arange(X.shape[0])
    data = np.ones_like(rows, dtype=np.int8)
    S = sparse.csr_matrix((data, (rows, cols)), shape=(n_days, X.shape[0]))
    day_tf = (S @ X).astype(np.float64)  # sparse, (D, V)
    day_n  = np.asarray(S.sum(axis=1)).ravel().astype(np.int64)
    return day_tf, day_n


def _decay_pass_score(day_tf: sparse.csr_matrix, day_n: np.ndarray,
                      day_gap_years: np.ndarray, day_doc_ids: list[np.ndarray],
                      X: sparse.csr_matrix, halflife: float, alpha: float,
                      reverse: bool = False):
    """Streaming forward (or reverse) pass: walk days in order, maintain
    a running decay-weighted term-count vector R, score each filing on
    each day inline, return per-filing KL and effective sample size.

    Memory: O(V) for R, plus per-day csr row materialisation.  No giant
    (D × V) log-reference matrix.
    """
    D, V = day_tf.shape
    lam = math.log(2.0) / halflife
    R     = np.zeros(V, dtype=np.float64)
    sum_w = 0.0
    sum_w2 = 0.0

    n_docs = X.shape[0]
    kl  = np.full(n_docs, np.nan, dtype=np.float64)
    n_eff_per_doc = np.full(n_docs, np.nan, dtype=np.float64)

    # Pre-extract X.indptr/indices/data once
    indptr = X.indptr
    indices = X.indices
    data = X.data

    log_alpha = math.log(alpha)
    direction = "reverse" if reverse else "forward"
    last_log = time.time()

    for step in range(D):
        d = D - 1 - step if reverse else step
        gap = day_gap_years[step]
        if step > 0 and gap > 0:
            decay = math.exp(-lam * gap)
            R *= decay
            sum_w *= decay
            sum_w2 *= decay * decay

        if sum_w > 0:
            R_sum = R.sum()
            denom = R_sum + alpha * V
            log_denom = math.log(denom)
            n_eff_d = (sum_w * sum_w) / sum_w2 if sum_w2 > 0 else 0.0
            for i in day_doc_ids[d]:
                s, e = indptr[i], indptr[i + 1]
                if s == e:
                    continue
                cols = indices[s:e]
                cnts = data[s:e].astype(np.float64)
                total = cnts.sum()
                if total <= 0:
                    continue
                p = cnts / total
                # log q for the cols this doc touches: log((R[cols]+α)/denom)
                log_q_cols = np.log(R[cols] + alpha) - log_denom
                kl[i] = float(np.sum(p * (np.log(p) - log_q_cols)))
                n_eff_per_doc[i] = n_eff_d

        # add day d's contribution
        n_d = float(day_n[d])
        if n_d > 0:
            row = day_tf.getrow(d).toarray().ravel()
            R += row
            sum_w += n_d
            sum_w2 += n_d

        if time.time() - last_log > 30:
            print(f"       [{direction}] day {step+1}/{D}  ({step+1}/{D*1.0:.0f}={100*(step+1)/D:.1f}%)  "
                  f"R_sum={R.sum():.2g}  sum_w={sum_w:.2g}", file=sys.stderr)
            last_log = time.time()

    return kl, n_eff_per_doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True,
                    help="per-class tm parquet (e.g. tm_class009.parquet)")
    ap.add_argument("--vocab", type=Path, required=True,
                    help="per-class vocab parquet")
    ap.add_argument("--out", type=Path, required=True,
                    help="output surprise-decay parquet")
    ap.add_argument("--min-df", type=int, default=50,
                    help="restrict dictionary to terms with doc_freq >= min-df")
    ap.add_argument("--halflife-pros", type=float, default=2.0,
                    help="prospective half-life in years; weight at age T is 2^(-T/H)")
    ap.add_argument("--halflife-retr", type=float, default=2.0,
                    help="retrospective half-life in years")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Laplace smoothing on reference Q")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] {args.records}", file=sys.stderr)
    df = pl.read_parquet(args.records)
    df = df.filter(
        pl.col("filing_date").is_not_null()
        & (pl.col("filing_date").str.len_chars() == 8)
        & (pl.col("goods_services").str.len_chars() > 0)
    ).with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("date"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"),
    ).filter(pl.col("date").is_not_null())
    print(f"       rows: {df.height:,}  span {df['date'].min()}..{df['date'].max()}", file=sys.stderr)

    print(f"[load] {args.vocab}  (min_df>={args.min_df})", file=sys.stderr)
    vocab = (pl.read_parquet(args.vocab)
             .filter(pl.col("doc_freq") >= args.min_df)["token"].to_list())
    V = len(vocab)
    print(f"       vocab: {V:,} terms", file=sys.stderr)

    print(f"[vec ] vectorizing {df.height:,} docs…", file=sys.stderr)
    X = _vectorize(df["goods_services"].to_list(), vocab).tocsr()
    print(f"       X shape={X.shape} nnz={X.nnz:,}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # Map filing dates to dense day indices
    dates = df["date"].to_numpy()  # numpy datetime64[ns] or [day]
    dates = dates.astype("datetime64[D]")
    unique_days, day_idx = np.unique(dates, return_inverse=True)
    D = len(unique_days)
    day_gap_days = np.diff(unique_days).astype("int64")
    day_gap_days = np.concatenate([[0], day_gap_days]).astype(np.float64)
    day_gap_years = day_gap_days / 365.25
    print(f"[days] D={D:,}  span {unique_days[0]}..{unique_days[-1]}", file=sys.stderr)

    print(f"[agg ] per-day aggregates ({D} rows × {V} cols)…", file=sys.stderr)
    day_tf, day_n = _per_day_aggregates(X, day_idx, D)
    print(f"       day_tf nnz={day_tf.nnz:,}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # Pre-build per-day doc-id lists so the streaming pass can score
    # filings inline without a (D × V) reference matrix in memory.
    day_doc_ids: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(D)]
    order = np.argsort(day_idx, kind="stable")
    sorted_days = day_idx[order]
    boundaries = np.searchsorted(sorted_days, np.arange(D + 1))
    for d in range(D):
        day_doc_ids[d] = order[boundaries[d]:boundaries[d + 1]]

    rev_gap_days = np.diff(unique_days[::-1].astype("int64") * -1)
    rev_gap_years = np.concatenate([[0], rev_gap_days]).astype(np.float64) / 365.25

    n_terms = np.diff(X.indptr).astype(np.int32)

    print(f"[pros] forward pass, H_pros={args.halflife_pros}y…", file=sys.stderr)
    pros_kl, n_pros = _decay_pass_score(
        day_tf, day_n, day_gap_years, day_doc_ids, X,
        args.halflife_pros, args.alpha, reverse=False,
    )
    print(f"       ({time.time()-t0:.1f}s)", file=sys.stderr)

    print(f"[retr] reverse pass, H_retr={args.halflife_retr}y…", file=sys.stderr)
    retr_kl, n_retr = _decay_pass_score(
        day_tf, day_n, rev_gap_years, day_doc_ids, X,
        args.halflife_retr, args.alpha, reverse=True,
    )
    print(f"       ({time.time()-t0:.1f}s)", file=sys.stderr)

    out = df.with_columns(
        pl.Series("kl_vs_past", pros_kl),
        pl.Series("kl_vs_future", retr_kl),
        pl.Series("n_eff_prospective", n_pros),
        pl.Series("n_eff_retrospective", n_retr),
        pl.Series("n_terms", n_terms),
    ).with_columns(
        pl.lit(args.halflife_pros).alias("halflife_pros"),
        pl.lit(args.halflife_retr).alias("halflife_retr"),
    ).select(
        "serial_number",
        "filing_date",
        "year",
        "mark_identification",
        "owner_name",
        "n_terms",
        "kl_vs_past",
        "n_eff_prospective",
        "kl_vs_future",
        "n_eff_retrospective",
        "halflife_pros",
        "halflife_retr",
    )
    out.write_parquet(args.out)
    print(
        f"[done] {out.height:,} rows -> {args.out}  "
        f"(pros non-null: {out['kl_vs_past'].is_not_null().sum():,}, "
        f"retr: {out['kl_vs_future'].is_not_null().sum():,})  "
        f"halflife_pros={args.halflife_pros} halflife_retr={args.halflife_retr}  "
        f"({time.time()-t0:.1f}s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
