"""For a small set of filings, recompute per-token contribution to
prospective and retrospective KL, so that the paper's discussion table can
explain what tokens are driving each filing's score.

KL(P||Q) = sum_v P(v) * (log P(v) - log Q(v)); we expose the per-token
summand so the table can show which dictionary entries pull the filing's
score up or down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

from .dictionary import STOPWORDS, _make_analyzer


@dataclass
class Attribution:
    serial_number: str
    mark: str
    owner: str
    year: int
    goods_services: str
    n_terms: int
    pros_kl: float
    retr_kl: float
    dkl: float
    top_pros_terms: list[tuple[str, float]]
    top_retr_terms: list[tuple[str, float]]


def attribute(
    records_path: Path,
    surprise_path: Path,
    vocab_path: Path,
    serial_numbers: list[str],
    *,
    min_df: int = 50,
    window: int = 5,
    alpha: float = 1.0,
    top_k: int = 5,
) -> list[Attribution]:
    rec = pl.read_parquet(records_path).filter(pl.col("serial_number").is_in(serial_numbers))
    sup = pl.read_parquet(surprise_path).filter(pl.col("serial_number").is_in(serial_numbers))
    if rec.is_empty():
        return []

    full = pl.read_parquet(records_path).filter(
        pl.col("filing_date").is_not_null()
        & (pl.col("filing_date").str.len_chars() >= 4)
        & (pl.col("goods_services").str.len_chars() > 0)
    ).with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"))

    vocab = (
        pl.read_parquet(vocab_path).filter(pl.col("doc_freq") >= min_df)["token"].to_list()
    )
    V = len(vocab)
    vec = CountVectorizer(
        analyzer=_make_analyzer(frozenset(STOPWORDS), (1, 2)),
        vocabulary=vocab,
    )
    X = vec.fit_transform(full["goods_services"].to_list()).tocsr()
    years = full["year"].to_numpy()

    target_serials = full["serial_number"].to_numpy()
    serial_to_row = {s: i for i, s in enumerate(target_serials)}

    unique_years = sorted(set(int(y) for y in years.tolist()))
    year_tf: dict[int, np.ndarray] = {}
    year_n: dict[int, int] = {}
    for y in unique_years:
        idx = np.where(years == y)[0]
        year_tf[y] = np.asarray(X[idx].sum(axis=0), dtype=np.int64).ravel()
        year_n[y] = int(len(idx))

    def ref_log_q(year: int, lo: int, hi: int) -> tuple[np.ndarray | None, int]:
        tf = np.zeros(V, dtype=np.int64)
        n = 0
        for yy in range(year + lo, year + hi + 1):
            if yy in year_tf:
                tf += year_tf[yy]
                n += year_n[yy]
        if n == 0:
            return None, 0
        q = (tf.astype(np.float64) + alpha) / (tf.sum() + alpha * V)
        return np.log(q), n

    out: list[Attribution] = []
    rec_ix = {r["serial_number"]: r for r in rec.iter_rows(named=True)}
    sup_ix = {r["serial_number"]: r for r in sup.iter_rows(named=True)}
    for sn in serial_numbers:
        if sn not in rec_ix or sn not in serial_to_row:
            continue
        rec_row = rec_ix[sn]
        sup_row = sup_ix.get(sn, {})
        i = serial_to_row[sn]
        s, e = X.indptr[i], X.indptr[i + 1]
        if s == e:
            continue
        cols = X.indices[s:e]
        cnts = X.data[s:e].astype(np.float64)
        total = cnts.sum()
        if total <= 0:
            continue
        p = cnts / total
        log_p = np.log(p)
        year = int(years[i])

        log_q_pros, _ = ref_log_q(year, -window, -1)
        log_q_retr, _ = ref_log_q(year, 1, window)

        def topk(log_q):
            if log_q is None:
                return []
            contrib = p * (log_p - log_q[cols])
            order = np.argsort(-contrib)[:top_k]
            return [(vocab[int(cols[j])], float(contrib[j])) for j in order]

        out.append(
            Attribution(
                serial_number=sn,
                mark=rec_row.get("mark_identification") or "",
                owner=rec_row.get("owner_name") or "",
                year=year,
                goods_services=rec_row.get("goods_services") or "",
                n_terms=int(e - s),
                pros_kl=float(sup_row.get("prospective_kl", float("nan"))),
                retr_kl=float(sup_row.get("retrospective_kl", float("nan"))),
                dkl=float(sup_row.get("prospective_kl", 0)) - float(sup_row.get("retrospective_kl", 0)),
                top_pros_terms=topk(log_q_pros),
                top_retr_terms=topk(log_q_retr),
            )
        )
    return out
