"""Diffusion Phase 0 (#8): borrowed novelty = within-class vs cross-corpus gap.

PROPOSAL_diffusion §2.3. For each GRANTED filing in classes {009,042,039,035},
compute prospective novelty (KL of the filing's token distribution against the
PAST, W=5y) twice, in a shared universal vocabulary:

    within_nov = KL(P || class-c past)      -- novel against its own industry
    cross_nov  = KL(P || all-classes past)  -- novel against the whole corpus

  high within, low cross  -> BORROWED novelty (locally new, globally old): a
                             theme arriving in this class from elsewhere.
  high within, high cross -> genuinely new (born here).
  low  within             -> derivative.

KL is vectorized: KL_i = selfent_i - (X[i] . logQ)/n_i, with
selfent_i = (sum_j c_ij log c_ij)/n_i - log n_i.

Input : data/processed/tm_class{009,042,039,035}.parquet
Output: data/processed/borrowed_novelty.parquet + paper/results/diff1_borrowed.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT_PARQUET = DATA / "borrowed_novelty.parquet"
OUT_JSON = REPO / "paper" / "results" / "diff1_borrowed.json"

CLASSES = ["009", "042", "039", "035"]
W = 5            # past-window years
ALPHA = 1.0
MIN_DF = 100     # universal vocab cutoff
YEAR_MIN = 1990  # drop thin early/burn-in years


def build_logref(tf_by_key, keys_for, V):
    """tf_by_key: dict key->(V,) int TF for that key's single year.
    keys_for(y): list of source keys whose TF to pool for the past window of y.
    Returns dict y->log Q array (V,)."""
    out = {}
    for y, srckeys in keys_for.items():
        tf = np.zeros(V, dtype=np.float64)
        for k in srckeys:
            arr = tf_by_key.get(k)
            if arr is not None:
                tf += arr
        if tf.sum() == 0:
            continue
        q = (tf + ALPHA) / (tf.sum() + ALPHA * V)
        out[y] = np.log(q)
    return out


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer

    t0 = time.time()
    frames = []
    for c in CLASSES:
        p = DATA / f"tm_class{c}.parquet"
        if not p.exists():
            print(f"MISSING {p} -- run download_all_classes.py first", file=sys.stderr)
            raise SystemExit(2)
        df = pl.read_parquet(p, columns=["serial_number", "filing_date", "registration_date",
                                         "mark_identification", "goods_services"])
        df = df.with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
            pl.lit(c).alias("nice"),
        ).filter(
            (pl.col("goods_services").str.len_chars() > 0)
            & pl.col("year").is_not_null() & (pl.col("year") >= YEAR_MIN)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 4)  # GRANTED
        )
        frames.append(df)
        print(f"  class {c}: {df.height:,} granted filings", file=sys.stderr)
    rec = pl.concat(frames)
    print(f"[pooled] {rec.height:,} granted filings  ({time.time()-t0:.1f}s)", file=sys.stderr)

    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    vec = CountVectorizer(analyzer=analyzer, min_df=MIN_DF)
    X = vec.fit_transform(rec["goods_services"].to_list()).tocsr()
    V = X.shape[1]
    print(f"[vec] X={X.shape} nnz={X.nnz:,} universal vocab V={V:,}  ({time.time()-t0:.1f}s)",
          file=sys.stderr)

    years = rec["year"].to_numpy()
    nice = rec["nice"].to_numpy()
    n = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    # self-entropy
    Xlog = X.copy().astype(np.float64)
    Xlog.data = X.data * np.log(X.data)
    S = np.asarray(Xlog.sum(axis=1)).ravel()
    selfent = np.divide(S, n, out=np.zeros_like(S), where=n > 0) - np.log(np.where(n > 0, n, 1))

    uyears = sorted(set(int(y) for y in years.tolist()))
    # per-year TF (universal) and per (class,year) TF
    tf_year = {}
    tf_cy = {}
    for y in uyears:
        ymask = years == y
        tf_year[y] = np.asarray(X[np.where(ymask)[0]].sum(axis=0)).ravel().astype(np.float64)
        for c in CLASSES:
            m = ymask & (nice == c)
            if m.any():
                tf_cy[(c, y)] = np.asarray(X[np.where(m)[0]].sum(axis=0)).ravel().astype(np.float64)

    # past-window reference keys
    univ_keys = {y: [yy for yy in range(y - W, y) if yy in tf_year] for y in uyears}
    logQ_univ = build_logref(tf_year, univ_keys, V)
    logQ_cls = {}
    for c in CLASSES:
        cy_keys = {y: [(c, yy) for yy in range(y - W, y) if (c, yy) in tf_cy] for y in uyears}
        logQ_cls[c] = build_logref(tf_cy, cy_keys, V)
    print(f"[refs] built  ({time.time()-t0:.1f}s)", file=sys.stderr)

    within = np.full(X.shape[0], np.nan)
    cross = np.full(X.shape[0], np.nan)
    for y in uyears:
        if y not in logQ_univ:
            continue
        idx = np.where(years == y)[0]
        cross[idx] = selfent[idx] - np.asarray(X[idx].dot(logQ_univ[y])).ravel() / n[idx]
        for c in CLASSES:
            if y not in logQ_cls[c]:
                continue
            cm = idx[nice[idx] == c]
            if len(cm):
                within[cm] = selfent[cm] - np.asarray(X[cm].dot(logQ_cls[c][y])).ravel() / n[cm]

    out = rec.select("serial_number", "nice", "year", "mark_identification").with_columns(
        pl.Series("within_nov", within), pl.Series("cross_nov", cross),
        pl.Series("n_terms", n.astype(np.int32)),
    ).filter(pl.col("within_nov").is_finite() & pl.col("cross_nov").is_finite()
             & (pl.col("n_terms") >= 5))
    out = out.with_columns((pl.col("within_nov") - pl.col("cross_nov")).alias("borrowed_gap"))
    out.write_parquet(OUT_PARQUET)
    print(f"[scored] {out.height:,} filings  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # 2x2 classification by within/cross medians
    wmed = out["within_nov"].median()
    cmed = out["cross_nov"].median()
    res = {"classes": CLASSES, "W": W, "min_df": MIN_DF, "n": out.height,
           "within_median": float(wmed), "cross_median": float(cmed), "by_class": {}}
    print(f"\nwithin median={wmed:.3f}  cross median={cmed:.3f}")
    print(f"{'class':>6} {'n':>9} {'borrowed%':>10} {'born-new%':>10} {'mean gap':>9}")
    for c in CLASSES:
        s = out.filter(pl.col("nice") == c)
        hi_w = s["within_nov"] > wmed
        lo_c = s["cross_nov"] <= cmed
        borrowed = float((hi_w & lo_c).mean())
        born = float((hi_w & ~lo_c).mean())
        res["by_class"][c] = {"n": s.height, "borrowed_share": borrowed,
                              "born_new_share": born, "mean_gap": float(s["borrowed_gap"].mean())}
        print(f"{c:>6} {s.height:>9,} {borrowed:>9.1%} {born:>9.1%} {s['borrowed_gap'].mean():>+9.3f}")

    # top borrowed-novelty marks per class (high within, low cross)
    res["examples"] = {}
    print("\nTop BORROWED-novelty marks (locally new, globally old) per class:")
    for c in CLASSES:
        s = out.filter((pl.col("nice") == c) & (pl.col("cross_nov") <= cmed)
                       & (pl.col("n_terms") >= 10)).sort("borrowed_gap", descending=True).head(6)
        res["examples"][c] = s.select("year", "mark_identification", "borrowed_gap").to_dicts()
        print(f"  [{c}]")
        for r in s.iter_rows(named=True):
            print(f"     {r['year']} gap={r['borrowed_gap']:+.2f}  {str(r['mark_identification'])[:42]}")

    OUT_JSON.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[done] -> {OUT_PARQUET} ; {OUT_JSON}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
