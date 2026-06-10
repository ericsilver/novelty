"""Workstream D (within-class version) on Class 009.

Does the resonance signal come from genuinely NEW class vocabulary, or from
recombining ESTABLISHED class vocabulary? With one class we can only split
within-class vintage (new-to-class vs established); the corpus-new vs
class-new(recombination) split needs other classes.

For each filing we compute novel_share = fraction of its tokens whose FIRST
appearance in Class 009 was within K years before the filing. Then we relate
novel_share to signed dKL. If high-resonance filings load on recently-introduced
tokens, the signal is new-vocabulary-driven (the diffusion/innovation reading);
if resonance is flat in novel_share, it is recombination of old terms.

Output: paper/results/wsH_provenance009.json (+ stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT = REPO / "paper" / "results" / "wsH_provenance009.json"
K = 3            # token counts as "new to class" if first seen <= K yrs before filing
BURN_IN, EDGE = 1996, 2020


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer

    rec = pl.read_parquet(DATA / "tm_class009.parquet").with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
    ).filter((pl.col("goods_services").str.len_chars() > 0) & pl.col("year").is_not_null())
    vocab = pl.read_parquet(DATA / "vocab_class009.parquet").filter(
        pl.col("doc_freq") >= 50)["token"].to_list()
    print(f"[load] {rec.height:,} filings, vocab {len(vocab):,}")

    vec = CountVectorizer(analyzer=_make_analyzer(frozenset(STOPWORDS), (1, 2)),
                          vocabulary=vocab, binary=True)
    X = vec.fit_transform(rec["goods_services"].to_list()).tocsr()
    years = rec["year"].to_numpy()
    V = len(vocab)

    # token first-appearance year in class
    first_year = np.full(V, 1 << 30, dtype=np.int64)
    for y in sorted(set(years.tolist())):
        cols = X[np.where(years == y)[0]].sum(axis=0).A1.nonzero()[0]
        first_year[cols] = np.minimum(first_year[cols], y)

    # per-filing novel_share, bucketed by year for a fast sparse mat-vec
    n_tok = np.asarray(X.sum(axis=1)).ravel()
    novel_cnt = np.zeros(X.shape[0], dtype=np.float64)
    for y in sorted(set(years.tolist())):
        idx = np.where(years == y)[0]
        novel_cols = (first_year >= (y - K)).astype(np.float64)   # token new as of year y
        novel_cnt[idx] = np.asarray(X[idx].dot(novel_cols)).ravel()
    novel_share = np.divide(novel_cnt, n_tok, out=np.zeros_like(novel_cnt), where=n_tok > 0)

    prov = pl.DataFrame({"serial_number": rec["serial_number"], "year": years,
                         "n_tok": n_tok, "novel_share": novel_share})
    sur = pl.read_parquet(DATA / "surprise_class009.parquet").with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl")
    ).filter(pl.col("prospective_kl").is_finite() & pl.col("retrospective_kl").is_finite())
    df = prov.join(sur.select("serial_number", "dkl"), on="serial_number", how="inner").filter(
        (pl.col("year") >= BURN_IN) & (pl.col("year") <= EDGE) & (pl.col("n_tok") >= 10))
    print(f"[join] {df.height:,} filings in {BURN_IN}-{EDGE} with >=10 tokens")

    ns, dk = df["novel_share"].to_numpy(), df["dkl"].to_numpy()
    r = float(np.corrcoef(ns, dk)[0, 1])
    sp = stats.spearmanr(ns, dk)
    res = {"K": K, "window": [BURN_IN, EDGE], "n": df.height,
           "mean_novel_share": float(ns.mean()),
           "pearson_novelshare_dkl": r, "spearman": float(sp.statistic),
           "spearman_p": float(sp.pvalue)}
    print(f"novel_share mean={ns.mean():.3f}  corr(novel_share, dKL)  "
          f"Pearson={r:+.3f}  Spearman={sp.statistic:+.3f} (p={sp.pvalue:.2g})")

    # mean signed dKL by novel_share quintile
    q = np.quantile(ns, np.linspace(0, 1, 6))
    binq = np.clip(np.digitize(ns, q[1:-1]), 0, 4)
    rows = []
    print("mean signed dKL by novel-token-share quintile:")
    for b in range(5):
        m = binq == b
        rows.append({"quintile": b + 1, "novel_share_mid": float(ns[m].mean()),
                     "mean_dkl": float(dk[m].mean()), "n": int(m.sum())})
        print(f"  Q{b+1}: novel_share~{ns[m].mean():.2f}  mean dKL={dk[m].mean():+.4f}  (n={m.sum():,})")
    res["by_quintile"] = rows
    OUT.write_text(json.dumps(res, indent=2))
    print(f"[done] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
