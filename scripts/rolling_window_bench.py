"""Time the LDA transform on a sample of class 009 to cost the rolling rescore."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
N = 25_000

t0 = time.time()
df = pl.read_parquet(PROC / "tm_class009.parquet",
                     columns=["serial_number", "filing_date", "goods_services"])
df = df.filter(pl.col("goods_services").is_not_null()
               & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
n_total = df.height
print(f"[load] {n_total:,} usable class-009 filings in {time.time()-t0:.1f}s", file=sys.stderr)

m = joblib.load(PROC / "topic_model_T200.joblib")
vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                      token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b", ngram_range=(1, 2))
lda = m["lda"]
print(f"[model] T={lda.n_components}, vocab={len(m['vocabulary']):,}", file=sys.stderr)

samp = df.head(N)["goods_services"].to_list()
t0 = time.time(); X = vec.transform(samp); t_vec = time.time() - t0
t0 = time.time(); th = lda.transform(X); t_lda = time.time() - t0

per = (t_vec + t_lda) / N
print(f"[bench] vectorize {t_vec:.1f}s + transform {t_lda:.1f}s on {N:,} docs", file=sys.stderr)
print(f"[bench] {per*1e3:.3f} ms/doc", file=sys.stderr)
print(f"[estimate] class 009 ({n_total:,} docs): {per*n_total/60:.1f} min", file=sys.stderr)
print(f"[estimate] full corpus (~16.7M class-rows): {per*16_730_022/3600:.1f} h", file=sys.stderr)
print(f"[memory] theta float32 for class 009: {n_total*lda.n_components*4/1e9:.2f} GB", file=sys.stderr)
