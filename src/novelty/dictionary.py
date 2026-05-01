"""Build a vocabulary (the "dictionary") from a parquet of trademark records.

Treats each filing's `goods_services` text as one document. Lowercases, drops
English + USPTO stopwords, extracts unigrams and bigrams, and computes both
document-frequency (in how many filings the term appears) and term-frequency
(total occurrences). Filters by min/max document frequency.

Outputs a vocabulary parquet:
    token         str
    ngram_size    uint8     (1 or 2)
    doc_freq      uint32
    term_freq     uint32

Usage:
    python -m novelty.dictionary \\
        --input data/processed/tm_class032.parquet \\
        --out   data/processed/vocab_class032.parquet

The same module also exposes `build_vocabulary` for use as a library.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sk_text

# USPTO goods/services-description boilerplate. These tokens carry no
# information about novelty — they're filing conventions. Keep this list
# aggressive on truly generic words and let the doc-frequency cutoff handle
# the rest.
USPTO_STOPWORDS = {
    "namely",
    "viz",
    "ie",
    "eg",
    "et",
    "al",
    "consisting",
    "comprising",
    "comprised",
    "containing",
    "including",
    "includes",
    "included",
    "also",
    "use",
    "used",
    "uses",
    "using",
    "various",
    "general",
    "certain",
    "providing",
    "provided",
    "provides",
    "made",
    "make",
    "makes",
    "form",
    "forms",
    "kind",
    "kinds",
    "sort",
    "sorts",
    "type",
    "types",
    "class",
    "classes",
    "thereof",
    "therein",
    "therewith",
    "wherein",
    "whereby",
    "such",
    "may",
    "shall",
    "would",
    "could",
    "should",
    "etc",
}

STOPWORDS = sorted(set(sk_text.ENGLISH_STOP_WORDS) | USPTO_STOPWORDS)

# Lowercase alphabetic tokens of length >= 2; drop pure numbers and 1-char.
TOKEN_RE = r"(?u)\b[a-z][a-z'-]{1,}\b"


_SEP_RE = re.compile(r"[;,.()\[\]{}\"]+")
_TOKEN_RE = re.compile(TOKEN_RE)


def _texts(df: pl.DataFrame) -> list[str]:
    return [t.lower() if t else "" for t in df["goods_services"].to_list()]


def _make_analyzer(stop: frozenset[str], ngram_range: tuple[int, int]):
    """Custom analyzer: split each document at punctuation, tokenize each
    chunk independently, and emit n-grams only within a chunk so they don't
    cross semicolon-separated goods/services items."""
    lo, hi = ngram_range

    def analyze(doc: str) -> list[str]:
        out: list[str] = []
        for chunk in _SEP_RE.split(doc):
            toks = [t for t in _TOKEN_RE.findall(chunk) if t not in stop]
            if not toks:
                continue
            for n in range(lo, hi + 1):
                if n == 1:
                    out.extend(toks)
                else:
                    for i in range(len(toks) - n + 1):
                        out.append(" ".join(toks[i : i + n]))
        return out

    return analyze


def build_vocabulary(
    df: pl.DataFrame,
    *,
    min_df: int | float = 5,
    max_df: float = 0.5,
    ngram_range: tuple[int, int] = (1, 2),
) -> pl.DataFrame:
    docs = _texts(df)
    if not any(docs):
        raise RuntimeError("input parquet has no goods_services text to tokenize")

    vec = CountVectorizer(
        analyzer=_make_analyzer(frozenset(STOPWORDS), ngram_range),
        min_df=min_df,
        max_df=max_df,
    )
    dtm = vec.fit_transform(docs)

    vocab = vec.get_feature_names_out()
    df_counts = (dtm > 0).sum(axis=0).A1.astype("uint32")
    tf_counts = dtm.sum(axis=0).A1.astype("uint32")
    ngram_sz = [token.count(" ") + 1 for token in vocab]

    out = pl.DataFrame(
        {
            "token": vocab.tolist(),
            "ngram_size": pl.Series(ngram_sz, dtype=pl.UInt8),
            "doc_freq": pl.Series(df_counts, dtype=pl.UInt32),
            "term_freq": pl.Series(tf_counts, dtype=pl.UInt32),
        }
    )
    return out.sort(["ngram_size", "doc_freq"], descending=[False, True])


def _print_qc(vocab: pl.DataFrame, n_docs: int) -> None:
    n1 = vocab.filter(pl.col("ngram_size") == 1).height
    n2 = vocab.filter(pl.col("ngram_size") == 2).height
    print(f"docs:       {n_docs}", file=sys.stderr)
    print(f"vocab size: {vocab.height}  (unigrams={n1}  bigrams={n2})", file=sys.stderr)
    print("\n--- top 25 unigrams by doc_freq ---", file=sys.stderr)
    for r in vocab.filter(pl.col("ngram_size") == 1).head(25).iter_rows(named=True):
        print(f"  {r['token']:<30}  df={r['doc_freq']:>6}  tf={r['term_freq']:>6}", file=sys.stderr)
    print("\n--- top 25 bigrams by doc_freq ---", file=sys.stderr)
    for r in vocab.filter(pl.col("ngram_size") == 2).head(25).iter_rows(named=True):
        print(f"  {r['token']:<30}  df={r['doc_freq']:>6}  tf={r['term_freq']:>6}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="parquet of trademark records")
    ap.add_argument("--out", required=True, type=Path, help="output vocabulary parquet path")
    ap.add_argument("--min-df", type=int, default=5, help="minimum document frequency")
    ap.add_argument("--max-df", type=float, default=0.5, help="maximum document-frequency fraction")
    ap.add_argument("--max-ngram", type=int, default=2, help="ngram upper bound (1 or 2)")
    args = ap.parse_args()

    df = pl.read_parquet(args.input)
    vocab = build_vocabulary(
        df,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, args.max_ngram),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vocab.write_parquet(args.out)
    _print_qc(vocab, n_docs=df.height)
    print(f"\n[done] vocab -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
