"""Pull the most popular tokens (across the whole corpus) that did NOT
appear in a given early reference window.

Why two windows: literal-first-five-years (1884-1888) is too sparse to
ground "did not exist" in. The defaults below use 1985-1989 (modern
USPTO density; pre-Internet) and 1995-1999 (pre-social-media), which
make "absent then, popular now" tokens interpretable.

Owner-level filters are intentionally NOT applied — this is about token
emergence, not firm dynamics.

For each window we report:
    - top-30 unigrams absent from the window with the largest total
      doc-frequency (i.e. "popular today, didn't exist then")
    - top-20 bigrams under the same rule
    - earliest-appearance year of each surfaced token (across all classes)

Outputs:
    paper/results/whats_new_tokens_w{start}_{end}.csv
    paper/results/whats_new_tokens.txt
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

from novelty.dictionary import STOPWORDS, _make_analyzer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def pooled_vocab(min_doc_freq: int) -> pl.DataFrame:
    """Sum doc_freq across all per-class vocab files."""
    parts = []
    for cls in class_list():
        vp = PROC / f"vocab_class{cls}.parquet"
        if not vp.exists(): continue
        parts.append(pl.read_parquet(vp).select("token", "doc_freq"))
    df = (pl.concat(parts)
            .group_by("token")
            .agg(pl.col("doc_freq").sum())
            .filter(pl.col("doc_freq") >= min_doc_freq)
            .sort("doc_freq", descending=True))
    print(f"[vocab] {df.height:,} tokens with total doc_freq >= {min_doc_freq:,}", flush=True)
    return df


def early_doc_counts(vocab: list[str], y_start: int, y_end: int) -> dict[str, int]:
    """For each token in vocab, count distinct filings in [y_start, y_end]
    (across all classes) that contain that token."""
    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    cv = CountVectorizer(analyzer=analyzer, vocabulary=vocab, binary=True)
    counts = {t: 0 for t in vocab}

    for cls in class_list():
        tm_p = PROC / f"tm_class{cls}.parquet"
        docs = (pl.read_parquet(tm_p, columns=["filing_date", "goods_services"])
                  .filter(
                      pl.col("filing_date").is_not_null()
                      & (pl.col("filing_date").str.len_chars() == 8)
                      & pl.col("goods_services").is_not_null()
                  ).with_columns(
                      pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("y")
                  ).filter(pl.col("y").is_between(y_start, y_end))
                  ["goods_services"].to_list())
        if not docs:
            continue
        X = cv.transform(docs)  # binary (docs x V) sparse
        col_sums = X.sum(axis=0).A1  # (V,)
        for j, t in enumerate(vocab):
            counts[t] += int(col_sums[j])
        print(f"  class {cls}: {len(docs):>7,} docs in {y_start}-{y_end}", flush=True)
        del docs, X
        gc.collect()
    return counts


def earliest_year(tokens: list[str]) -> dict[str, int]:
    """For each token, earliest filing-year in which it appears."""
    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    cv = CountVectorizer(analyzer=analyzer, vocabulary=tokens, binary=True)
    first_year = {t: 9999 for t in tokens}

    for cls in class_list():
        tm_p = PROC / f"tm_class{cls}.parquet"
        df = pl.read_parquet(tm_p, columns=["filing_date", "goods_services"]).filter(
            pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
            & pl.col("goods_services").is_not_null()
        ).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("y")
        )
        # Walk year by year ascending; stop tracking a token once we have its earliest year.
        for y in sorted(df["y"].unique().to_list()):
            still_unknown = [t for t in tokens if first_year[t] == 9999]
            if not still_unknown:
                break
            sub = df.filter(pl.col("y") == y)["goods_services"].to_list()
            if not sub:
                continue
            cv2 = CountVectorizer(analyzer=analyzer, vocabulary=still_unknown, binary=True)
            X = cv2.transform(sub)
            col_sums = X.sum(axis=0).A1
            for j, t in enumerate(still_unknown):
                if col_sums[j] > 0:
                    first_year[t] = min(first_year[t], int(y))
        print(f"  class {cls}: resolved earliest year for "
              f"{sum(1 for v in first_year.values() if v != 9999)}/{len(tokens)} tokens",
              flush=True)
        del df
        gc.collect()
    return first_year


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="1985-1989,1995-1999",
                    help="comma-separated y_start-y_end pairs")
    ap.add_argument("--min-doc-freq", type=int, default=100_000,
                    help="restrict to globally-popular tokens with doc_freq >= this")
    ap.add_argument("--top-uni", type=int, default=30)
    ap.add_argument("--top-bi",  type=int, default=20)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    vocab_df = pooled_vocab(args.min_doc_freq)
    vocab_tokens = vocab_df["token"].to_list()

    text_lines = []
    for win in args.windows.split(","):
        ys, ye = (int(x) for x in win.strip().split("-"))
        print(f"\n=== window {ys}-{ye} ===", flush=True)
        counts = early_doc_counts(vocab_tokens, ys, ye)
        absent = (vocab_df
                  .with_columns(pl.Series("early_n", [counts[t] for t in vocab_tokens]))
                  .filter(pl.col("early_n") == 0)
                  .sort("doc_freq", descending=True))
        unis = absent.filter(~pl.col("token").str.contains(" ")).head(args.top_uni)
        bis  = absent.filter(pl.col("token").str.contains(" ")).head(args.top_bi)

        all_surfaced = unis["token"].to_list() + bis["token"].to_list()
        first_year = earliest_year(all_surfaced)

        def fmt_rows(rows, label):
            out = [f"\n  --- {label} absent in {ys}-{ye}, ranked by total doc_freq ---",
                   f"  {'token':<32s}{'doc_freq':>14s}{'first_year':>12s}"]
            for t, f_ in rows.iter_rows():
                out.append(f"  {t:<32s}{f_:>14,}{first_year.get(t,9999):>12d}")
            return "\n".join(out)

        text_lines.append(f"\n=== {ys}-{ye} ===")
        text_lines.append(fmt_rows(unis.select("token","doc_freq"), "unigrams"))
        text_lines.append(fmt_rows(bis.select("token","doc_freq"),  "bigrams"))
        absent.write_csv(OUT / f"whats_new_tokens_w{ys}_{ye}.csv")
        print(text_lines[-2]); print(text_lines[-1])

    (OUT / "whats_new_tokens.txt").write_text("\n".join(text_lines))
    print(f"\n[done] wrote {OUT/'whats_new_tokens.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
