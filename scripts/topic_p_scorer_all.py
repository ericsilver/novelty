"""Full-corpus topic-distribution scoring (all 45 classes, all filings).

Method-grade version of the Darwin/FRev operationalization: fit LDA (T=50)
on a stratified sample across every class, transform every filing
1990-2024 to a dense 50-topic distribution, and score prospective and
retrospective KL against class-year topic references (W=5, full windows,
scored years 1995-2019).

References use ALL filings (granted and not), mirroring the token scorer,
so registration analyses remain well-defined.

Outputs:
  data/processed/topic_surprise_class{NNN}.parquet  (per class)
  data/processed/topic_lda_meta.json
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"

T = 50
SAMPLE_PER_CLASS = 10_000
MIN_DF_SAMPLE = 50
YEAR_MIN, YEAR_MAX = 1990, 2024
W = 5
SCORE_MIN, SCORE_MAX = YEAR_MIN + W, YEAR_MAX - W
RNG = np.random.default_rng(42)
CHUNK = 200_000


def kl_rows(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (P * (np.log(P) - np.log(q)[None, :])).sum(axis=1)


def all_classes() -> list[str]:
    return sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def load_class_docs(cls: str) -> pl.DataFrame:
    return pl.read_parquet(
        PROC / f"tm_class{cls}.parquet",
        columns=["serial_number", "filing_date", "goods_services"],
    ).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
    ).filter(
        (pl.col("goods_services").str.len_chars() > 0)
        & pl.col("year").is_between(YEAR_MIN, YEAR_MAX)
    ).select("serial_number", "year", "goods_services")


def main() -> int:
    t0 = time.time()
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer

    classes = all_classes()

    # ---- stratified sample for vocab + LDA fit ----
    sample_docs: list[str] = []
    for cls in classes:
        d = load_class_docs(cls)
        take = min(d.height, SAMPLE_PER_CLASS)
        idx = np.sort(RNG.choice(d.height, size=take, replace=False))
        sample_docs.extend(d["goods_services"][idx].to_list())
        del d
        gc.collect()
    print(f"[sample] {len(sample_docs):,} docs ({time.time()-t0:.0f}s)",
          file=sys.stderr, flush=True)

    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    vec = CountVectorizer(analyzer=analyzer, min_df=MIN_DF_SAMPLE)
    X_sample = vec.fit_transform(sample_docs)
    V = X_sample.shape[1]
    print(f"[vocab] V={V:,} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    lda = LatentDirichletAllocation(n_components=T, max_iter=8,
                                    learning_method="online", batch_size=2048,
                                    random_state=42, n_jobs=1, evaluate_every=0)
    lda.fit(X_sample)
    del X_sample, sample_docs
    gc.collect()
    print(f"[lda ] fit done ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    feature_names = vec.get_feature_names_out()
    top_words = {int(k): [feature_names[i] for i in comp.argsort()[-8:][::-1]]
                 for k, comp in enumerate(lda.components_)}
    (PROC / "topic_lda_meta.json").write_text(json.dumps(
        {"T": T, "V": int(V), "sample_per_class": SAMPLE_PER_CLASS,
         "min_df_sample": MIN_DF_SAMPLE, "top_words": top_words},
        indent=1))

    # ---- per class: transform + score ----
    for ci, cls in enumerate(classes, 1):
        out_path = PROC / f"topic_surprise_class{cls}.parquet"
        if out_path.exists():
            print(f"[{ci}/{len(classes)}] class {cls}: exists, skip",
                  file=sys.stderr, flush=True)
            continue
        d = load_class_docs(cls)
        n = d.height
        thetas = []
        for i in range(0, n, CHUNK):
            Xc = vec.transform(d["goods_services"][i:i + CHUNK].to_list())
            thetas.append(lda.transform(Xc).astype(np.float64))
            del Xc
            gc.collect()
        theta = np.vstack(thetas)
        del thetas
        gc.collect()
        theta = np.clip(theta, 1e-12, None)
        theta /= theta.sum(axis=1, keepdims=True)

        years = d["year"].to_numpy()
        year_means: dict[int, np.ndarray] = {}
        year_counts: dict[int, int] = {}
        for y in range(YEAR_MIN, YEAR_MAX + 1):
            m = years == y
            c = int(m.sum())
            if c:
                year_means[y] = theta[m].mean(axis=0)
                year_counts[y] = c

        pros = np.full(n, np.nan)
        retr = np.full(n, np.nan)
        for y in range(SCORE_MIN, SCORE_MAX + 1):
            past = [yy for yy in range(y - W, y) if yy in year_means]
            fut = [yy for yy in range(y + 1, y + W + 1) if yy in year_means]
            if len(past) < W or len(fut) < W:
                continue
            wp = np.array([year_counts[yy] for yy in past], dtype=np.float64)
            q_p = np.average([year_means[yy] for yy in past], axis=0, weights=wp)
            wf = np.array([year_counts[yy] for yy in fut], dtype=np.float64)
            q_f = np.average([year_means[yy] for yy in fut], axis=0, weights=wf)
            q_p = np.clip(q_p, 1e-12, None); q_p /= q_p.sum()
            q_f = np.clip(q_f, 1e-12, None); q_f /= q_f.sum()
            m = years == y
            P = theta[m]
            pros[m] = kl_rows(P, q_p)
            retr[m] = kl_rows(P, q_f)

        pl.DataFrame({
            "serial_number": d["serial_number"],
            "year": d["year"],
            "topic_pros": pros,
            "topic_retr": retr,
        }).with_columns(
            (pl.col("topic_pros") - pl.col("topic_retr")).alias("topic_dkl")
        ).write_parquet(out_path)
        n_scored = int(np.isfinite(pros).sum())
        print(f"[{ci}/{len(classes)}] class {cls}: {n:,} filings, "
              f"{n_scored:,} scored ({time.time()-t0:.0f}s)",
              file=sys.stderr, flush=True)
        del d, theta, pros, retr
        gc.collect()

    print(f"[done] ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
