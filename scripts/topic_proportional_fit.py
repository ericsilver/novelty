"""Refit the topic model on a filing-proportional sample instead of a class cap.

The production fit draws at most 10,000 filings from each Nice class. Because
classes differ in size by more than two orders of magnitude, that cap turns into
a sampling rate running from 0.63% in class 009 to 100% in class 023 -- a
158-fold spread, inversely related to class size. Themes are then allocated
roughly per class rather than per filing, and vocabulary concentrated in the
largest classes is thinned before the model sees it. Measured against the
corpus, the fit sample holds blockchain vocabulary at 0.23% against 0.62%, AI at
0.39% against 0.95%, cloud at 0.39% against 2.52%.

scripts/theme_bundles.py shows the consequence: at T=500 there is no AI theme,
no blockchain theme and no cloud theme, while solar -- the one concept that
spreads over many small classes and so enters the sample at its true rate -- has
several clean ones.

That is a diagnosis, and it cannot be confirmed by comparing classes, because a
class's sampling rate is a deterministic function of its size: the two are
perfectly collinear and no cross-class regression can separate thinning from
whatever else varies with class size. It can be confirmed by intervention.

This draws the same number of documents, allocated in proportion to each class's
share of filings rather than equally, and fits at the same resolution with the
same vocabulary settings and the same number of passes. If the diagnosis is
right, AI, blockchain and cloud themes appear. If they do not, the cap was never
the reason and the explanation has to be looked for elsewhere.

A floor of MIN_PER_CLASS keeps the smallest classes from vanishing entirely,
which would trade one distortion for its mirror image.

Usage:  python scripts/topic_proportional_fit.py [T]
Output: data/processed/topic_model_T{T}_prop.joblib
        paper/results/topic_proportional_fit.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
TOTAL = 448_437          # the production sample size, held fixed
MIN_PER_CLASS = 500
MIN_DF_SAMPLE = 20

# The concepts the diagnosis makes a prediction about, checked after the fit.
PROBES = {
    "blockchain": ["blockchain", "blockchains", "cryptocurrency",
                   "cryptocurrencies", "bitcoin", "distributed ledger",
                   "digital currency", "non-fungible token"],
    "ai": ["artificial intelligence", "machine learning", "deep learning",
           "neural networks", "computer vision", "predictive analytics"],
    "cloud": ["cloud computing", "saas", "software as a service",
              "virtualization", "data center"],
    "internet": ["internet", "online", "website", "web site", "e-commerce"],
    "solar": ["solar", "solar cells", "solar energy", "photovoltaic"],
}


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def build_sample() -> tuple[sp.csr_matrix, dict]:
    xp = PROC / "_lda_prop_matrix.npz"
    vp = PROC / "_lda_prop_vocab.joblib"
    if xp.exists() and vp.exists():
        log("[sample] reusing cached proportional design matrix")
        return sp.load_npz(xp), joblib.load(vp)

    sizes = {}
    for c in CLASSES:
        f = PROC / f"tm_class{c}.parquet"
        if not f.exists():
            continue
        sizes[c] = (pl.scan_parquet(f)
                    .select(pl.col("filing_date").str.slice(0, 4)
                            .cast(pl.Int32, strict=False).alias("y"),
                            pl.col("goods_services"))
                    .filter((pl.col("goods_services").str.len_chars() > 0)
                            & pl.col("y").is_between(1990, 2024))
                    .select(pl.len()).collect().item())
    grand = sum(sizes.values())
    # Proportional allocation with a floor, then rescaled so the total is held.
    raw = {c: max(MIN_PER_CLASS, int(TOTAL * n / grand)) for c, n in sizes.items()}
    scale = TOTAL / sum(raw.values())
    take = {c: min(sizes[c], max(MIN_PER_CLASS, int(raw[c] * scale)))
            for c in raw}
    log(f"[sample] {grand:,} eligible filings; drawing {sum(take.values()):,}")
    log(f"[sample] class 009 gets {take['009']:,} "
        f"({take['009']/sizes['009']:.2%}) against 10,000 (0.63%) under the cap")

    rng = np.random.default_rng(42)
    docs = []
    for c in CLASSES:
        if c not in sizes:
            continue
        d = pl.read_parquet(
            PROC / f"tm_class{c}.parquet",
            columns=["filing_date", "goods_services"]).with_columns(
            pl.col("filing_date").str.slice(0, 4)
            .cast(pl.Int32, strict=False).alias("y")
        ).filter((pl.col("goods_services").str.len_chars() > 0)
                 & pl.col("y").is_between(1990, 2024))
        k = min(d.height, take[c])
        idx = np.sort(rng.choice(d.height, size=k, replace=False))
        docs.extend(d["goods_services"][idx].to_list())
        del d
        gc.collect()
    log(f"[sample] {len(docs):,} documents")

    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer
    vec = CountVectorizer(analyzer=_make_analyzer(frozenset(STOPWORDS), (1, 2)),
                          min_df=MIN_DF_SAMPLE)
    X = vec.fit_transform(docs)
    vocab = vec.vocabulary_
    del docs
    gc.collect()
    sp.save_npz(xp, X)
    joblib.dump(vocab, vp)
    return X, vocab


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    out = PROC / f"topic_model_T{T}_prop.joblib"
    X, vocab = build_sample()
    log(f"[fit] T={T} on {X.shape[0]:,} x {X.shape[1]:,}, nnz {X.nnz:,}")

    if out.exists():
        log("[fit] model exists, skipping to the probe")
        m = joblib.load(out)
        lda = m["lda"]
    else:
        ck = PROC / f"_lda_ckpt_T{T}_prop.joblib"
        N_BLOCKS = 8
        bounds = np.linspace(0, X.shape[0], N_BLOCKS + 1).astype(int)
        if ck.exists():
            st = joblib.load(ck)
            lda, ep, blk = st["lda"], st["epochs"], st["block"]
            log(f"[fit] resuming at epoch {ep}/8 block {blk}/{N_BLOCKS}")
        else:
            lda = LatentDirichletAllocation(
                n_components=T, max_iter=1, learning_method="online",
                batch_size=2048, random_state=42, n_jobs=1, evaluate_every=0)
            ep, blk = 0, 0
        budget = float(os.environ.get("FIT_BUDGET_MIN", "600"))
        t0, nb = time.time(), 0
        while ep < 8:
            lda.partial_fit(X[bounds[blk]:bounds[blk + 1]])
            blk += 1
            nb += 1
            if blk == N_BLOCKS:
                blk, ep = 0, ep + 1
            joblib.dump({"lda": lda, "epochs": ep, "block": blk}, ck, compress=0)
            el = (time.time() - t0) / 60
            if ep < 8 and el + el / nb > budget:
                log(f"[fit] stopped at epoch {ep}/8 block {blk}; rerun to continue")
                return 2
            if blk == 0:
                log(f"[fit] epoch {ep}/8 ({el:.1f} min)")
        joblib.dump({"vocabulary": vocab, "lda": lda}, out, compress=3)
        ck.unlink(missing_ok=True)
        log(f"[fit] complete -> {out.name}")

    # The prediction: concepts thinned by the cap should now hold a theme.
    comp = lda.components_
    inv = {v: k for k, v in vocab.items()}
    share = comp / comp.sum(axis=1, keepdims=True)
    res = {"T": T, "sample": int(X.shape[0]), "probes": {}}
    print("")
    print(f"{'concept':<12}{'best theme mass':>17}   top words of that theme")
    for name, terms in PROBES.items():
        idx = [vocab[t] for t in terms if t in vocab]
        if not idx:
            continue
        mass = share[:, idx].sum(axis=1)
        k = int(np.argmax(mass))
        words = ", ".join(inv[j] for j in np.argsort(comp[k])[::-1][:8])
        res["probes"][name] = {"theme": k, "mass": float(mass[k]),
                               "n_themes_above_015": int((mass >= 0.015).sum()),
                               "words": words}
        print(f"{name:<12}{mass[k]:>17.3f}   {words}")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "topic_proportional_fit.json").write_text(json.dumps(res, indent=1))
    print("")
    print("[done] topic_proportional_fit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
