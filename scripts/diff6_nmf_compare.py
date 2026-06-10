"""Diffusion §2.4 cross-method stability: NMF vs LDA on the same corpus.

Fit NMF (T=50) on the same 200k stratified sample diff3 used for LDA, with the
same vocabulary (min_df=200 unigram+bigram), transform the full 2.32M-filing
pool, and ask three questions:
  (A) Topic-word alignment: greedy Jaccard match NMF<->LDA on top-10 words.
      Themes that name the same business-model component should align across
      methods. The unaligned residual is the §2.4 firewall reject.
  (B) Per-filing top-topic agreement: adjusted Rand index between LDA's
      top_topic and NMF's top_topic across all 2.32M filings. ARI ~0 = methods
      partition independently; ARI ~1 = identical structure.
  (C) Communication with existing constructs: (i) which NMF topics best
      represent the diff2 seed phrases (blockchain, cloud, AI, ...); (ii) does
      NMF give a parallel diffusion signal -- does the SHUFFLE NULL also pass
      on NMF panels?

Inputs: data/processed/tm_class{009,042,039,035}.parquet
        data/processed/filing_topics.parquet                 (diff3 LDA)
        paper/results/diff3_themes.json                       (LDA top words)
Outputs: data/processed/filing_topics_nmf.parquet
         paper/results/diff6_nmf.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT_JSON = REPO / "paper" / "results" / "diff6_nmf.json"
CLASSES = {"009": "software", "042": "tech-svcs", "039": "transport", "035": "advert-retail"}
T = 50
SAMPLE_PER_CLASS = 50_000
MIN_DF = 200
YEAR_MIN, YEAR_MAX = 1990, 2024
RNG_SEED = 42

# Seed phrases from diff2 -- mapped to single-token searchable forms in vocab
SEED_TOKENS = {
    "blockchain": ["blockchain", "distributed ledger", "cryptocurrenc"],
    "cloud": ["cloud computing", "cloud based", "cloud storage"],
    "ai_ml": ["artificial intelligence", "machine learning"],
    "saas": ["software service", "as service"],
    "mobile_app": ["mobile app", "mobile application", "downloadable application"],
    "streaming": ["streaming", "streaming media"],
    "ar_vr": ["augmented reality", "virtual reality"],
    "iot": ["internet things", "connected device"],
    "marketplace": ["marketplace", "online marketplace"],
    "on_demand": ["demand"],
}


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    # --- load, exactly as diff3 ---
    frames = []
    for c in CLASSES:
        df = pl.read_parquet(DATA / f"tm_class{c}.parquet",
                             columns=["serial_number", "filing_date", "registration_date",
                                      "goods_services", "mark_identification"])
        df = df.with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
            pl.lit(c).alias("nice"),
        ).filter(
            (pl.col("goods_services").str.len_chars() > 0) & pl.col("year").is_not_null()
            & (pl.col("year") >= YEAR_MIN) & (pl.col("year") <= YEAR_MAX)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 4)
        )
        frames.append(df)
    rec = pl.concat(frames)
    n_total = rec.height
    nice_arr = rec["nice"].to_numpy()
    years_arr = rec["year"].to_numpy().astype(np.int32)
    print(f"[load] {n_total:,} granted filings  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # same stratified sample (same RNG seed -> same indices)
    sample_idx_list = []
    rec_idx = np.arange(n_total)
    for c in CLASSES:
        idx_c = rec_idx[nice_arr == c]
        take = min(len(idx_c), SAMPLE_PER_CLASS)
        sample_idx_list.append(rng.choice(idx_c, size=take, replace=False))
    sample_idx = np.sort(np.concatenate(sample_idx_list))

    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    vec = CountVectorizer(analyzer=analyzer, min_df=MIN_DF)
    X_full = vec.fit_transform(rec["goods_services"].to_list()).tocsr()
    V = X_full.shape[1]
    feat = np.array(vec.get_feature_names_out())
    print(f"[vec ] V={V:,} X_full.nnz={X_full.nnz:,}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # --- fit NMF on sample ---
    X_sample = X_full[sample_idx]
    print(f"[nmf ] fitting NMF T={T} on {X_sample.shape[0]:,} docs...", file=sys.stderr)
    nmf = NMF(n_components=T, init="nndsvd", solver="cd", max_iter=200,
              tol=1e-3, random_state=RNG_SEED)
    W_sample = nmf.fit_transform(X_sample)
    print(f"[nmf ] fit done  ({time.time()-t0:.1f}s); transforming full...", file=sys.stderr)
    W_full = nmf.transform(X_full)
    H = nmf.components_     # (T, V)
    top_nmf_topic = W_full.argmax(axis=1).astype(np.int32)
    top_nmf_weight = W_full[np.arange(W_full.shape[0]), top_nmf_topic]
    # normalize for prevalence semantics (W rows are non-negative weights, not probabilities)
    row_sums = W_full.sum(axis=1, keepdims=True)
    nmf_share_top = np.divide(top_nmf_weight, row_sums.ravel(),
                              out=np.zeros_like(top_nmf_weight), where=row_sums.ravel() > 0)
    print(f"[xfrm] W_full={W_full.shape}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # --- write filing_topics_nmf ---
    out_df = pl.DataFrame({
        "serial_number": rec["serial_number"], "year": years_arr, "nice": nice_arr,
        "top_nmf_topic": top_nmf_topic, "top_nmf_weight": nmf_share_top,
    })
    out_df.write_parquet(DATA / "filing_topics_nmf.parquet")

    # --- (A) topic-word alignment to LDA ---
    lda_top = json.loads((REPO / "paper" / "results" / "diff3_themes.json").read_text())["top_words"]
    lda_top = {int(k): v for k, v in lda_top.items()}
    nmf_top = {}
    for k in range(T):
        nmf_top[k] = [feat[i] for i in np.argsort(-H[k])[:10]]

    # greedy bipartite match by Jaccard on top-10 words
    used_lda = set()
    align = []
    for k_nmf in range(T):
        nset = set(nmf_top[k_nmf])
        best, best_j = None, -1.0
        for k_lda in range(T):
            if k_lda in used_lda:
                continue
            j = len(nset & set(lda_top[k_lda])) / len(nset | set(lda_top[k_lda]))
            if j > best_j:
                best_j, best = j, k_lda
        used_lda.add(best)
        align.append({"nmf": k_nmf, "lda": best, "jaccard": best_j})
    jacc = np.array([a["jaccard"] for a in align])
    print(f"\n[A] NMF<->LDA alignment (greedy Jaccard top-10 words):")
    print(f"    median Jaccard={np.median(jacc):.3f}  mean={jacc.mean():.3f}  "
          f"  share>=0.3={(jacc>=0.3).mean():.1%}  share>=0.5={(jacc>=0.5).mean():.1%}")

    # show 8 example aligned pairs (sorted by Jaccard, top + bottom)
    align_sorted = sorted(align, key=lambda a: -a["jaccard"])
    print("    --- top 5 aligned ---")
    for a in align_sorted[:5]:
        print(f"    J={a['jaccard']:.2f}  NMF[{a['nmf']:>2}] {nmf_top[a['nmf']][:6]}")
        print(f"              LDA[{a['lda']:>2}] {lda_top[a['lda']][:6]}")
    print("    --- bottom 5 (least aligned) ---")
    for a in align_sorted[-5:]:
        print(f"    J={a['jaccard']:.2f}  NMF[{a['nmf']:>2}] {nmf_top[a['nmf']][:6]}")
        print(f"              LDA[{a['lda']:>2}] {lda_top[a['lda']][:6]}")

    # --- (B) per-filing top-topic agreement ---
    lda_filings = pl.read_parquet(DATA / "filing_topics.parquet",
                                   columns=["serial_number", "top_topic"]).rename({"top_topic": "lda_top"})
    nmf_filings = pl.DataFrame({"serial_number": rec["serial_number"], "nmf_top": top_nmf_topic})
    joined = lda_filings.join(nmf_filings, on="serial_number", how="inner")
    lda_arr = joined["lda_top"].to_numpy()
    nmf_arr = joined["nmf_top"].to_numpy()
    print(f"\n[B] per-filing top-topic agreement on {joined.height:,} filings",
          file=sys.stderr)
    # sample to keep ARI tractable (sklearn ARI is O(n) but the contingency build can be slow at 2M)
    smp = rng.choice(joined.height, size=min(500_000, joined.height), replace=False)
    ari = float(adjusted_rand_score(lda_arr[smp], nmf_arr[smp]))
    print(f"    Adjusted Rand Index (500k sample) = {ari:+.4f}")

    # --- (C1) Which NMF topic best represents each diff2 seed phrase? ---
    # match by presence of seed bigrams/tokens in topic's top-50 words
    seed_hits = {}
    for phrase, alts in SEED_TOKENS.items():
        scores = np.zeros(T)
        for k in range(T):
            words = [feat[i] for i in np.argsort(-H[k])[:50]]
            scores[k] = sum(any(alt in w for w in words) for alt in alts)
        best = int(scores.argmax())
        if scores[best] == 0:
            seed_hits[phrase] = {"best_topic": None}
            continue
        # also pull the matching words from the topic
        words = [feat[i] for i in np.argsort(-H[best])[:50]]
        matched = [w for w in words if any(alt in w for alt in alts)]
        seed_hits[phrase] = {"best_topic": best, "matched_words": matched[:5],
                             "score": int(scores[best]),
                             "topic_top_words": nmf_top[best][:8]}
    print(f"\n[C1] NMF topics for diff2 seed phrases:")
    for phrase, h in seed_hits.items():
        if h["best_topic"] is None:
            print(f"    {phrase:>15}: -- no NMF topic")
        else:
            print(f"    {phrase:>15}: NMF[{h['best_topic']:>2}]  {h['matched_words']}")

    # --- (C2) shuffle-null on NMF panel ---
    print(f"\n[C2] shuffle null on NMF top-topic panel...", file=sys.stderr)
    years_uniq = np.arange(YEAR_MIN, YEAR_MAX + 1)

    def autocorr_score(topics, nices, years):
        scores = []
        for k in range(T):
            for c in CLASSES:
                cmask = nices == c
                series = []
                for y in years_uniq:
                    ymask = cmask & (years == y)
                    n = int(ymask.sum())
                    if n < 10:
                        continue
                    series.append(float((topics[ymask] == k).mean()))
                if len(series) < 5:
                    continue
                a, b = np.array(series[:-1]), np.array(series[1:])
                if a.std() == 0 or b.std() == 0:
                    continue
                scores.append(float(np.corrcoef(a, b)[0, 1]))
        return np.array(scores)

    real_ac = autocorr_score(top_nmf_topic, nice_arr, years_arr)
    print(f"    NMF real mean lag-1 autocorr = {real_ac.mean():+.3f} (n_cells={len(real_ac)})")
    sh_means = []
    for s in range(3):
        perm = rng.permutation(n_total)
        sh = autocorr_score(top_nmf_topic, nice_arr[perm], years_arr[perm])
        sh_means.append(float(sh.mean()))
    sh_mean = float(np.mean(sh_means)); sh_sd = float(np.std(sh_means) + 1e-9)
    z = (real_ac.mean() - sh_mean) / sh_sd
    print(f"    NMF shuffle = {sh_mean:+.3f} +/- {sh_sd:.3f};  z = {z:+.1f}")

    # --- output ---
    res = {
        "T": T, "vocab_size": int(V), "sample_size": int(len(sample_idx)),
        "alignment": {
            "median_jaccard": float(np.median(jacc)), "mean_jaccard": float(jacc.mean()),
            "share_ge_0.3": float((jacc >= 0.3).mean()),
            "share_ge_0.5": float((jacc >= 0.5).mean()),
            "pairs": align,
        },
        "ari": ari, "ari_n": int(len(smp)),
        "seed_phrase_topics": seed_hits,
        "nmf_shuffle_null": {
            "real_mean_autocorr": float(real_ac.mean()),
            "n_cells": int(len(real_ac)),
            "shuffle_means": sh_means,
            "shuffle_mean": sh_mean, "shuffle_sd": sh_sd, "z": float(z),
        },
        "nmf_top_words": {str(k): nmf_top[k] for k in range(T)},
    }
    OUT_JSON.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[done] -> {OUT_JSON}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
