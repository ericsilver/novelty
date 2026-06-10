"""Diffusion Phase 0 (#9 + #10): LDA theme panel + kill conditions.

(#9) Fit a v0 LDA topic model (T=50) on a stratified sample of the pooled
granted corpus (009/042/039/035), transform all granted filings to a per-filing
topic distribution, and build the theme x class x year prevalence panel.

(#10) Run the PROPOSAL_diffusion §5 kill conditions:
  - SHUFFLE NULL: randomize each filing's (class, year) label; the prevalence
    panel becomes flat; real themes' temporal trajectories have positive lag-1
    autocorrelation, shuffled trajectories ~0. Mean autocorr (real) >> mean
    autocorr (shuffled) is the pass criterion.
  - AHEAD-PRECEDES-BEHIND: within each (theme, class), correlate (signed dKL,
    year) for filings carrying the theme. Negative correlation = ahead filings
    are earlier, behind filings later (the adoption-curve interpretation).

Inputs:
  data/processed/tm_class{009,042,039,035}.parquet
  data/processed/surprise_class{009,042,039,035}.parquet   (for signed dKL)
Outputs:
  data/processed/theme_panel.parquet   (theme, class, year, prevalence)
  data/processed/filing_topics.parquet (serial, top_topic, top_weight, year, nice)
  paper/results/diff3_themes.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT_JSON = REPO / "paper" / "results" / "diff3_themes.json"

CLASSES = {"009": "software", "042": "tech-svcs", "039": "transport", "035": "advert-retail"}
T = 50              # number of LDA topics
SAMPLE_PER_CLASS = 50_000   # fit sample cap per class
MIN_DF = 200
YEAR_MIN = 1990
YEAR_MAX = 2024     # truncate last year (incomplete + retro window)
TAU_TOP = 0.20      # filing "carries" theme if topic weight >= tau (or is top topic)
N_SHUFFLE = 4
RNG = np.random.default_rng(42)


def main() -> int:
    t0 = time.time()
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer

    # --- load granted filings across the 4 classes ---
    frames = []
    for c in CLASSES:
        df = pl.read_parquet(DATA / f"tm_class{c}.parquet",
                             columns=["serial_number", "filing_date",
                                      "registration_date", "goods_services",
                                      "mark_identification"])
        df = df.with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
            pl.lit(c).alias("nice"),
        ).filter(
            (pl.col("goods_services").str.len_chars() > 0) & pl.col("year").is_not_null()
            & (pl.col("year") >= YEAR_MIN) & (pl.col("year") <= YEAR_MAX)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 4)
        )
        frames.append(df)
        print(f"  class {c}: {df.height:,} granted ({YEAR_MIN}-{YEAR_MAX})", file=sys.stderr)
    rec = pl.concat(frames)
    n_total = rec.height
    print(f"[load] {n_total:,} granted filings  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # --- stratified sample per class for LDA fit ---
    sample_idx_list = []
    rec_pd_index = np.arange(n_total)
    nice_arr = rec["nice"].to_numpy()
    for c in CLASSES:
        idx_c = rec_pd_index[nice_arr == c]
        take = min(len(idx_c), SAMPLE_PER_CLASS)
        sample_idx_list.append(RNG.choice(idx_c, size=take, replace=False))
    sample_idx = np.sort(np.concatenate(sample_idx_list))
    sample_docs = rec["goods_services"][sample_idx].to_list()
    print(f"[sample] LDA fit on {len(sample_docs):,} docs", file=sys.stderr)

    # --- vocab from full pooled (min_df=200) ---
    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    vec = CountVectorizer(analyzer=analyzer, min_df=MIN_DF)
    X_full = vec.fit_transform(rec["goods_services"].to_list()).tocsr()
    V = X_full.shape[1]
    print(f"[vec ] X_full={X_full.shape} nnz={X_full.nnz:,}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    X_sample = X_full[sample_idx]

    # --- fit LDA ---
    print(f"[lda ] fitting LDA T={T} on sample...", file=sys.stderr)
    lda = LatentDirichletAllocation(n_components=T, max_iter=8, learning_method="online",
                                    batch_size=2048, random_state=42, n_jobs=1,
                                    evaluate_every=0)
    lda.fit(X_sample)
    print(f"[lda ] fit done  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # --- transform full corpus ---
    theta = lda.transform(X_full)
    top_topic = theta.argmax(axis=1).astype(np.int32)
    top_weight = theta[np.arange(theta.shape[0]), top_topic]
    print(f"[xfrm] theta={theta.shape}  ({time.time()-t0:.1f}s)", file=sys.stderr)

    years_arr = rec["year"].to_numpy().astype(np.int32)
    filings = pl.DataFrame({
        "serial_number": rec["serial_number"], "year": years_arr,
        "nice": nice_arr, "top_topic": top_topic, "top_weight": top_weight,
    })
    filings.write_parquet(DATA / "filing_topics.parquet")

    # --- theme prevalence panel via top-topic assignment ---
    panel_rows = []
    years_uniq = np.arange(YEAR_MIN, YEAR_MAX + 1)
    # also a (theme present | weight>=tau) measure
    for c in CLASSES:
        cmask = nice_arr == c
        for y in years_uniq:
            ymask = cmask & (years_arr == y)
            n_cy = int(ymask.sum())
            if n_cy < 10:
                continue
            for k in range(T):
                share_top = float((top_topic[ymask] == k).mean())
                share_tau = float((theta[ymask, k] >= TAU_TOP).mean())
                panel_rows.append({"theme": k, "nice": c, "year": int(y),
                                   "share_top": share_top, "share_tau": share_tau,
                                   "n_cy": n_cy})
    panel = pl.DataFrame(panel_rows)
    panel.write_parquet(DATA / "theme_panel.parquet")
    print(f"[panel] {panel.height:,} (theme,class,year) cells  ({time.time()-t0:.1f}s)", file=sys.stderr)

    # --- top words per topic for face validity ---
    feat = np.array(vec.get_feature_names_out())
    top_words = {}
    for k in range(T):
        top_idx = np.argsort(-lda.components_[k])[:10]
        top_words[k] = [feat[i] for i in top_idx]
    # print a few
    print("\nSample topics (top words):")
    for k in [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 49]:
        print(f"  T{k:>2}: {', '.join(top_words[k][:8])}")

    # --- shuffle null: lag-1 autocorrelation of within-class prevalence ---
    def autocorr_score(panel_arr):
        """panel_arr: theme x class -> series indexed by year (share_top)."""
        scores = []
        for k in range(T):
            for c in CLASSES:
                cell = (panel_arr[k][c] if c in panel_arr[k] else None)
                if cell is None or len(cell) < 5:
                    continue
                a = cell[:-1]; b = cell[1:]
                if a.std() == 0 or b.std() == 0:
                    continue
                scores.append(float(np.corrcoef(a, b)[0, 1]))
        return np.array(scores)

    def build_panel_arrays(topics, nices, years):
        # returns dict[k][c] = np.array prevalence by year
        out = {k: {} for k in range(T)}
        for c in CLASSES:
            cmask = nices == c
            for k in range(T):
                series = []
                for y in years_uniq:
                    ymask = cmask & (years == y)
                    n_cy = int(ymask.sum())
                    if n_cy < 10:
                        continue
                    series.append((y, float((topics[ymask] == k).mean())))
                if series:
                    arr = np.array([v for _, v in series])
                    out[k][c] = arr
        return out

    real_arrays = build_panel_arrays(top_topic, nice_arr, years_arr)
    real_ac = autocorr_score(real_arrays)
    print(f"\n[null] real mean lag-1 autocorr = {real_ac.mean():+.3f}  (n_cells={len(real_ac)})",
          file=sys.stderr)

    shuffle_means = []
    for s in range(N_SHUFFLE):
        perm = RNG.permutation(n_total)
        sh_nice = nice_arr[perm]
        sh_year = years_arr[perm]
        sh_arr = build_panel_arrays(top_topic, sh_nice, sh_year)
        sh_ac = autocorr_score(sh_arr)
        shuffle_means.append(sh_ac.mean())
        print(f"[null] shuffle {s+1}: mean autocorr = {sh_ac.mean():+.3f}", file=sys.stderr)

    sh_mean = float(np.mean(shuffle_means)); sh_sd = float(np.std(shuffle_means))
    z = (real_ac.mean() - sh_mean) / (sh_sd + 1e-9)
    print(f"[null] real {real_ac.mean():+.3f}  vs shuffle {sh_mean:+.3f}+/-{sh_sd:.3f}  z={z:+.1f}",
          file=sys.stderr)

    # --- ahead-precedes-behind: signed dKL vs year within (theme, class) ---
    sur_dkl = {}
    for c in CLASSES:
        p = DATA / f"surprise_class{c}.parquet"
        if not p.exists():
            print(f"[skip ahead-behind for {c}: no surprise file yet]", file=sys.stderr)
            continue
        s = pl.read_parquet(p, columns=["serial_number", "prospective_kl", "retrospective_kl"])
        s = s.with_columns((pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"))
        sur_dkl[c] = s.filter(pl.col("dkl").is_finite()).select("serial_number", "dkl")

    apb_rows = []
    if sur_dkl:
        full = filings.join(pl.concat(list(sur_dkl.values())), on="serial_number", how="inner")
        for c in sur_dkl:
            sub_c = full.filter(pl.col("nice") == c)
            for k in range(T):
                sub = sub_c.filter((pl.col("top_topic") == k) & (pl.col("top_weight") >= TAU_TOP))
                if sub.height < 50:
                    continue
                r = stats.spearmanr(sub["dkl"], sub["year"])
                apb_rows.append({"theme": k, "nice": c, "n": sub.height,
                                 "rho_dkl_year": float(r.statistic), "p": float(r.pvalue)})
        rho = np.array([r["rho_dkl_year"] for r in apb_rows])
        share_neg = float((rho < 0).mean()) if len(rho) else float("nan")
        print(f"\n[apb] {len(rho)} (theme,class) cells; mean rho(dkl,year) = {rho.mean():+.3f}; "
              f"share negative = {share_neg:.1%}", file=sys.stderr)

    # --- save JSON ---
    res = {
        "T": T, "sample_size": len(sample_docs), "vocab_size": int(V),
        "panel_cells": int(panel.height),
        "shuffle_null": {
            "real_mean_autocorr": float(real_ac.mean()),
            "real_n_cells": int(len(real_ac)),
            "shuffle_means": shuffle_means,
            "shuffle_mean": sh_mean, "shuffle_sd": sh_sd, "z_score": float(z),
        },
        "ahead_precedes_behind": {
            "n_cells": len(apb_rows),
            "mean_rho_dkl_year": float(rho.mean()) if len(apb_rows) else None,
            "share_negative": float((rho < 0).mean()) if len(apb_rows) else None,
            "share_neg_p05": float(((rho < 0) & (np.array([r["p"] for r in apb_rows]) < 0.05)).mean()) if len(apb_rows) else None,
        },
        "top_words": top_words,
    }
    OUT_JSON.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[done] -> {OUT_JSON}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
