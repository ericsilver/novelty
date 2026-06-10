"""Topic-distribution P (DeDeo suggestion #2): re-score prospective and
retrospective KL on each filing's LDA topic distribution rather than its
token distribution, then test whether the renewal validation reproduces and
whether the short-document length artifact dissolves.

Replicates diff3_lda_themes.py's fit exactly (same corpus filter, sample,
seed, vectorizer, and LDA params) so topics match the paper's theme set.

References: Q_past(c, t) = mean theta over granted filings in class c,
years [t-5, t-1]; Q_future symmetric. Full-window years only (1995-2019).

Outputs:
  data/processed/filing_topic_surprise.parquet   (per-filing topic KL scores)
  paper/results/topic_p_validation.json          (concordance + renewal stats)
  paper/results/topic_p_renewal_compare.png      (token vs topic dKL renewal)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

CLASSES = {"009": "software", "042": "tech-svcs", "039": "transport", "035": "advert-retail"}
T = 50
SAMPLE_PER_CLASS = 50_000
MIN_DF = 200
YEAR_MIN, YEAR_MAX = 1990, 2024
W = 5
SCORE_MIN, SCORE_MAX = YEAR_MIN + W, YEAR_MAX - W  # full windows only
RNG = np.random.default_rng(42)

OUT_PARQUET = DATA / "filing_topic_surprise.parquet"
OUT_JSON = RES / "topic_p_validation.json"
OUT_PNG = RES / "topic_p_renewal_compare.png"


def kl_rows(P: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(P_i || q) per row; P rows and q are positive and sum to 1."""
    return (P * (np.log(P) - np.log(q)[None, :])).sum(axis=1)


def main() -> int:
    t0 = time.time()
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer

    # --- load granted filings (diff3-identical) ---
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
    rec = pl.concat(frames)
    n_total = rec.height
    print(f"[load] {n_total:,} granted filings ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)

    # --- stratified sample (diff3-identical RNG draw order) ---
    sample_idx_list = []
    rec_pd_index = np.arange(n_total)
    nice_arr = rec["nice"].to_numpy()
    for c in CLASSES:
        idx_c = rec_pd_index[nice_arr == c]
        take = min(len(idx_c), SAMPLE_PER_CLASS)
        sample_idx_list.append(RNG.choice(idx_c, size=take, replace=False))
    sample_idx = np.sort(np.concatenate(sample_idx_list))

    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    vec = CountVectorizer(analyzer=analyzer, min_df=MIN_DF)
    X_full = vec.fit_transform(rec["goods_services"].to_list()).tocsr()
    print(f"[vec ] X_full={X_full.shape} ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)

    lda = LatentDirichletAllocation(n_components=T, max_iter=8, learning_method="online",
                                    batch_size=2048, random_state=42, n_jobs=1,
                                    evaluate_every=0)
    lda.fit(X_full[sample_idx])
    print(f"[lda ] fit done ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)

    # --- transform full corpus in chunks ---
    chunks = []
    step = 200_000
    for i in range(0, n_total, step):
        chunks.append(lda.transform(X_full[i:i + step]).astype(np.float64))
        print(f"[xfrm] {min(i+step, n_total):,}/{n_total:,} ({time.time()-t0:.1f}s)",
              file=sys.stderr, flush=True)
    theta = np.vstack(chunks)
    del chunks, X_full
    gc.collect()
    # guard against numerical zeros before log
    theta = np.clip(theta, 1e-12, None)
    theta /= theta.sum(axis=1, keepdims=True)

    years = rec["year"].to_numpy()
    nice = rec["nice"].to_numpy()

    # --- per (class, year) reference distributions and scoring ---
    pros = np.full(n_total, np.nan)
    retr = np.full(n_total, np.nan)
    for c in CLASSES:
        mask_c = nice == c
        yrs_c = years[mask_c]
        th_c = theta[mask_c]
        # per-year mean theta for this class
        year_means: dict[int, np.ndarray] = {}
        year_counts: dict[int, int] = {}
        for y in range(YEAR_MIN, YEAR_MAX + 1):
            m = yrs_c == y
            n = int(m.sum())
            if n:
                year_means[y] = th_c[m].mean(axis=0)
                year_counts[y] = n
        p_c = np.full(len(yrs_c), np.nan)
        r_c = np.full(len(yrs_c), np.nan)
        for y in range(SCORE_MIN, SCORE_MAX + 1):
            past_years = [yy for yy in range(y - W, y) if yy in year_means]
            fut_years = [yy for yy in range(y + 1, y + W + 1) if yy in year_means]
            if len(past_years) < W or len(fut_years) < W:
                continue
            wp = np.array([year_counts[yy] for yy in past_years], dtype=np.float64)
            q_past = np.average([year_means[yy] for yy in past_years], axis=0, weights=wp)
            wf = np.array([year_counts[yy] for yy in fut_years], dtype=np.float64)
            q_fut = np.average([year_means[yy] for yy in fut_years], axis=0, weights=wf)
            q_past = np.clip(q_past, 1e-12, None); q_past /= q_past.sum()
            q_fut = np.clip(q_fut, 1e-12, None); q_fut /= q_fut.sum()
            m = yrs_c == y
            P = th_c[m]
            p_c[m] = kl_rows(P, q_past)
            r_c[m] = kl_rows(P, q_fut)
        pros[mask_c] = p_c
        retr[mask_c] = r_c
        print(f"[score] class {c} done ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)

    out = pl.DataFrame({
        "serial_number": rec["serial_number"],
        "nice": rec["nice"],
        "year": rec["year"],
        "topic_pros": pros,
        "topic_retr": retr,
    }).with_columns((pl.col("topic_pros") - pl.col("topic_retr")).alias("topic_dkl"))
    out.write_parquet(OUT_PARQUET)
    scored = out.filter(pl.col("topic_dkl").is_not_null() & pl.col("topic_dkl").is_finite())
    print(f"[save] {scored.height:,} scored filings -> {OUT_PARQUET}", file=sys.stderr, flush=True)

    # ================= validation =================
    results: dict = {"n_scored": scored.height}

    # --- concordance + length artifact vs token-based scores ---
    tok_parts = []
    for c in CLASSES:
        tok_parts.append(pl.read_parquet(
            DATA / f"surprise_class{c}.parquet",
            columns=["serial_number", "n_terms", "prospective_kl", "retrospective_kl",
                     "n_ref_prospective", "n_ref_retrospective"]))
    tok = pl.concat(tok_parts).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("token_dkl"))
    j = scored.join(tok, on="serial_number", how="inner").filter(
        pl.col("token_dkl").is_finite()
        & (pl.col("n_ref_prospective") >= 1000) & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3))
    results["n_joined"] = j.height
    jd = j.select("token_dkl", "topic_dkl", "topic_pros", "topic_retr",
                  "prospective_kl", "retrospective_kl", "n_terms").to_pandas()
    results["corr_token_topic_dkl_pearson"] = float(jd["token_dkl"].corr(jd["topic_dkl"]))
    results["corr_token_topic_dkl_spearman"] = float(
        jd["token_dkl"].corr(jd["topic_dkl"], method="spearman"))
    results["corr_topic_pros_retr"] = float(jd["topic_pros"].corr(jd["topic_retr"]))
    results["corr_token_pros_retr"] = float(
        jd["prospective_kl"].corr(jd["retrospective_kl"]))

    # length artifact: mean |dkl| by n_terms band, token vs topic
    bands = [(3, 4), (5, 9), (10, 19), (20, 39), (40, 10_000)]
    la = []
    for lo, hi in bands:
        m = (jd["n_terms"] >= lo) & (jd["n_terms"] <= hi)
        la.append({
            "band": f"{lo}-{hi if hi < 10_000 else '+'}",
            "n": int(m.sum()),
            "token_abs_dkl": float(jd.loc[m, "token_dkl"].abs().mean()),
            "topic_abs_dkl": float(jd.loc[m, "topic_dkl"].abs().mean()),
        })
    results["length_artifact"] = la

    # --- renewal validation, conditional on registration ---
    status_parts = []
    for c in CLASSES:
        status_parts.append(pl.read_parquet(
            DATA / f"tm_class{c}.parquet", columns=["serial_number", "status_code"]))
    st = pl.concat(status_parts).with_columns(
        pl.col("status_code").str.slice(0, 1).alias("b")).with_columns(
        ((pl.col("b") == "6") | (pl.col("b") == "8")).alias("reached_registration"),
        (pl.col("b") == "6").alias("renewed"))
    jr = j.join(st, on="serial_number", how="inner").filter(
        pl.col("year").is_between(1995, 2019))
    reg = jr.filter(pl.col("reached_registration"))
    results["n_registered"] = reg.height

    def quintiles(frame: pl.DataFrame, var: str, outcome: str) -> dict:
        arr = frame[var].to_numpy()
        cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(cuts, arr)
        g = frame.with_columns(pl.Series("q", qi)).group_by("q").agg(
            pl.len().alias("n"),
            pl.col(outcome).cast(pl.Float64).mean().alias("rate")).sort("q")
        d = {f"q{int(r['q'])+1}": r["rate"] for r in g.iter_rows(named=True)}
        d["lift_q5_q1"] = d["q5"] - d["q1"]
        return d

    results["renewal_by_token_dkl"] = quintiles(reg, "token_dkl", "renewed")
    results["renewal_by_topic_dkl"] = quintiles(reg, "topic_dkl", "renewed")
    per_class = {}
    for c in CLASSES:
        rc = reg.filter(pl.col("nice") == c)
        if rc.height >= 5000:
            per_class[c] = {
                "n": rc.height,
                "token": quintiles(rc, "token_dkl", "renewed"),
                "topic": quintiles(rc, "topic_dkl", "renewed"),
            }
    results["renewal_per_class"] = per_class

    OUT_JSON.write_text(json.dumps(results, indent=1, default=float))
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("renewal_per_class", "length_artifact")},
                     indent=1, default=float), file=sys.stderr, flush=True)

    # --- comparison figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, key, title in [(axes[0], "renewal_by_token_dkl", "Token-distribution $\\Delta$KL"),
                           (axes[1], "renewal_by_topic_dkl", "Topic-distribution $\\Delta$KL (T=50)")]:
        d = results[key]
        xs = np.arange(1, 6)
        ys = [d[f"q{i}"] for i in xs]
        ax.plot(xs, ys, "o-", lw=2, color="#2b6cb0")
        ax.set_xticks(xs)
        ax.set_xlabel("$\\Delta$KL quintile")
        ax.set_title(f"{title}\nlift Q5$-$Q1 = {d['lift_q5_q1']*100:+.1f}pp")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("P(survived 5y | registration)")
    fig.suptitle("Conditional renewal under token-P vs topic-P scoring "
                 f"(four-class pool, n={reg.height:,} registered)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"[done] ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
