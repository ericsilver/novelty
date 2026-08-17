"""Does the number of themes change the answer?

The primary scoring groups a 62,168-term vocabulary into 50 themes, which is
coarse: it is roughly one theme per industry, and a reader is entitled to ask
whether the findings are a property of the corpus or of that partition. Raising
T does not add coverage -- the vocabulary is fixed, so a larger T re-partitions
the same words more finely -- but it changes what counts as a filing being far
from its industry, and it could change the sign of anything.

Two modes:

  compare   Read whatever per-filing scorings already exist at each resolution
            and report, for each, the first-gate lead contrast and the
            correlation of lead with the production T = 50 build. Cheap.

  build     Fit an LDA at a resolution that has no model, score one class under
            per-filing reference windows, and write the scores so `compare` can
            pick them up. Expensive: fitting is roughly linear in T, about
            25 minutes at T = 100 and six hours at T = 2500 on the full
            448,437-filing stratified sample.

The sweep scores class 009 (software and electronics, 1.81M filings) rather
than all 45, because a full re-score at every resolution is days of compute and
009 is the class every other diagnostic in this paper uses. Where a resolution
already exists corpus-wide -- T = 50 and T = 200 do -- the comparison is run on
all 45 classes as well, and the two are reported separately so the reader can
see whether the single class behaves like the corpus.

Usage:
  python scripts/topic_resolution_sweep.py compare
  python scripts/topic_resolution_sweep.py build 100 500 1000 2500

Output: paper/results/topic_resolution_sweep.json
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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

CLASSES = [f"{i:03d}" for i in range(1, 46)]
SWEEP_CLASS = "009"
SAMPLE_PER_CLASS = 10_000
MIN_DF_SAMPLE = 50
WDAYS = 1826
MIN_REF = 500
YEAR_LO, YEAR_HI = 1995, 2019
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
EPS = 1e-12
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def model_path(T: int) -> Path:
    return PROC / ("topic_model.joblib" if T == 50 else f"topic_model_T{T}.joblib")


def score_path(cls: str, T: int) -> Path:
    return PROC / ("rolling_surprise_class%s.parquet" % cls if T == 50
                   else f"rolling_surprise_class{cls}_T{T}.parquet")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def fit_model(T: int) -> bool:
    """Fit an LDA at resolution T exactly as the production T=50 model was fitted.

    Matching matters more than convenience here: the point of the sweep is to
    vary T and nothing else. The production fit draws a random 10,000 filings
    per class under seed 42 and builds the vocabulary with the stopword-aware
    analyzer from novelty.dictionary. Sampling with head() and a plain token
    pattern instead gives 34,720 terms against the production 62,168, which
    would confound resolution with vocabulary.

    Note that the production pipeline fits the vocabulary with that analyzer but
    transforms filings with a plain token pattern keyed to the stored
    vocabulary (rolling_rescore_all.py). The sweep reproduces both halves as
    they are, so its scores are comparable to the paper's.
    """
    if model_path(T).exists():
        log(f"[fit T={T}] model exists, skipping")
        return True

    # Fitting is resumable. A T=2500 fit is hours of work, longer than this
    # environment lets a single background job live, so the design matrix is
    # cached once and the model is checkpointed after every epoch. Re-running
    # picks up where the last run stopped; production's max_iter=8 is
    # reproduced as eight explicit passes so the result is the same object.
    xp = PROC / "_lda_fit_matrix.npz"
    vp = PROC / "_lda_fit_vocab.joblib"
    if xp.exists() and vp.exists():
        import scipy.sparse as sp
        X = sp.load_npz(xp)
        vocab = joblib.load(vp)
        log(f"[fit T={T}] reusing cached design matrix {X.shape[0]:,} x {X.shape[1]:,}")
    else:
        sys.path.insert(0, str(REPO / "src"))
        from novelty.dictionary import STOPWORDS, _make_analyzer
        rng = np.random.default_rng(42)
        docs = []
        for c in CLASSES:
            f = PROC / f"tm_class{c}.parquet"
            if not f.exists():
                continue
            d = pl.read_parquet(
                f, columns=["filing_date", "goods_services"]).with_columns(
                pl.col("filing_date").str.slice(0, 4)
                .cast(pl.Int32, strict=False).alias("y")
            ).filter((pl.col("goods_services").str.len_chars() > 0)
                     & pl.col("y").is_between(1990, 2024))
            take = min(d.height, SAMPLE_PER_CLASS)
            idx = np.sort(rng.choice(d.height, size=take, replace=False))
            docs.extend(d["goods_services"][idx].to_list())
            del d
            gc.collect()
        log(f"[fit T={T}] {len(docs):,} sampled filings")
        vec = CountVectorizer(analyzer=_make_analyzer(frozenset(STOPWORDS), (1, 2)),
                              min_df=MIN_DF_SAMPLE)
        X = vec.fit_transform(docs)
        vocab = vec.vocabulary_
        del docs
        gc.collect()
        import scipy.sparse as sp
        sp.save_npz(xp, X)
        joblib.dump(vocab, vp)
    log(f"[fit T={T}] {X.shape[0]:,} x {X.shape[1]:,}, nnz {X.nnz:,}")

    ck = PROC / f"_lda_ckpt_T{T}.joblib"
    # Checkpoint inside the epoch, not just between epochs. A single T=2500
    # pass over 448k filings takes longer than a background job is allowed to
    # live, so progress is tracked as (epoch, block) and any run resumes at the
    # block it stopped on. partial_fit over blocks is the online algorithm's
    # own update unit, so eight passes in blocks equal max_iter=8.
    N_BLOCKS = 8
    bounds = np.linspace(0, X.shape[0], N_BLOCKS + 1).astype(int)
    if ck.exists():
        st = joblib.load(ck)
        lda, ep, blk = st["lda"], st["epochs"], st.get("block", 0)
        log(f"[fit T={T}] resuming at epoch {ep}/8, block {blk}/{N_BLOCKS}")
    else:
        lda = LatentDirichletAllocation(
            n_components=T, max_iter=1, learning_method="online",
            batch_size=2048, random_state=42, n_jobs=1, evaluate_every=0)
        ep, blk = 0, 0
    budget = float(os.environ.get("FIT_BUDGET_MIN", "18"))
    t0 = time.time()
    nb = 0
    while ep < 8:
        lda.partial_fit(X[bounds[blk]:bounds[blk + 1]])
        blk += 1
        nb += 1
        if blk == N_BLOCKS:
            blk = 0
            ep += 1
        el = (time.time() - t0) / 60
        per = el / nb
        if ep < 8 and el + per > budget:
            joblib.dump({"lda": lda, "epochs": ep, "block": blk}, ck, compress=0)
            log(f"[fit T={T}] stopped at epoch {ep}/8 block {blk}/{N_BLOCKS} "
                f"after {el:.1f} min ({per:.1f} min/block); rerun to continue")
            return False
        if blk == 0:
            log(f"[fit T={T}] epoch {ep}/8 done ({el:.1f} min this run)")
        joblib.dump({"vocabulary": vocab, "lda": lda}, model_path(T), compress=3)
    ck.unlink(missing_ok=True)
    log(f"[fit T={T}] complete")
    del X, lda
    gc.collect()
    return True


def kl_stream(theta, lo, hi, T):
    """KL of every row against the mean of its window, without materialising Q.

    The flat scorer builds an n x T array of window means and then takes the
    divergence. At T = 2500 on a 1.8M-filing class that array is 36 GB, so the
    two are fused here: the running sum is carried in a single T-vector and each
    row's divergence is taken as the pointers pass it. Peak memory is theta plus
    a few vectors, and theta itself is a float32 memmap, so nothing large is
    ever resident. Both pointers advance monotonically, so the memmap is read
    strictly forward and pages cleanly.
    """
    n = theta.shape[0]
    kl = np.full(n, np.nan)
    cnt = (hi - lo).astype(np.int64)
    run = np.zeros(T, dtype=np.float64)
    cl = ch = 0
    for i in range(n):
        while ch < hi[i]:
            run += theta[ch]; ch += 1
        while cl < lo[i]:
            run -= theta[cl]; cl += 1
        if cnt[i] <= 0:
            continue
        q = run / cnt[i]
        np.clip(q, EPS, None, out=q)
        q /= q.sum()
        p = theta[i].astype(np.float64)
        kl[i] = float(np.dot(p, np.log(p) - np.log(q)))
    return kl, cnt


def score_class(cls: str, T: int) -> None:
    if score_path(cls, T).exists():
        log(f"[score T={T}] {cls} exists, skipping")
        return
    m = joblib.load(model_path(T))
    lda, vocab = m["lda"], m["vocabulary"]
    df = pl.read_parquet(
        PROC / f"tm_class{cls}.parquet",
        columns=["serial_number", "filing_date", "goods_services"]).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd")
    ).drop_nulls("fd").sort("fd")
    n = df.height
    vec = CountVectorizer(vocabulary=vocab, lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    # float32 memmap: at T = 2500 a dense float64 theta for this class is 36 GB.
    mm = PROC / f"_theta_{cls}_T{T}.f32"
    theta = np.memmap(mm, dtype=np.float32, mode="w+", shape=(n, T))
    texts = df["goods_services"].to_list()
    t0 = time.time()
    for s in range(0, n, 100_000):
        e = min(s + 100_000, n)
        th = lda.transform(vec.transform(texts[s:e]))
        np.clip(th, EPS, None, out=th)
        th /= th.sum(axis=1, keepdims=True)
        theta[s:e] = th.astype(np.float32)
        if s % 500_000 == 0:
            log(f"    [T={T}] transformed {e:,}/{n:,} ({(time.time()-t0)/60:.1f} min)")
    del texts, lda, m
    gc.collect()
    theta.flush()

    days = df["fd"].to_numpy().astype("datetime64[D]").astype(np.int64)
    years = df["fd"].dt.year().to_numpy()
    lo_p = np.searchsorted(days, days - WDAYS, side="left")
    hi_p = np.searchsorted(days, days, side="left")
    lo_f = np.searchsorted(days, days, side="right")
    hi_f = np.searchsorted(days, days + WDAYS, side="right")
    k_past, n_p = kl_stream(theta, lo_p, hi_p, T)
    k_fut, n_f = kl_stream(theta, lo_f, hi_f, T)
    ok = ((n_p >= MIN_REF) & (n_f >= MIN_REF)
          & (days - WDAYS >= days[0]) & (days + WDAYS <= days[-1])
          & (years >= YEAR_LO) & (years <= YEAR_HI))
    df.select("serial_number").with_columns(
        pl.Series("year", years),
        pl.Series("topic_kl_vs_past", np.where(ok, k_past, np.nan)),
        pl.Series("topic_kl_vs_future", np.where(ok, k_fut, np.nan)),
        pl.Series("topic_dkl", np.where(ok, k_past - k_fut, np.nan)),
        pl.Series("n_ref_past", n_p),
        pl.Series("n_ref_future", n_f),
    ).write_parquet(score_path(cls, T))
    log(f"[score T={T}] {cls}: {int(ok.sum()):,} scored of {n:,} "
        f"({(time.time()-t0)/60:.1f} min)")
    del theta, df
    gc.collect()
    mm.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
def gate_frame(classes: list[str], T: int) -> pl.DataFrame | None:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())
    parts = []
    for c in classes:
        sp = score_path(c, T)
        tp = PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()
    if not parts:
        return None
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("A"),
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry"))
    d = d.filter(pl.col("ry").is_between(REG_LO, REG_HI))
    return d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"),
        pl.concat_str([pl.col("cls"), pl.col("ry").cast(pl.Utf8)],
                      separator="-").alias("cell"))


def contrast(d: pl.DataFrame, col: str) -> dict | None:
    if d.height < 5000:
        return None
    s = d.sort(["cell", col, "serial_number"]).with_columns(
        ((pl.col(col).rank("ordinal").over("cell") - 1) * 5
         // pl.len().over("cell")).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"),
                            pl.len().alias("n")).sort("q")
    if g.height < 5:
        return None
    v = [float(r["p"]) for r in g.iter_rows(named=True)]
    p1, p5 = v[0], v[4]
    n1, n5 = int(g["n"][0]), int(g["n"][4])
    se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
    return {"n": int(d.height), "base": float(d["failed"].mean()),
            "lift": p5 - p1, "se": se, "t": (p5 - p1) / se if se else None}


def compare(Ts: list[int]) -> dict:
    out = {"sweep_class": SWEEP_CLASS, "all_classes": {}, "class009": {}}
    # A resolution only enters the corpus-wide panel if it was scored for
    # (nearly) every class; the high-T builds cover one class, and reporting
    # those two files as "all 45 classes" would be a different sample wearing
    # the same label.
    full = [T for T in Ts
            if sum((score_path(c, T)).exists() for c in CLASSES) >= 40]
    log(f"[compare] corpus-wide at: {full}; single-class at: {Ts}")
    for label, classes, key, use in (("all 45 classes", CLASSES, "all_classes", full),
                                     (f"class {SWEEP_CLASS}", [SWEEP_CLASS],
                                      "class009", Ts)):
        ref = None
        log(f"\n=== {label} ===")
        log("   T   scored      lead Q5-Q1      atyp Q5-Q1     r(lead, T=50)")
        for T in use:
            d = gate_frame(classes, T)
            if d is None:
                continue
            row = {"lead": contrast(d, "topic_dkl"), "atypicality": contrast(d, "A"),
                   "n_scored": int(d.height)}
            if T == 50:
                ref = d.select("serial_number", pl.col("topic_dkl").alias("r_L"),
                               pl.col("A").alias("r_A"))
            elif ref is not None:
                j = d.select("serial_number", "topic_dkl", "A").join(
                    ref, on="serial_number", how="inner")
                row["vs_T50"] = {
                    "n": int(j.height),
                    "lead_pearson": float(j.select(pl.corr("topic_dkl", "r_L")).item()),
                    "lead_spearman": float(
                        j.select(pl.corr("topic_dkl", "r_L", method="spearman")).item()),
                    "atypicality_pearson": float(j.select(pl.corr("A", "r_A")).item())}
                del j
            out[key][str(T)] = row
            L, A = row["lead"], row["atypicality"]
            r = row.get("vs_T50", {}).get("lead_pearson")
            log(f"  {T:>4} {row['n_scored']:>9,}  "
                f"{100*L['lift']:+6.2f}pp (t{L['t']:+5.1f})  "
                f"{100*A['lift']:+6.2f}pp  "
                f"{('%+.3f' % r) if r is not None else '   ---'}")
            del d
            gc.collect()
    return out


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if mode == "build":
        for T in [int(a) for a in sys.argv[2:]]:
            # Fitting is resumable and may stop mid-way on a wall-clock budget;
            # scoring can only run once the model is complete.
            if not fit_model(T):
                log(f"[build] T={T} fit incomplete, stopping before scoring")
                return 0
            score_class(SWEEP_CLASS, T)
        return 0
    Ts = [int(a) for a in sys.argv[2:]] or [50, 100, 200, 500, 1000, 2500]
    have = [T for T in Ts if score_path(SWEEP_CLASS, T).exists()
            or score_path("001", T).exists()]
    log(f"[compare] resolutions with scores present: {have}")
    out = compare(have)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "topic_resolution_sweep.json").write_text(json.dumps(out, indent=1))
    log("\n[done] topic_resolution_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
