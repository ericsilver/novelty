"""Refit the topic model at an unchanged resolution under a different seed.

The resolution sweep shows scores at different T agreeing only moderately
(r about 0.77 on lead), and no better for close resolutions than for distant
ones. That pattern has two very different readings. It may mean the number of
themes genuinely changes what is measured. Or it may mean an LDA fit is simply
unstable -- that two runs land in different local optima -- in which case the
sweep is measuring fitting noise and resolution has nothing to do with it.

The two are separated by holding T fixed and moving only the seed. If a seed-7
fit at T=200 disagrees with the seed-42 fit at T=200 about as much as the
seed-42 fits at T=200 and T=1000 disagree with each other, the sweep's spread is
fitting noise. If the same-T replicate agrees far more closely, the spread is
resolution.

The design matrix and vocabulary are the ones cached by the sweep, so the
vocabulary, the sample and the number of passes are identical and the seed is
the only thing that moves.

Usage:  python scripts/topic_seed_replicate.py [T] [seed]
Output: data/processed/topic_model_T{T}_seed{seed}.joblib
        data/processed/rolling_surprise_class009_T{T}_seed{seed}.parquet
        paper/results/topic_seed_replicate.json
"""
from __future__ import annotations

import importlib.util
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

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLS = "009"

spec = importlib.util.spec_from_file_location(
    "sweep", REPO / "scripts" / "topic_resolution_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def log(m: str) -> None:
    print(m, flush=True, file=sys.stderr)


def fit(T: int, seed: int) -> Path:
    out = PROC / f"topic_model_T{T}_seed{seed}.joblib"
    if out.exists():
        log(f"[fit] {out.name} exists, skipping")
        return out
    X = sp.load_npz(PROC / "_lda_fit_matrix.npz")
    vocab = joblib.load(PROC / "_lda_fit_vocab.joblib")
    log(f"[fit] T={T} seed={seed} on {X.shape[0]:,} x {X.shape[1]:,}")

    ck = PROC / f"_lda_ckpt_T{T}_seed{seed}.joblib"
    N_BLOCKS = 8
    bounds = np.linspace(0, X.shape[0], N_BLOCKS + 1).astype(int)
    if ck.exists():
        st = joblib.load(ck)
        lda, ep, blk = st["lda"], st["epochs"], st["block"]
        log(f"[fit] resuming at epoch {ep}/8 block {blk}/{N_BLOCKS}")
    else:
        lda = LatentDirichletAllocation(
            n_components=T, max_iter=1, learning_method="online",
            batch_size=2048, random_state=seed, n_jobs=1, evaluate_every=0)
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
            raise SystemExit(2)
        if blk == 0:
            log(f"[fit] epoch {ep}/8 ({el:.1f} min)")
    joblib.dump({"vocabulary": vocab, "lda": lda}, out, compress=3)
    ck.unlink(missing_ok=True)
    log(f"[fit] complete -> {out.name}")
    return out


def main() -> int:
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    fit(T, seed)

    # Score class 009 with the sweep's own scorer, so the replicate and the
    # original differ in the seed and in nothing else.
    base_model, base_score = sweep.model_path, sweep.score_path
    sweep.model_path = lambda t: PROC / f"topic_model_T{t}_seed{seed}.joblib"
    sweep.score_path = lambda c, t: PROC / f"rolling_surprise_class{c}_T{t}_seed{seed}.parquet"
    sweep.score_class(CLS, T)
    rep = sweep.score_path(CLS, T)
    sweep.model_path, sweep.score_path = base_model, base_score

    def load(p: Path, tag: str) -> pl.DataFrame:
        return (pl.read_parquet(p, columns=["serial_number", "topic_kl_vs_past",
                                            "topic_kl_vs_future"])
                .drop_nulls()
                .filter(pl.col("topic_kl_vs_past").is_finite()
                        & pl.col("topic_kl_vs_future").is_finite())
                .with_columns(((pl.col("topic_kl_vs_past")
                                + pl.col("topic_kl_vs_future")) / 2).alias("A" + tag),
                              (pl.col("topic_kl_vs_past")
                               - pl.col("topic_kl_vs_future")).alias("L" + tag))
                .select(["serial_number", "A" + tag, "L" + tag]))

    m = load(base_score(CLS, T), "a").join(load(rep, "b"), on="serial_number")
    out = {"T": T, "seed_a": 42, "seed_b": seed, "n": m.height,
           "r_lead": m.select(pl.corr("La", "Lb")).item(),
           "r_atyp": m.select(pl.corr("Aa", "Ab")).item()}
    log(f"\n=== same resolution (T={T}), seed 42 vs seed {seed}, "
        f"n={out['n']:,} ===")
    log(f"  r(lead)        {out['r_lead']:.3f}")
    log(f"  r(atypicality) {out['r_atyp']:.3f}")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "topic_seed_replicate.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
