"""Vocabulary forensics: what can the topic model not see, and does the
term-level firm signal live exactly there?

1. Rebuild the topic model's vocabulary deterministically (same stratified
   sample, seed, min_df as topic_p_scorer_all; identical for T=50/T=200).
2. Probe signature terms (blockchain, nft, cannabidiol, ...) for membership
   in the topic vocabulary, the class-009 term vocabulary, and the
   T=50/T=200 topic top-word lists.
3. Class 009: per-filing "invisible share" = share of token mass on terms
   in the term vocabulary but absent from the topic vocabulary. Then the
   discriminating test: the debut EDGAR gradient and the first-gate
   outcome by token-dKL, split by invisible-share halves. If the firm
   signal concentrates where the topic model is blind, the granularity
   reading survives; if not, the artifact reading stands.

Output: paper/results/vocab_forensics.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SAMPLE_PER_CLASS = 10_000
MIN_DF_SAMPLE = 50
YEAR_MIN, YEAR_MAX = 1990, 2024
RNG = np.random.default_rng(42)
PASS = {"701", "702", "703", "704", "705", "800"}
FAIL = {"710"}

PROBES = ["blockchain", "cryptocurrency", "nft", "non-fungible", "nfts",
          "cannabidiol", "cbd", "saas", "metaverse", "chatbot",
          "machine learning", "artificial intelligence", "generative",
          "crypto", "hard seltzer", "vape", "e-liquid", "typesetting",
          "dial-up", "smartwatch", "fintech", "telehealth"]


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
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer
    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))

    # ---- 1. rebuild topic-model vocabulary (deterministic) ----
    sample_docs: list[str] = []
    for cls in all_classes():
        d = load_class_docs(cls)
        take = min(d.height, SAMPLE_PER_CLASS)
        idx = np.sort(RNG.choice(d.height, size=take, replace=False))
        sample_docs.extend(d["goods_services"][idx].to_list())
        del d
        gc.collect()
    vec_topic = CountVectorizer(analyzer=analyzer, min_df=MIN_DF_SAMPLE)
    vec_topic.fit(sample_docs)
    topic_vocab = set(vec_topic.get_feature_names_out())
    del sample_docs, vec_topic
    gc.collect()
    print(f"[topic vocab] {len(topic_vocab):,} terms", file=sys.stderr, flush=True)

    # ---- 2. probes ----
    term_vocab9 = set(pl.read_parquet(PROC / "vocab_class009.parquet").filter(pl.col("doc_freq") >= 50)["token"].to_list())
    tw50 = json.loads((PROC / "topic_lda_meta.json").read_text())["top_words"]
    tw200 = json.loads((PROC / "topic_lda_meta_T200.json").read_text())["top_words"]
    in_top50 = {w for ws in tw50.values() for w in ws}
    in_top200 = {w for ws in tw200.values() for w in ws}
    probes = {}
    for p in PROBES:
        probes[p] = {
            "in_topic_vocab": p in topic_vocab,
            "in_term_vocab_009": p in term_vocab9,
            "in_T50_topwords": p in in_top50,
            "in_T200_topwords": p in in_top200,
        }
        print(f"  {p:>24}: topic_vocab={probes[p]['in_topic_vocab']} "
              f"term009={probes[p]['in_term_vocab_009']} "
              f"top50={probes[p]['in_T50_topwords']} top200={probes[p]['in_T200_topwords']}",
              file=sys.stderr, flush=True)

    # vocabulary overlap for class 009
    invisible9 = term_vocab9 - topic_vocab
    overlap = {
        "term_vocab_009": len(term_vocab9),
        "topic_vocab": len(topic_vocab),
        "invisible_009_terms": len(invisible9),
        "invisible_009_share_of_terms": len(invisible9) / len(term_vocab9),
        "invisible_unigram_examples": sorted(
            [t for t in invisible9 if " " not in t])[:40],
    }
    print(f"[overlap] class-009 term vocab {len(term_vocab9):,}; invisible to "
          f"topic model: {len(invisible9):,} "
          f"({100*overlap['invisible_009_share_of_terms']:.1f}%)",
          file=sys.stderr, flush=True)

    # ---- 3. invisible share per filing, class 009, + outcome tests ----
    terms9 = sorted(term_vocab9)
    vec9 = CountVectorizer(analyzer=analyzer, vocabulary=terms9)
    d9 = load_class_docs("009")
    X = vec9.transform(d9["goods_services"].to_list()).tocsr()
    inv_mask = np.array([t in invisible9 for t in terms9])
    rowsum = np.asarray(X.sum(axis=1)).ravel()
    inv_mass = np.asarray(X[:, inv_mask].sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_share = np.where(rowsum > 0, inv_mass / np.maximum(rowsum, 1), np.nan)
    print(f"[invshare] mean {np.nanmean(inv_share):.4f}, "
          f"p90 {np.nanpercentile(inv_share, 90):.4f}", file=sys.stderr, flush=True)
    del X
    gc.collect()

    inv_df = pl.DataFrame({"serial_number": d9["serial_number"],
                           "inv_share": inv_share})
    tok = pl.read_parquet(
        PROC / "surprise_class009.parquet",
        columns=["serial_number", "year", "prospective_kl", "retrospective_kl",
                 "n_ref_prospective", "n_ref_retrospective", "n_terms"],
    ).filter(
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
    ).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"))
    tm = pl.read_parquet(
        PROC / "tm_class009.parquet",
        columns=["serial_number", "owner_name", "filing_date",
                 "registration_date", "status_code"],
    ).with_columns(
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8
         ).alias("registered"),
        pl.col("registration_date").str.slice(0, 4)
          .cast(pl.Int32, strict=False).alias("reg_year"),
        pl.col("status_code").is_in(list(PASS)).alias("st_pass"),
        pl.col("status_code").is_in(list(FAIL)).alias("st_fail"),
    )
    j = tok.join(inv_df, on="serial_number", how="inner").join(
        tm, on="serial_number", how="inner").filter(
        pl.col("inv_share").is_not_nan())

    # owner-ever-EDGAR
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))
    j = j.join(sec, on="owner_name", how="left").with_columns(
        pl.col("in_sec").fill_null(False))

    def quintiles(frame, var, outcome):
        arr = frame[var].to_numpy()
        cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(cuts, arr)
        out = {}
        y = frame[outcome].cast(pl.Float64).to_numpy()
        for q in range(5):
            m = qi == q
            out[f"q{q+1}"] = float(y[m].mean())
            out[f"q{q+1}_n"] = int(m.sum())
        out["lift_q5_q1"] = out["q5"] - out["q1"]
        return out

    results_tests = {}
    # correlation of inv_share with dkl and year
    samp = j.select("inv_share", "dkl", "year").to_pandas()
    results_tests["corr_invshare_dkl"] = float(samp.inv_share.corr(samp.dkl))
    results_tests["corr_invshare_year"] = float(samp.inv_share.corr(samp.year))

    # gate test by invisible-share halves
    gate = j.filter(pl.col("reg_year").is_between(2016, 2018)
                    & (pl.col("st_pass") | pl.col("st_fail")))
    med = float(gate["inv_share"].median())
    results_tests["gate_low_inv"] = quintiles(
        gate.filter(pl.col("inv_share") <= med), "dkl", "st_pass")
    results_tests["gate_high_inv"] = quintiles(
        gate.filter(pl.col("inv_share") > med), "dkl", "st_pass")

    # EDGAR test among registered filings 1995-2018, by invisible halves
    reg = j.filter(pl.col("registered") & pl.col("year").is_between(1995, 2018))
    medr = float(reg["inv_share"].median())
    results_tests["edgar_low_inv"] = quintiles(
        reg.filter(pl.col("inv_share") <= medr), "dkl", "in_sec")
    results_tests["edgar_high_inv"] = quintiles(
        reg.filter(pl.col("inv_share") > medr), "dkl", "in_sec")
    # and the direct outcome gradient on invisible share itself
    results_tests["edgar_by_invshare"] = quintiles(reg, "inv_share", "in_sec")
    results_tests["gate_by_invshare"] = quintiles(gate, "inv_share", "st_pass")

    for k, v in results_tests.items():
        print(f"  {k}: {v if not isinstance(v, dict) else {kk: round(vv,5) for kk, vv in v.items() if not kk.endswith('_n')}}",
              file=sys.stderr, flush=True)

    out = {"probes": probes, "overlap": overlap, "tests": results_tests,
           "gate_median_invshare": med, "edgar_median_invshare": medr}
    (RES / "vocab_forensics.json").write_text(json.dumps(out, indent=1, default=float))
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
