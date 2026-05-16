"""Test whether the within-firm $\\Delta KL$ persistence is genuine or
single-phrase.

Hypothesis under test: a firm with one novel commitment (e.g., a
blockchain consultancy that puts ``blockchain'' in every filing's G/S)
will look persistently innovative under our pooled measure even if every
non-debut filing simply recycles the same novel vocabulary.  The
alternative is that high-$\\Delta KL$ firms genuinely introduce new novel
tokens per filing (e.g., a software firm that over time adds vocabulary
for cloud, AI, IoT, etc., as separate novel commitments).

For each firm with $\\geq$ 3 clean filings in a class:
  1. Tokenize each filing's goods_services with the same analyzer the KL
     pipeline uses (unigrams only, post-stopword).
  2. Walk filings chronologically.  For each non-debut filing, partition
     its unique tokens into ``recurring'' (appeared in any prior firm
     filing) and ``new-to-firm''.
  3. Aggregate to per-firm:
       - mean_share_new_tokens = unweighted mean across non-debut
         filings of (n_new_tokens / n_unique_tokens_in_filing)
       - mean_consecutive_jaccard = mean Jaccard overlap between
         each filing and the union of prior filings' tokens
  4. Bucket firms into $\\Delta KL$ quartiles using firm-mean $\\Delta KL$
     across all clean filings, and compare:
       - mean share-new-tokens by quartile
       - mean within-firm Jaccard by quartile

Interpretation:
  * If high-$\\Delta KL$ firms have HIGH within-firm Jaccard (and LOW
    share-new) -> persistence is recycling-driven (the ``blockchain co.''
    pattern).
  * If high-$\\Delta KL$ firms have LOW within-firm Jaccard (and HIGH
    share-new) -> persistence is genuine; each filing is a fresh bet.

Outputs (paper/results/):
  firm_vocab_recurrence_<label>.png
  firm_vocab_recurrence_<label>.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

sys.path.insert(0, str(REPO_ROOT / "src"))
from novelty.dictionary import STOPWORDS, _make_analyzer

CLEAN_BASE = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
    & pl.col("prospective_kl").is_finite()
    & pl.col("retrospective_kl").is_finite()
    & pl.col("owner_name").is_not_null()
)

ANALYZER = _make_analyzer(frozenset(STOPWORDS), (1, 1))
MIN_DF = 20
MIN_FIRM_FILINGS = 3


def load_class(cls: str, year_min: int, year_max: int) -> pl.DataFrame:
    tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                         columns=["serial_number", "filing_date", "goods_services"])
    out = pl.read_parquet(PROC / f"outcomes_class{cls}.parquet")
    out = out.filter(CLEAN_BASE & (pl.col("year") >= year_min) & (pl.col("year") <= year_max))
    out = out.with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl")
    ).select("serial_number", "owner_name", "year", "prospective_kl",
             "retrospective_kl", "dkl")
    df = tm.join(out, on="serial_number", how="inner").filter(
        pl.col("goods_services").is_not_null()
    )
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="009")
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2023)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    classes = [c.strip().zfill(3) for c in args.classes.split(",")]
    label = args.label or "_".join(classes)

    parts = []
    for cls in classes:
        if not (PROC / f"tm_class{cls}.parquet").exists():
            print(f"  [{cls}] missing, skip", flush=True); continue
        d = load_class(cls, args.year_min, args.year_max)
        d = d.with_columns(pl.lit(cls).alias("nice_class"))
        parts.append(d)
    df = pl.concat(parts)
    print(f"[load] {df.height:,} clean filings", flush=True)

    # Restrict to firms with enough filings to study persistence
    df = df.with_columns(pl.len().over("owner_name").alias("n_firm_filings"))
    df = df.filter(pl.col("n_firm_filings") >= MIN_FIRM_FILINGS)
    n_firms = df["owner_name"].n_unique()
    print(f"[filter] {df.height:,} filings from {n_firms:,} firms with "
          f">={MIN_FIRM_FILINGS} clean filings", flush=True)

    # Tokenize all filings once
    print("[tok] tokenizing all filings…", flush=True)
    docs = [t.lower() if t else "" for t in df["goods_services"].to_list()]
    vec = CountVectorizer(analyzer=ANALYZER, min_df=MIN_DF, max_df=0.5, binary=True)
    dtm = vec.fit_transform(docs)  # (n_filings, n_tokens) sparse 0/1
    vocab = vec.get_feature_names_out()
    print(f"  vocab size {len(vocab):,}; dtm nnz {dtm.nnz:,}", flush=True)

    # Convert to per-filing token-set list
    print("[tok] materializing per-filing token sets…", flush=True)
    dtm_csr = dtm.tocsr()
    indptr = dtm_csr.indptr
    indices = dtm_csr.indices
    # token sets as numpy arrays of token indices

    # Sort df by (owner, filing_date) so we walk chronologically
    df_pd = df.to_pandas().reset_index(drop=True)
    df_pd["filing_date"] = df_pd["filing_date"].fillna("99999999")
    df_pd = df_pd.sort_values(["owner_name", "filing_date"]).reset_index(drop=False)
    # 'index' column now holds the original row id (matches dtm row)
    orig_idx = df_pd["index"].to_numpy()

    # For each firm, walk filings, accumulate seen-token set, compute per-filing
    # share_new = |new_tokens| / |unique_tokens_in_filing|
    # jaccard_with_prior = |intersect(filing, prior_union)| / |union(filing, prior_union)|
    print("[walk] walking firms chronologically…", flush=True)

    owners = df_pd["owner_name"].to_numpy()
    n_rows = len(df_pd)

    share_new_per_filing = np.full(n_rows, np.nan)
    jaccard_with_prior = np.full(n_rows, np.nan)
    is_debut = np.zeros(n_rows, dtype=bool)
    n_unique_in_filing = np.zeros(n_rows, dtype=np.int32)

    cur_owner = None
    seen = set()
    for i in range(n_rows):
        ow = owners[i]
        if ow != cur_owner:
            cur_owner = ow
            seen = set()
            is_debut[i] = True
        row_idx = orig_idx[i]
        toks = set(indices[indptr[row_idx]:indptr[row_idx + 1]].tolist())
        n_unique_in_filing[i] = len(toks)
        if not is_debut[i] and toks:
            new_toks = toks - seen
            share_new_per_filing[i] = len(new_toks) / len(toks)
            inter = len(toks & seen)
            uni = len(toks | seen)
            jaccard_with_prior[i] = inter / uni if uni > 0 else 0.0
        seen |= toks

    df_pd["share_new"] = share_new_per_filing
    df_pd["jaccard_prior"] = jaccard_with_prior
    df_pd["is_debut"] = is_debut
    df_pd["n_unique_tok"] = n_unique_in_filing

    # Aggregate to firm
    nondebut = df_pd[~df_pd["is_debut"]].copy()
    firm = nondebut.groupby("owner_name").agg(
        mean_share_new=("share_new", "mean"),
        mean_jaccard_prior=("jaccard_prior", "mean"),
        n_nondebut=("share_new", "size"),
    ).reset_index()
    firm_dkl = df_pd.groupby("owner_name").agg(
        firm_mean_dkl=("dkl", "mean"),
        firm_mean_pros=("prospective_kl", "mean"),
        firm_mean_retr=("retrospective_kl", "mean"),
        n_firm_filings=("dkl", "size"),
    ).reset_index()
    firm = firm.merge(firm_dkl, on="owner_name", how="inner")
    print(f"[firm] {len(firm):,} firms with >=2 non-debut filings + dkl info", flush=True)

    # Bucket by firm-mean ΔKL quartile
    firm["dkl_quartile"] = (
        firm["firm_mean_dkl"].rank(method="first", pct=True).mul(4).clip(upper=3.999).astype(int)
    )

    print("\n[main] within-firm vocabulary recurrence by ΔKL quartile:")
    print(f"  {'quartile':<10s}{'n_firms':>10s}{'mean_dkl':>12s}"
          f"{'share_new':>12s}{'jaccard_prior':>16s}")
    rows_q = []
    for q in range(4):
        sub = firm[firm["dkl_quartile"] == q]
        n = len(sub)
        if n == 0: continue
        mn = float(sub["firm_mean_dkl"].mean())
        sn = float(sub["mean_share_new"].mean())
        jp = float(sub["mean_jaccard_prior"].mean())
        rows_q.append({"quartile": q + 1, "n_firms": n, "mean_dkl": mn,
                       "share_new": sn, "jaccard_prior": jp})
        print(f"  Q{q+1:<9d}{n:>10,}{mn:>+12.4f}{sn:>12.3f}{jp:>16.3f}")

    # Also: of the firm's TOTAL prospective-KL "mass" (sum of per-filing pros_kl
    # across non-debut filings), what fraction came from filings that
    # introduced mostly NEW tokens (share_new >= 0.5) vs MOSTLY RECURRING
    # (share_new < 0.5)?
    print("\n[main] for each firm-quartile, share of non-debut filings whose "
          "'share_new >= 0.5' (i.e., majority of tokens are new-to-firm):")
    for q in range(4):
        ow = firm[firm["dkl_quartile"] == q]["owner_name"]
        sub = nondebut[nondebut["owner_name"].isin(ow)]
        if sub.empty: continue
        share = float((sub["share_new"] >= 0.5).mean())
        print(f"  Q{q+1}: share-of-nondebut-filings with share_new>=0.5 = {share:.3f}  "
              f"(n filings = {len(sub):,})")

    # Top 10 high-DKL firms with HIGHEST jaccard (most recycling)
    top_dkl = firm.sort_values("firm_mean_dkl", ascending=False).head(int(len(firm) * 0.1))
    print(f"\n[diag] within top-decile of firm-mean ΔKL "
          f"(n={len(top_dkl):,} firms):")
    print(f"  median jaccard with prior tokens: {float(top_dkl['mean_jaccard_prior'].median()):.3f}")
    print(f"  median share-new tokens:         {float(top_dkl['mean_share_new'].median()):.3f}")
    bot_dkl = firm.sort_values("firm_mean_dkl").head(int(len(firm) * 0.1))
    print(f"\n[diag] within bottom-decile of firm-mean ΔKL "
          f"(n={len(bot_dkl):,} firms):")
    print(f"  median jaccard with prior tokens: {float(bot_dkl['mean_jaccard_prior'].median()):.3f}")
    print(f"  median share-new tokens:         {float(bot_dkl['mean_share_new'].median()):.3f}")

    # Sanity examples: show 5 firms in top-decile w/ highest jaccard
    examples = top_dkl.sort_values("mean_jaccard_prior", ascending=False).head(8)
    print("\n[examples] top-decile-ΔKL firms with HIGHEST recycling (jaccard):")
    for _, r in examples.iterrows():
        print(f"  {r['owner_name'][:50]:<50s}  dkl={r['firm_mean_dkl']:+.3f}  "
              f"jaccard={r['mean_jaccard_prior']:.3f}  share_new={r['mean_share_new']:.3f}  "
              f"n_filings={int(r['n_firm_filings'])}")
    examples2 = top_dkl.sort_values("mean_jaccard_prior").head(8)
    print("\n[examples] top-decile-ΔKL firms with LOWEST recycling "
          "(genuinely diverse novel filings):")
    for _, r in examples2.iterrows():
        print(f"  {r['owner_name'][:50]:<50s}  dkl={r['firm_mean_dkl']:+.3f}  "
              f"jaccard={r['mean_jaccard_prior']:.3f}  share_new={r['mean_share_new']:.3f}  "
              f"n_filings={int(r['n_firm_filings'])}")

    # ---- text summary ----
    out_txt = RESULTS / f"firm_vocab_recurrence_{label}.txt"
    with out_txt.open("w") as f:
        f.write("Within-firm vocabulary recurrence vs ΔKL persistence\n")
        f.write(f"Classes: {','.join(classes)}; window {args.year_min}-{args.year_max}\n")
        f.write(f"Min firm filings: {MIN_FIRM_FILINGS}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"N firms (with >=2 non-debut filings + dkl): {len(firm):,}\n\n")
        f.write("By firm-mean ΔKL quartile:\n")
        f.write(f"  {'quartile':<10s}{'n_firms':>10s}{'mean_dkl':>12s}"
                f"{'share_new':>12s}{'jaccard':>10s}\n")
        for r in rows_q:
            f.write(f"  Q{r['quartile']:<9d}{r['n_firms']:>10,}{r['mean_dkl']:>+12.4f}"
                    f"{r['share_new']:>12.3f}{r['jaccard_prior']:>10.3f}\n")
        f.write("\nInterpretation:\n")
        f.write("  share_new = mean (across firm's non-debut filings) of\n")
        f.write("              (tokens new-to-firm in this filing) / (unique tokens in this filing)\n")
        f.write("  jaccard   = mean (across firm's non-debut filings) of Jaccard(this, prior_union)\n")
        f.write("  If high-ΔKL firms have HIGH jaccard (LOW share_new), persistent novelty\n")
        f.write("  is recycling-driven; if LOW jaccard (HIGH share_new), each filing is\n")
        f.write("  a fresh novel commitment.\n")
    print(f"\n[done] wrote {out_txt}", flush=True)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    qs = [r["quartile"] for r in rows_q]
    sn = [r["share_new"] for r in rows_q]
    jp = [r["jaccard_prior"] for r in rows_q]
    colors = ["#cc4444", "#dd9966", "#66aacc", "#2b6cb0"]
    ax = axes[0]
    ax.bar(qs, sn, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(sn):
        ax.text(qs[i], v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(qs)
    ax.set_xticklabels([f"Q{q}\n(tired)" if q == 1 else f"Q{q}\n(innovative)" if q == 4 else f"Q{q}"
                       for q in qs])
    ax.set_xlabel(r"Firm-mean $\Delta KL$ quartile")
    ax.set_ylabel("Mean per-filing share of new-to-firm tokens")
    ax.set_title("(a) Share of non-debut filing's tokens that are new to the firm")
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(qs, jp, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(jp):
        ax2.text(qs[i], v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(qs)
    ax2.set_xticklabels([f"Q{q}\n(tired)" if q == 1 else f"Q{q}\n(innovative)" if q == 4 else f"Q{q}"
                        for q in qs])
    ax2.set_xlabel(r"Firm-mean $\Delta KL$ quartile")
    ax2.set_ylabel("Mean Jaccard(non-debut filing, prior tokens)")
    ax2.set_title("(b) Token recycling vs prior firm-vocabulary")
    ax2.grid(True, axis="y", alpha=0.3)

    cls_label = ",".join(classes) if len(classes) <= 5 else f"{len(classes)} classes"
    fig.suptitle(f"Within-firm vocabulary recurrence by $\\Delta KL$ quartile "
                 f"({len(firm):,} firms, classes {cls_label}, "
                 f"$\\geq${MIN_FIRM_FILINGS} clean filings)", fontsize=11)
    fig.tight_layout()
    out_png = RESULTS / f"firm_vocab_recurrence_{label}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
