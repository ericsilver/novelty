"""Single-token emergence: do early adopters of a newly-arrived term
survive better than late adopters?

For each NICE class:
  1. Tokenize each filing's goods_services with the existing analyzer.
  2. Build per-(token,year) counts and per-(token,year) sums of passed_5y.
  3. Define "emergence" as the first year a token's doc-frequency in this
     class crosses EMERGE_MIN (default 50), conditional on cumulative
     prior doc-frequency < EMERGE_MIN/2 (so we're catching genuine novelty,
     not noise/spelling variants on a previously-rare term).
  4. Restrict to tokens whose emergence year is in [1995, 2015] so we
     have at least 5y of post-emergence cohorts AND 5y of survival
     observation for the late cohort.
  5. For each emerging token define cohorts relative to emergence year y:
       early  : year in [y, y+1]
       mid    : year in [y+2, y+4]
       late   : year in [y+5, y+9]
     Compute survival rate (passed_5y) per cohort.
  6. Pool by averaging cohort survival rates across all emerging tokens
     (unweighted: each emerging token = one observation).  Also report a
     filing-weighted pool for sensitivity.

Usage:
  python scripts/token_emergence.py                  # smoke test, class 9
  python scripts/token_emergence.py --classes all    # all 45
  python scripts/token_emergence.py --classes 009,035,041,042

Outputs (paper/results/):
  token_emergence_<label>.txt     numerical summary
  token_emergence_<label>.csv     per-token table
  token_emergence_<label>.png     pooled figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from novelty.dictionary import STOPWORDS, _make_analyzer

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

EMERGE_MIN = 50           # token must reach this many filings in a year to count as "emerged"
EMERGE_PRIOR_MAX = 25     # cumulative pre-emergence doc-freq must be < this
MIN_DF = 20               # global doc_freq filter for vocabulary
EMERGE_YEAR_LO = 1995
EMERGE_YEAR_HI = 2015

ANALYZER = _make_analyzer(frozenset(STOPWORDS), (1, 1))  # unigrams only


def load_class(cls: str, year_max: int) -> pl.DataFrame:
    tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                         columns=["serial_number", "filing_date", "goods_services"])
    out = pl.read_parquet(PROC / f"outcomes_class{cls}.parquet",
                          columns=["serial_number", "year", "reached_registration",
                                   "currently_live"]).with_columns(
        (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
    )
    df = tm.join(out, on="serial_number", how="inner").filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("year") >= 1980)
        & (pl.col("year") <= year_max)
    )
    return df


def emergence_table_for_class(cls: str, year_max: int, verbose: bool = True) -> pl.DataFrame:
    """Return one row per emerging token in this class with cohort survival rates."""
    df = load_class(cls, year_max)
    if df.is_empty():
        return pl.DataFrame()
    if verbose:
        print(f"  [{cls}] {df.height:,} filings to tokenize", flush=True)

    docs = [t.lower() if t else "" for t in df["goods_services"].to_list()]
    vec = CountVectorizer(analyzer=ANALYZER, min_df=MIN_DF, max_df=0.5, binary=True)
    dtm = vec.fit_transform(docs)  # (n_docs, n_tokens) sparse 0/1
    dtm = sp.csc_matrix(dtm)
    vocab = vec.get_feature_names_out()
    if verbose:
        print(f"  [{cls}] vocab size {len(vocab):,}", flush=True)

    years = df["year"].to_numpy()
    passed = df["passed_5y"].to_numpy().astype(np.int8)
    uniq_years = np.array(sorted(np.unique(years)))
    yr_index = {y: i for i, y in enumerate(uniq_years)}
    yr_pos = np.array([yr_index[y] for y in years], dtype=np.int32)
    n_years = len(uniq_years)

    # Per-token per-year doc-count and per-token per-year survivor count
    rows = []
    indptr = dtm.indptr
    indices = dtm.indices
    for ti in range(dtm.shape[1]):
        rows_idx = indices[indptr[ti]:indptr[ti + 1]]
        if rows_idx.size < EMERGE_MIN:
            continue
        # year hist for this token
        y_hist = np.bincount(yr_pos[rows_idx], minlength=n_years)
        s_hist = np.bincount(yr_pos[rows_idx], weights=passed[rows_idx],
                             minlength=n_years).astype(np.float64)
        # Find emergence year: first year with count >= EMERGE_MIN
        cand = np.where(y_hist >= EMERGE_MIN)[0]
        if cand.size == 0:
            continue
        ey_pos = cand[0]
        emergence_year = int(uniq_years[ey_pos])
        prior_total = int(y_hist[:ey_pos].sum())
        if prior_total >= EMERGE_PRIOR_MAX:
            continue
        if emergence_year < EMERGE_YEAR_LO or emergence_year > EMERGE_YEAR_HI:
            continue

        def cohort(lo, hi):
            mask = (uniq_years >= emergence_year + lo) & (uniq_years <= emergence_year + hi)
            n = int(y_hist[mask].sum())
            s = float(s_hist[mask].sum())
            rate = (s / n) if n > 0 else float("nan")
            return n, rate
        n_e, r_e = cohort(0, 1)
        n_m, r_m = cohort(2, 4)
        n_l, r_l = cohort(5, 9)
        if n_e < 20 or n_l < 20:
            continue
        rows.append({
            "nice_class": cls,
            "token": vocab[ti],
            "emergence_year": emergence_year,
            "prior_doc_freq": prior_total,
            "n_early": n_e, "surv_early": r_e,
            "n_mid": n_m, "surv_mid": r_m,
            "n_late": n_l, "surv_late": r_l,
            "n_total": int(rows_idx.size),
            "delta_late_minus_early": (r_l - r_e) if (np.isfinite(r_l) and np.isfinite(r_e)) else float("nan"),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="009",
                    help="comma-separated NICE classes, or 'all'")
    ap.add_argument("--year-max", type=int, default=2023)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.classes == "all":
        classes = sorted(p.stem.replace("tm_class", "")
                         for p in PROC.glob("tm_class*.parquet")
                         if p.stem.replace("tm_class", "").isdigit())
    else:
        classes = [c.strip().zfill(3) for c in args.classes.split(",")]
    label = args.label or ("all" if len(classes) > 5 else "_".join(classes))

    print(f"[token-emergence] {len(classes)} class(es); EMERGE_MIN={EMERGE_MIN}, "
          f"PRIOR_MAX={EMERGE_PRIOR_MAX}, year window [{EMERGE_YEAR_LO},{EMERGE_YEAR_HI}]",
          flush=True)

    parts = []
    for cls in classes:
        if not (PROC / f"tm_class{cls}.parquet").exists():
            print(f"  [{cls}] skipping (no tm_class{cls}.parquet)", flush=True)
            continue
        try:
            tbl = emergence_table_for_class(cls, args.year_max)
        except Exception as e:
            print(f"  [{cls}] ERROR: {e}", flush=True)
            continue
        if tbl.is_empty():
            print(f"  [{cls}] no emerging tokens", flush=True)
            continue
        print(f"  [{cls}] {tbl.height} emerging tokens", flush=True)
        parts.append(tbl)

    if not parts:
        print("[done] no emerging tokens found across requested classes", flush=True)
        return 1
    full = pl.concat(parts).sort(["nice_class", "emergence_year", "token"])
    print(f"\n[pooled] {full.height} emerging tokens total", flush=True)

    # ---- pooled results ----
    e_mean = float(full["surv_early"].mean())
    m_mean = float(full["surv_mid"].mean())
    l_mean = float(full["surv_late"].mean())
    delta = float(full["delta_late_minus_early"].mean())
    n_tokens = full.height

    # Filing-weighted pool: sum survivors / sum filings, across tokens
    def fw(num_col, den_col):
        n = full[den_col].to_numpy()
        s = (full[num_col].to_numpy() * n)
        return float(s.sum() / n.sum()) if n.sum() > 0 else float("nan")

    e_fw = fw("surv_early", "n_early")
    m_fw = fw("surv_mid", "n_mid")
    l_fw = fw("surv_late", "n_late")

    print(f"\nUnweighted (1 token = 1 obs, n={n_tokens}):")
    print(f"  early : surv={e_mean:.3f}")
    print(f"  mid   : surv={m_mean:.3f}")
    print(f"  late  : surv={l_mean:.3f}")
    print(f"  late - early = {l_mean - e_mean:+.3f}")
    print(f"\nFiling-weighted (each filing of an emerged token = 1 obs):")
    print(f"  early : surv={e_fw:.3f}  (n_filings={int(full['n_early'].sum()):,})")
    print(f"  mid   : surv={m_fw:.3f}  (n_filings={int(full['n_mid'].sum()):,})")
    print(f"  late  : surv={l_fw:.3f}  (n_filings={int(full['n_late'].sum()):,})")

    # ---- per-token CSV ----
    out_csv = RESULTS / f"token_emergence_{label}.csv"
    full.write_csv(out_csv)
    print(f"\n[done] wrote {out_csv}", flush=True)

    # ---- text summary ----
    out_txt = RESULTS / f"token_emergence_{label}.txt"
    with out_txt.open("w") as f:
        f.write(f"Token emergence: do early adopters survive better than late?\n")
        f.write(f"Classes: {','.join(classes) if len(classes) <= 10 else f'{len(classes)} classes'}\n")
        f.write(f"EMERGE_MIN={EMERGE_MIN}, PRIOR_MAX={EMERGE_PRIOR_MAX}, "
                f"emergence-year window [{EMERGE_YEAR_LO},{EMERGE_YEAR_HI}]\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"N emerging tokens: {n_tokens}\n\n")
        f.write("Pooled (one-token-one-obs):\n")
        f.write(f"  early (y..y+1)   surv = {e_mean:.3f}\n")
        f.write(f"  mid   (y+2..y+4) surv = {m_mean:.3f}\n")
        f.write(f"  late  (y+5..y+9) surv = {l_mean:.3f}\n")
        f.write(f"  late - early     = {l_mean - e_mean:+.3f}\n\n")
        f.write(f"Filing-weighted:\n")
        f.write(f"  early surv = {e_fw:.3f}  (n_filings={int(full['n_early'].sum()):,})\n")
        f.write(f"  mid   surv = {m_fw:.3f}  (n_filings={int(full['n_mid'].sum()):,})\n")
        f.write(f"  late  surv = {l_fw:.3f}  (n_filings={int(full['n_late'].sum()):,})\n\n")
        # Top 30 tokens with biggest late-vs-early gap (favor late)
        f.write("Top 20 tokens where LATE adopters survived MUCH better than early:\n")
        for r in full.sort("delta_late_minus_early", descending=True).head(20).iter_rows(named=True):
            f.write(f"  [{r['nice_class']}] {r['token']:<25s} "
                    f"emerge={r['emergence_year']}  "
                    f"early={r['surv_early']:.2f} (n={r['n_early']})  "
                    f"late={r['surv_late']:.2f} (n={r['n_late']})  "
                    f"Δ={r['delta_late_minus_early']:+.2f}\n")
        f.write("\nTop 20 tokens where EARLY adopters survived MUCH better than late:\n")
        for r in full.sort("delta_late_minus_early").head(20).iter_rows(named=True):
            f.write(f"  [{r['nice_class']}] {r['token']:<25s} "
                    f"emerge={r['emergence_year']}  "
                    f"early={r['surv_early']:.2f} (n={r['n_early']})  "
                    f"late={r['surv_late']:.2f} (n={r['n_late']})  "
                    f"Δ={r['delta_late_minus_early']:+.2f}\n")
    print(f"[done] wrote {out_txt}", flush=True)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cohorts = ["early\n(y, y+1)", "mid\n(y+2, y+4)", "late\n(y+5, y+9)"]
    means_uw = [e_mean, m_mean, l_mean]
    means_fw = [e_fw, m_fw, l_fw]
    ax = axes[0]
    ax.bar(cohorts, means_uw, color=["#2b6cb0", "#5ba0e0", "#a8caea"],
           edgecolor="black", linewidth=0.5)
    for i, v in enumerate(means_uw):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean per-token survival rate")
    ax.set_ylim(0, max(means_uw) * 1.2)
    ax.set_title(f"(a) Pooled across {n_tokens} emerging tokens (unweighted)")
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(cohorts, means_fw, color=["#2b6cb0", "#5ba0e0", "#a8caea"],
            edgecolor="black", linewidth=0.5)
    for i, v in enumerate(means_fw):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Filing-weighted survival rate")
    ax2.set_ylim(0, max(means_fw) * 1.2)
    ax2.set_title(f"(b) Filing-weighted "
                  f"(n_e={int(full['n_early'].sum()):,}, "
                  f"n_l={int(full['n_late'].sum()):,})")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Single-token emergence — early vs late adopter survival "
        f"(EMERGE_MIN={EMERGE_MIN}, prior<{EMERGE_PRIOR_MAX}; emergence years {EMERGE_YEAR_LO}-{EMERGE_YEAR_HI})",
        fontsize=11)
    fig.tight_layout()
    out_png = RESULTS / f"token_emergence_{label}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
