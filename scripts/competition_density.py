"""Live-competitor density mediation test.

For each focal filing F in Class 9 (Software & Electronics), find
near-competitors j where:
  - j is in the same NICE class
  - j's filing_date is in [t_F - 2y, t_F + 5y]   (pre + during survival window)
  - cosine(tfidf(g/s_F), tfidf(g/s_j)) >= 0.7
and compute
  total_competitors = #{j satisfying the above}
  live_competitors  = #{j satisfying the above AND survived_5y = True}

Then test:
  (a) is live_competitors U-shaped or monotone in |ΔKL|?
  (b) does live_competitors mediate the ΔKL -> survival U-shape?
      Fit logit  survived_5y ~ z(dKL) + z(dKL)^2 + log(1+live_comp)
                              + log(n_terms) + year_c + year_c^2
      Compare quadratic-on-dKL with and without log(live_comp).

Stratified sample of ~10k focal filings across dKL deciles to keep
the all-pairs cosine tractable.

Outputs:
  paper/results/competition_density.csv          (per-focal metrics)
  paper/results/competition_density_bins.csv     (per-dKL-decile summary)
  paper/results/competition_density.png          (two-panel plot)
  paper/results/competition_density.tex          (paper-ready table)
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import statsmodels.api as sm
import matplotlib.pyplot as plt

from novelty.dictionary import STOPWORDS, _make_analyzer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

CLASS_ID = "009"
WINDOW_PRE_YEARS  = 2
WINDOW_POST_YEARS = 5
SIM_THRESHOLD = 0.7
N_FOCAL_PER_DECILE = 1000   # 10k focal filings total
RNG_SEED = 42


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[load] Class {CLASS_ID} surprise + outcomes + g/s …", flush=True)
    sp_df = pl.read_parquet(
        PROC / f"surprise_class{CLASS_ID}.parquet",
        columns=["serial_number","year","kl_vs_past","kl_vs_future",
                 "n_ref_past","n_ref_future","n_terms"],
    ).filter(
        (pl.col("n_ref_past") >= 1000)
        & (pl.col("n_ref_future") >= 1000)
        & (pl.col("n_terms") >= 3)
        & pl.col("kl_vs_past").is_finite()
        & pl.col("kl_vs_future").is_finite()
        & pl.col("year").is_between(1990, 2019)   # need 5y post-window observable
    ).with_columns(
        (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
    )
    op = pl.read_parquet(
        PROC / f"outcomes_class{CLASS_ID}.parquet",
        columns=["serial_number","survived_5y","reached_registration"],
    )
    tm = pl.read_parquet(
        PROC / f"tm_class{CLASS_ID}.parquet",
        columns=["serial_number","goods_services"],
    ).filter(pl.col("goods_services").is_not_null())

    df = sp_df.join(op, on="serial_number", how="inner").join(tm, on="serial_number", how="inner")
    n = df.height
    print(f"       n = {n:,}  (panel for cosine search)", flush=True)

    print("[vec] TF-IDF on goods/services …", flush=True)
    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    vec = TfidfVectorizer(analyzer=analyzer, min_df=50, max_df=0.5, sublinear_tf=True)
    X = vec.fit_transform(df["goods_services"].to_list())
    X = normalize(X, axis=1, norm="l2")   # so cosine = dot product
    print(f"       X shape={X.shape}  nnz={X.nnz:,}", flush=True)

    years = df["year"].to_numpy()
    survived = df["survived_5y"].to_numpy().astype(bool)
    dkl = df["dkl"].to_numpy()
    nt  = df["n_terms"].to_numpy()

    # Stratified sample of focal filings across dKL deciles
    rng = np.random.default_rng(RNG_SEED)
    deciles = np.quantile(dkl, np.linspace(0, 1, 11))
    bin_idx = np.clip(np.searchsorted(deciles[1:-1], dkl), 0, 9)
    focal_idx = []
    for b in range(10):
        candidates = np.where(bin_idx == b)[0]
        if len(candidates) <= N_FOCAL_PER_DECILE:
            focal_idx.extend(candidates.tolist())
        else:
            picked = rng.choice(candidates, N_FOCAL_PER_DECILE, replace=False)
            focal_idx.extend(picked.tolist())
    focal_idx = np.array(focal_idx)
    rng.shuffle(focal_idx)
    print(f"[focal] {len(focal_idx):,} stratified focal filings", flush=True)

    # Pre-index: for each year, the row indices of filings in that year.
    yr_min, yr_max = int(years.min()), int(years.max())
    year_to_rows = {y: np.where(years == y)[0] for y in range(yr_min, yr_max + 1)}

    total_competitors = np.zeros(len(focal_idx), dtype=np.int32)
    live_competitors  = np.zeros(len(focal_idx), dtype=np.int32)

    print("[scan] cosine search per focal filing …", flush=True)
    for k, i in enumerate(focal_idx):
        if k % 500 == 0:
            print(f"  {k:>5d} / {len(focal_idx):,}", flush=True)
        y = int(years[i])
        # window of candidate rows
        cand_rows = np.concatenate([
            year_to_rows[yy] for yy in range(y - WINDOW_PRE_YEARS, y + WINDOW_POST_YEARS + 1)
            if yy in year_to_rows
        ])
        if cand_rows.size <= 1:
            continue
        # similarity = X[i] dot X[cand_rows].T  (both L2-normalized)
        v = X.getrow(i)
        sims = X[cand_rows] @ v.T            # sparse * sparse = sparse
        sims = np.asarray(sims.todense()).ravel()
        # exclude self
        self_mask = (cand_rows == i)
        sims[self_mask] = 0.0
        neighbors_mask = sims >= SIM_THRESHOLD
        total_competitors[k] = int(neighbors_mask.sum())
        live_competitors[k]  = int(neighbors_mask[neighbors_mask] @
                                    survived[cand_rows[neighbors_mask]].astype(np.int32))

    # Assemble focal-level frame
    foc = pl.DataFrame({
        "row_idx": focal_idx,
        "year": years[focal_idx],
        "n_terms": nt[focal_idx],
        "dkl": dkl[focal_idx],
        "survived_5y": survived[focal_idx],
        "total_competitors": total_competitors,
        "live_competitors": live_competitors,
    }).with_columns(
        (pl.col("live_competitors") / pl.col("total_competitors").clip(1))
        .alias("share_neighbors_alive"),
    )
    foc.write_csv(OUT / "competition_density.csv")
    print(f"\n[done] wrote per-focal csv  ({foc.height:,} rows)", flush=True)

    # Bin by dKL decile
    arr = foc["dkl"].to_numpy()
    qs = np.linspace(0, 1, 11)
    cuts = np.quantile(arr, qs)
    bin_idx = np.clip(np.searchsorted(cuts[1:-1], arr), 0, 9)
    foc = foc.with_columns(pl.Series("bin", bin_idx))
    bins = (foc.group_by("bin").agg([
        pl.len().alias("n"),
        pl.col("dkl").mean().alias("dkl_mid"),
        pl.col("live_competitors").mean().alias("mean_live"),
        pl.col("total_competitors").mean().alias("mean_total"),
        pl.col("share_neighbors_alive").mean().alias("share_alive"),
        pl.col("survived_5y").cast(pl.Float64).mean().alias("survival_rate"),
    ]).sort("bin"))
    bins.write_csv(OUT / "competition_density_bins.csv")
    print(bins.to_pandas().to_string(index=False, formatters={
        'mean_live': '{:.1f}'.format, 'mean_total': '{:.1f}'.format,
        'share_alive': '{:.3f}'.format, 'survival_rate': '{:.3f}'.format,
        'dkl_mid': '{:+.3f}'.format,
    }))

    # Mediation regression
    pdf = foc.to_pandas()
    z = (pdf["dkl"] - pdf["dkl"].mean()) / pdf["dkl"].std()
    z2 = z * z
    yr_c = pdf["year"] - pdf["year"].mean()
    log_n = np.log(pdf["n_terms"].clip(lower=1))
    log_live = np.log1p(pdf["live_competitors"])
    y = pdf["survived_5y"].astype(int).to_numpy()

    X1 = sm.add_constant(np.column_stack([z, z2, log_n, yr_c, yr_c**2]))
    X2 = sm.add_constant(np.column_stack([z, z2, log_live, log_n, yr_c, yr_c**2]))
    m1 = sm.Logit(y, X1).fit(disp=False, maxiter=80)
    m2 = sm.Logit(y, X2).fit(disp=False, maxiter=80)
    print()
    print("Mediation results:")
    print(f"  Model 1: survived_5y ~ z(dKL) + z(dKL)^2 + log(n_terms) + year + year^2")
    print(f"    beta_dKL          = {m1.params[1]:+.4f}  (SE {m1.bse[1]:.4f}, z {m1.params[1]/m1.bse[1]:+.2f})")
    print(f"    beta_dKL^2        = {m1.params[2]:+.4f}  (SE {m1.bse[2]:.4f}, z {m1.params[2]/m1.bse[2]:+.2f})")
    print(f"  Model 2: + log(1 + live_competitors)")
    print(f"    beta_dKL          = {m2.params[1]:+.4f}  (SE {m2.bse[1]:.4f}, z {m2.params[1]/m2.bse[1]:+.2f})")
    print(f"    beta_dKL^2        = {m2.params[2]:+.4f}  (SE {m2.bse[2]:.4f}, z {m2.params[2]/m2.bse[2]:+.2f})")
    print(f"    beta_log_live     = {m2.params[3]:+.4f}  (SE {m2.bse[3]:.4f}, z {m2.params[3]/m2.bse[3]:+.2f})")
    delta_q = m1.params[2] - m2.params[2]
    print(f"  Attenuation of quadratic: {delta_q:+.4f}  "
          f"({100*delta_q/m1.params[2]:+.1f}% of M1 quadratic)")

    # LaTeX table
    tex = [r"\begin{tabular}{lrr}", r"\toprule",
           r"Coefficient & Model 1 & Model 2 \\",
           r" & (no density control) & (+ $\log(1+\text{live competitors})$) \\",
           r"\midrule"]
    rows_tex = [
        (r"$z(\Delta KL)$ (linear)", m1.params[1], m1.bse[1], m2.params[1], m2.bse[1]),
        (r"$z(\Delta KL)^2$ (quadratic)", m1.params[2], m1.bse[2], m2.params[2], m2.bse[2]),
    ]
    for label, b1, s1, b2, s2 in rows_tex:
        tex.append(f"{label} & {b1:+.4f} ({s1:.4f}) & {b2:+.4f} ({s2:.4f}) \\\\")
    tex.append(r"$\log(1+\text{live competitors})$ & --- & "
               f"{m2.params[3]:+.4f} ({m2.bse[3]:.4f}) \\\\")
    tex.append(r"\midrule")
    tex.append(f"$n$ focal filings (Class 9) & {len(focal_idx):,} & {len(focal_idx):,} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "competition_density.tex").write_text("\n".join(tex))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    pdf_bins = bins.to_pandas()
    ax = axes[0]
    ax.plot(pdf_bins["dkl_mid"], pdf_bins["mean_live"], "-o", color="#cc4444",
            lw=2, ms=6, label="mean live competitors")
    ax.plot(pdf_bins["dkl_mid"], pdf_bins["mean_total"], "-s", color="#888",
            lw=1.5, ms=5, label="mean total near-competitors")
    ax.set_xlabel(r"$\Delta KL$ decile midpoint")
    ax.set_ylabel("mean count of similar filings")
    ax.set_title(r"Near-competitor density by $\Delta KL$ decile" +
                 "\n(cosine $\\geq 0.7$ on g/s TF-IDF; window $[t-2y, t+5y]$)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(pdf_bins["dkl_mid"], pdf_bins["survival_rate"], "-o", color="#2b6cb0",
            lw=2, ms=6, label="focal 5y survival rate")
    ax.plot(pdf_bins["dkl_mid"], pdf_bins["share_alive"], "-^", color="#229922",
            lw=1.5, ms=5, label="share of neighbors alive at 5y")
    ax.set_xlabel(r"$\Delta KL$ decile midpoint")
    ax.set_ylabel("rate")
    ax.set_title(r"Focal survival and neighbor-survival rate by $\Delta KL$")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "competition_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote plots + bins + tex", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
