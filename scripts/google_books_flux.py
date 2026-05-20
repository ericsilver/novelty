"""External-corpus confound test: is the trademark-derived flux signal
tracking general English-language usage trends, or does the
trademark corpus show flux even after controlling for general-corpus
usage?

Pull Google Books n-gram frequencies (1985-2021, English) for a
curated set of flux-extreme tokens from symmetric_flux_tokens.csv.
For each token compute:
  google_first  = mean frequency 1985-1989
  google_last   = mean frequency 2017-2021
  google_lift   = google_last / google_first
  google_slope  = OLS slope of log(freq) vs year, 1985-2021

Then ask: is trademark-corpus 'lift' (the enrichment in extreme
ΔKL strata) correlated with google_lift? If r ≈ 1, our signal is
just tracking general English-language usage. If r is modest, the
trademark signal is distinct.

Pulls from the unofficial Google Ngram Viewer JSON endpoint, which is
rate-limited but tolerant of small batches. We use 1-sec sleep between
calls to be polite.

Outputs:
  paper/results/google_books_flux.csv
  paper/results/google_books_flux.png
  paper/results/google_books_flux.txt
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote
import requests

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT  = REPO / "paper" / "results"

YEAR_LO, YEAR_HI = 1985, 2021
SLEEP_S = 0.8     # polite delay between requests


def fetch_ngram(token: str) -> dict | None:
    """Returns dict {token, years, freqs} or None on failure."""
    url = "https://books.google.com/ngrams/json"
    params = {
        "content": token,
        "year_start": YEAR_LO,
        "year_end": YEAR_HI,
        "corpus": "en",
        "smoothing": 0,
    }
    try:
        r = requests.get(url, params=params, timeout=20,
                         headers={"User-Agent": "novelty-research/1.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ! fetch fail {token!r}: {e}", flush=True)
        return None
    if not data:
        return {"token": token, "years": [], "freqs": []}
    ts = data[0].get("timeseries") or []
    years = list(range(YEAR_LO, YEAR_LO + len(ts)))
    return {"token": token, "years": years, "freqs": ts}


def stats_for_token(token: str, ts: list[float]) -> dict:
    if not ts or all(f == 0 for f in ts):
        return {"token": token, "google_first": 0.0, "google_last": 0.0,
                "google_lift": float("nan"), "google_slope": float("nan"),
                "google_present": False}
    years = np.arange(YEAR_LO, YEAR_LO + len(ts))
    freqs = np.array(ts)
    first = float(freqs[(years >= YEAR_LO) & (years <= YEAR_LO + 4)].mean()) if len(freqs) >= 5 else 0.0
    last  = float(freqs[(years >= YEAR_HI - 4) & (years <= YEAR_HI)].mean()) if len(freqs) >= 5 else float(freqs[-1])
    lift  = (last / first) if first > 0 else float("inf")
    nz = freqs > 0
    if nz.sum() >= 5:
        slope = float(np.polyfit(years[nz], np.log(freqs[nz]), 1)[0])
    else:
        slope = float("nan")
    return {"token": token, "google_first": first, "google_last": last,
            "google_lift": lift, "google_slope": slope,
            "google_present": bool(nz.sum() > 0)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    flux = pl.read_csv(OUT / "symmetric_flux_tokens.csv").select(
        "stratum", "token", "lift", "survival"
    )
    keep_per = 25
    tokens = []
    for stratum in ("negative", "neutral", "positive"):
        sub = (flux.filter(pl.col("stratum") == stratum)
                   .sort("lift", descending=True).head(keep_per))
        for r in sub.iter_rows(named=True):
            tokens.append({"stratum": stratum, **r})
    print(f"[plan] {len(tokens)} tokens to fetch "
          f"({keep_per} per stratum × 3 strata)", flush=True)

    rows = []
    for i, t in enumerate(tokens, 1):
        token = t["token"]
        print(f"  [{i:>3d}/{len(tokens)}]  {t['stratum']:<8s}  {token}", flush=True)
        ts_data = fetch_ngram(token)
        if ts_data is None:
            rows.append({**t, "google_first": 0.0, "google_last": 0.0,
                         "google_lift": float("nan"), "google_slope": float("nan"),
                         "google_present": False, "google_freqs": []})
        else:
            s = stats_for_token(token, ts_data["freqs"])
            rows.append({**t, **s, "google_freqs": ts_data["freqs"]})
        time.sleep(SLEEP_S)

    # Persist tabular form (drop freq series for csv)
    pl.from_dicts([{k: v for k, v in r.items() if k != "google_freqs"}
                    for r in rows]).write_csv(OUT / "google_books_flux.csv")

    pdf = pl.from_dicts([{k: v for k, v in r.items() if k != "google_freqs"}
                          for r in rows]).to_pandas()
    pdf["log_google_lift"] = np.log(pdf["google_lift"].replace([np.inf, -np.inf], np.nan))
    pdf["log_tm_lift"]     = np.log(pdf["lift"])
    pdf_ok = pdf.dropna(subset=["log_google_lift", "log_tm_lift", "google_slope"])
    pdf_ok = pdf_ok[np.isfinite(pdf_ok["log_google_lift"]) & np.isfinite(pdf_ok["log_tm_lift"])]

    # Correlations
    r_lift_p = pdf_ok["log_tm_lift"].corr(pdf_ok["log_google_lift"], method="pearson")
    r_lift_s = pdf_ok["log_tm_lift"].corr(pdf_ok["log_google_lift"], method="spearman")
    r_slope_p = pdf_ok["log_tm_lift"].corr(pdf_ok["google_slope"], method="pearson")
    r_slope_s = pdf_ok["log_tm_lift"].corr(pdf_ok["google_slope"], method="spearman")
    print()
    print(f"corr(log trademark-lift, log Google-lift):     Pearson={r_lift_p:+.3f}  Spearman={r_lift_s:+.3f}  n={len(pdf_ok)}")
    print(f"corr(log trademark-lift, Google log-slope):    Pearson={r_slope_p:+.3f}  Spearman={r_slope_s:+.3f}")

    # Per-stratum summary
    summary = pdf.groupby("stratum").agg(
        n=("token", "count"),
        mean_tm_lift=("lift", "mean"),
        mean_google_lift=("google_lift", "mean"),
        median_google_lift=("google_lift", "median"),
        mean_google_slope=("google_slope", "mean"),
        share_with_google_data=("google_present", "mean"),
    ).reset_index()
    print()
    print(summary.round(4).to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    palette = {"negative": "#cc4444", "neutral": "#888", "positive": "#229922"}

    ax = axes[0]
    for s in ("negative", "neutral", "positive"):
        sub = pdf_ok[pdf_ok["stratum"] == s]
        ax.scatter(sub["log_google_lift"], sub["log_tm_lift"],
                   s=42, alpha=0.8, color=palette[s], edgecolor="black",
                   label=f"{s}-flux ΔKL stratum  (n={len(sub)})")
    if len(pdf_ok) >= 2:
        m, b = np.polyfit(pdf_ok["log_google_lift"], pdf_ok["log_tm_lift"], 1)
        xs = np.linspace(pdf_ok["log_google_lift"].min(), pdf_ok["log_google_lift"].max(), 50)
        ax.plot(xs, m * xs + b, ls="--", color="#444", lw=1, label=f"OLS slope = {m:.2f}")
    ax.axhline(0, ls=":", color="#999")
    ax.axvline(0, ls=":", color="#999")
    ax.set_xlabel(r"$\log$ Google Books frequency lift (2017--2021 / 1985--1989)")
    ax.set_ylabel(r"$\log$ trademark-corpus enrichment lift in $\Delta KL$ stratum")
    ax.set_title(f"Trademark flux vs general-corpus usage (Pearson r = {r_lift_p:+.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Right panel: per-stratum mean Google slope (does positive-flux stratum have positive Google slope?)
    ax = axes[1]
    means = summary.set_index("stratum")["mean_google_slope"]
    stds  = pdf.groupby("stratum")["google_slope"].std()
    order = ["negative", "neutral", "positive"]
    xs = range(len(order))
    ax.bar(xs, [means[s] for s in order],
           yerr=[stds[s]/np.sqrt(summary.set_index("stratum").loc[s,"n"]) for s in order],
           color=[palette[s] for s in order], edgecolor="black", alpha=0.85)
    ax.axhline(0, ls=":", color="#444")
    ax.set_xticks(xs); ax.set_xticklabels(order)
    ax.set_ylabel(r"mean Google Books $\log$-frequency slope (1/yr)")
    ax.set_title("Google Books log-frequency trend by ΔKL stratum")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "google_books_flux.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Text summary
    lines = ["Google Books n-gram confound test\n",
             f"Tokens fetched: {len(tokens)} ({keep_per} per stratum)",
             f"Tokens with non-zero Google Books data: {int(pdf['google_present'].sum())}",
             f"Tokens used in correlations: {len(pdf_ok)}\n",
             "Per-stratum mean Google log-frequency slope:",
             summary.round(4).to_string(index=False),
             "",
             f"Cross-token correlations between trademark-flux lift and Google-corpus lift:",
             f"  log(trademark lift) ~ log(Google lift):  Pearson r = {r_lift_p:+.3f}  Spearman = {r_lift_s:+.3f}",
             f"  log(trademark lift) ~ Google log-slope:  Pearson r = {r_slope_p:+.3f}  Spearman = {r_slope_s:+.3f}",
             "",
             "Interpretation:",
             "  r near 1.0 -> trademark flux just tracks general English-language usage.",
             "  r near 0.0 -> trademark flux is a corpus-specific signal.",
             ]
    (OUT / "google_books_flux.txt").write_text("\n".join(lines))
    print(f"\n[done] wrote csv + png + txt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
