"""Apples-to-apples test: for each candidate token, compute its
per-year frequency in the trademark corpus (parallel to the per-year
Google Books frequency we already have) and compare them as time-series.

For each token T:
  tm_freq(T, y)     = share of clean filings in year y whose g/s text contains T
  google_freq(T, y) = Google Books English n-gram frequency in year y

Then:
  tm_lift     = mean(tm_freq, 2017-2021) / mean(tm_freq, 1985-1989)
  google_lift = mean(google_freq, 2017-2021) / mean(google_freq, 1985-1989)

If trademark flux is just measuring general English-language usage,
log(tm_lift) and log(google_lift) should be near-perfectly correlated
across tokens. If trademark flux is corpus-specific, the correlation
should be much weaker.

Also computes per-year correlation: for each year, the cross-sectional
correlation between log(tm_freq) and log(google_freq) across tokens.

Outputs:
  paper/results/google_books_tm_token_flux.csv
  paper/results/google_books_tm_token_flux.png
  paper/results/google_books_tm_token_flux.txt
"""

from __future__ import annotations

import gc
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

YEAR_LO, YEAR_HI = 1985, 2021


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Load the 75 tokens + Google data we already fetched
    gbf = pl.read_csv(OUT / "google_books_flux.csv")
    tokens = gbf["token"].to_list()
    print(f"[plan] {len(tokens)} tokens. Counting tm-corpus presence by year …",
          flush=True)

    # Split each token into its constituent words. For unigrams, we
    # check set membership of the doc's tokenization. For bigrams we
    # check substring (cheaper than regex when restricted to known docs).
    unigrams = {t for t in tokens if " " not in t and "-" not in t}
    multigrams = {t for t in tokens if t not in unigrams}
    # Compile lower-cased substring patterns for multi-word/hyphenated tokens
    multigram_lc = {t: t.lower() for t in multigrams}
    word_re = re.compile(r"[A-Za-z][A-Za-z]+")

    tok_year = defaultdict(lambda: defaultdict(int))
    year_total = defaultdict(int)
    for cls in class_list():
        tm_p = PROC / f"tm_class{cls}.parquet"
        df = pl.read_parquet(tm_p, columns=["filing_date","goods_services"]).filter(
            pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
            & pl.col("goods_services").is_not_null()
        ).with_columns(
            pl.col("filing_date").str.slice(0,4).cast(pl.Int32).alias("y")
        ).filter(pl.col("y").is_between(YEAR_LO, YEAR_HI))
        if df.height == 0:
            continue
        ys = df["y"].to_numpy()
        gs = df["goods_services"].to_list()
        for j, doc in enumerate(gs):
            y = int(ys[j])
            year_total[y] += 1
            lc = doc.lower()
            # unigram presence via tokenization set
            words = set(word_re.findall(lc))
            for t in unigrams:
                if t in words:
                    tok_year[t][y] += 1
            # multi-gram via substring on lowercased doc
            for t, lcp in multigram_lc.items():
                if lcp in lc:
                    tok_year[t][y] += 1
        print(f"  class {cls}: {df.height:>9,} docs", flush=True)
        del df, gs
        gc.collect()

    # Build per-token per-year frequency
    rows = []
    for t in tokens:
        for y in range(YEAR_LO, YEAR_HI + 1):
            n_total = year_total.get(y, 0)
            n_present = tok_year[t].get(y, 0)
            if n_total > 0:
                rows.append({"token": t, "year": y,
                             "tm_freq": n_present / n_total,
                             "n_total_filings": n_total,
                             "n_filings_with_token": n_present})
    tm_panel = pl.from_dicts(rows)

    # Compute tm_lift (mean of last 5y / mean of first 5y)
    def lift(sub: pl.DataFrame, col: str = "tm_freq") -> float:
        last5 = sub.filter(pl.col("year").is_between(2017, 2021))[col].mean()
        first5 = sub.filter(pl.col("year").is_between(1985, 1989))[col].mean()
        if first5 is None or first5 == 0:
            return float("inf") if last5 and last5 > 0 else float("nan")
        return last5 / first5

    summary_rows = []
    for t in tokens:
        sub = tm_panel.filter(pl.col("token") == t)
        if sub.height == 0:
            continue
        tm_lift = lift(sub, "tm_freq")
        # Trademark slope of log(tm_freq) vs year
        years = sub["year"].to_numpy()
        freqs = sub["tm_freq"].to_numpy()
        nz = freqs > 0
        tm_slope = float(np.polyfit(years[nz], np.log(freqs[nz]), 1)[0]) if nz.sum() >= 5 else float("nan")
        # Pull google_lift / google_slope from the prior table
        grow = gbf.filter(pl.col("token") == t)
        glift = float(grow["google_lift"][0]) if grow.height else float("nan")
        gslope = float(grow["google_slope"][0]) if grow.height else float("nan")
        stratum = grow["stratum"][0] if grow.height else "?"
        summary_rows.append({
            "stratum": stratum, "token": t,
            "tm_lift": tm_lift, "tm_slope": tm_slope,
            "google_lift": glift, "google_slope": gslope,
        })

    sdf = pl.from_dicts(summary_rows)
    sdf.write_csv(OUT / "google_books_tm_token_flux.csv")
    pdf = sdf.to_pandas()
    pdf["log_tm_lift"] = np.log(pdf["tm_lift"].replace([np.inf, -np.inf], np.nan))
    pdf["log_g_lift"]  = np.log(pdf["google_lift"].replace([np.inf, -np.inf], np.nan))
    ok = pdf.dropna(subset=["log_tm_lift", "log_g_lift", "tm_slope", "google_slope"])
    ok = ok[np.isfinite(ok["log_tm_lift"]) & np.isfinite(ok["log_g_lift"])]

    r_lift_p = ok["log_tm_lift"].corr(ok["log_g_lift"], method="pearson")
    r_lift_s = ok["log_tm_lift"].corr(ok["log_g_lift"], method="spearman")
    r_slope_p = ok["tm_slope"].corr(ok["google_slope"], method="pearson")
    r_slope_s = ok["tm_slope"].corr(ok["google_slope"], method="spearman")
    print()
    print(f"=== APPLES-TO-APPLES TOKEN-LEVEL TIME-SERIES TEST ===")
    print(f"corr(log tm-lift, log Google-lift):     Pearson={r_lift_p:+.3f}  Spearman={r_lift_s:+.3f}  n={len(ok)}")
    print(f"corr(tm log-slope, Google log-slope):   Pearson={r_slope_p:+.3f}  Spearman={r_slope_s:+.3f}")

    # Plot scatter with token labels for the top-movers
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    palette = {"negative": "#cc4444", "neutral": "#888", "positive": "#229922"}

    ax = axes[0]
    for s in ("negative","neutral","positive"):
        sub = ok[ok["stratum"] == s]
        ax.scatter(sub["log_g_lift"], sub["log_tm_lift"],
                   s=44, alpha=0.8, color=palette[s], edgecolor="black",
                   label=f"{s}-flux ΔKL stratum  (n={len(sub)})")
    # label some interesting tokens
    n_label = min(20, len(ok))
    labelled = ok.sort_values("log_tm_lift", ascending=False).head(n_label // 2)
    labelled = pl.concat([pl.from_pandas(labelled), pl.from_pandas(ok.sort_values("log_tm_lift").head(n_label // 2))]).to_pandas()
    for _, r in labelled.iterrows():
        ax.annotate(r["token"], (r["log_g_lift"], r["log_tm_lift"]),
                    fontsize=7, alpha=0.7, color="#333", xytext=(3,3),
                    textcoords="offset points")
    if len(ok) >= 2:
        m, b = np.polyfit(ok["log_g_lift"], ok["log_tm_lift"], 1)
        xs = np.linspace(ok["log_g_lift"].min(), ok["log_g_lift"].max(), 50)
        ax.plot(xs, m*xs + b, ls="--", color="#444", lw=1, label=f"OLS slope = {m:.2f}")
    ax.axhline(0, ls=":", color="#999"); ax.axvline(0, ls=":", color="#999")
    ax.set_xlabel(r"$\log$ Google Books frequency lift (2017--2021 / 1985--1989)")
    ax.set_ylabel(r"$\log$ trademark-corpus frequency lift (same windows)")
    ax.set_title(f"Token-level time-series flux\nPearson r = {r_lift_p:+.2f}  (Spearman {r_lift_s:+.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    for s in ("negative","neutral","positive"):
        sub = ok[ok["stratum"] == s]
        ax.scatter(sub["google_slope"], sub["tm_slope"],
                   s=44, alpha=0.8, color=palette[s], edgecolor="black",
                   label=f"{s}-flux stratum")
    if len(ok) >= 2:
        m, b = np.polyfit(ok["google_slope"], ok["tm_slope"], 1)
        xs = np.linspace(ok["google_slope"].min(), ok["google_slope"].max(), 50)
        ax.plot(xs, m*xs + b, ls="--", color="#444", lw=1, label=f"OLS slope = {m:.2f}")
    ax.axhline(0, ls=":", color="#999"); ax.axvline(0, ls=":", color="#999")
    ax.set_xlabel(r"Google Books $\log$-frequency slope (1/yr)")
    ax.set_ylabel(r"trademark $\log$-frequency slope (1/yr)")
    ax.set_title(f"Time-trend correlation\nPearson r = {r_slope_p:+.2f}  (Spearman {r_slope_s:+.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "google_books_tm_token_flux.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Text summary
    lines = ["Apples-to-apples token-level flux: trademark-corpus vs Google Books.\n",
             f"Tokens with usable data on both sides: {len(ok)}",
             "",
             f"corr(log trademark-lift, log Google-lift):  Pearson r = {r_lift_p:+.3f}  Spearman = {r_lift_s:+.3f}",
             f"corr(trademark slope,    Google slope):    Pearson r = {r_slope_p:+.3f}  Spearman = {r_slope_s:+.3f}",
             "",
             "Per-stratum medians:",
             ]
    for s in ("negative","neutral","positive"):
        sub = ok[ok["stratum"] == s]
        lines.append(f"  {s:<10s}  n={len(sub):>2d}  median log_tm_lift={sub['log_tm_lift'].median():+.2f}  median log_g_lift={sub['log_g_lift'].median():+.2f}")
    (OUT / "google_books_tm_token_flux.txt").write_text("\n".join(lines))
    print("\n[done] wrote .csv .png .txt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
