"""Cross-class vocabulary transfer: do filings that import vocabulary from
another NICE class perform better or worse than within-class filings?

Construct
---------
For each (class A, year T) we compute the per-token "rising rate" — recent
frequency in [T-2, T-0] vs older frequency in [T-4, T-2). A token is "rising
in class A at T" if both windows have enough mass, the recent share is
materially larger than the older share, AND the recent rate exceeds an
absolute floor (so tiny tokens don't blow up the ratio).

For each clean-panel filing (class B, year T) we tokenize its goods/services
prose and check whether any of its tokens were rising in some other class A
(A != B) at T but NOT rising in B at T. If yes, that filing is a
*cross-class transfer*. The source class A is the class with the largest
rising-rate for the imported token; the import-strength score is the sum of
that rate over imported tokens, scaled by total terms in the filing.

We then ask: do transfer filings reach registration, survive 5y, and earn
higher prospective KL ΔKL than non-transfer filings, conditional on year
and destination class.

Outputs
-------
  data/processed/cross_class_transfer_panel.parquet
  paper/results/cross_class_transfer.txt
  paper/results/cross_class_transfer.png
  paper/results/cross_class_transfer_examples.txt

Run:
  PYTHONPATH=src .venv/bin/python scripts/cross_class_transfer.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RESULTS = REPO / "paper" / "results"

CLASSES = list(range(1, 46))
YEAR_LO = 1990  # need T-4 to T-2 lookback for the earliest cohort we use
YEAR_HI = 2021

# Rising-token thresholds. These are the load-bearing knobs.
RECENT_MIN_COUNT = 50       # token must appear >=50x in the recent 2y window of the class
RECENT_MIN_RATE = 5e-5      # token must be at least 5e-5 of class vocabulary in recent window
RISE_RATIO = 2.0            # recent_rate / old_rate (with Laplace +1 in old window) >= 2.0

# Filings panel
ANALYSIS_YEAR_LO = 1995
ANALYSIS_YEAR_HI = 2018

# Top-K rising tokens per (class, year) to keep — caps memory
TOP_K_RISING = 500

# Maximum number of source classes a token can be rising in simultaneously
# and still count as a "class-A specific" rise. If it's rising in many
# classes at once, it's a megatrend (e.g. "internet" in 2001 was rising in
# 42 of 45 classes), not a cross-class transfer.
MAX_SOURCE_CLASSES = 3

TOKEN_RE = r"[a-z][a-z'-]+"


def _load_stopwords() -> list[str]:
    from novelty.dictionary import STOPWORDS

    return list(STOPWORDS)


def _year_class_tokens(c: int, stop: list[str]) -> pl.DataFrame:
    """Return per-(year, token) unigram counts for class c.

    Process in year batches to keep memory bounded — for the big classes
    (9, 35, 41, 42) the full explode would be tens of GB.
    """
    import gc

    df = pl.read_parquet(
        PROC / f"tm_class{c:03d}.parquet",
        columns=["filing_date", "goods_services"],
    ).filter(
        pl.col("filing_date").is_not_null()
        & (pl.col("filing_date").str.len_chars() >= 4)
        & (pl.col("goods_services").str.len_chars() > 0)
    ).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year")
    ).filter((pl.col("year") >= YEAR_LO) & (pl.col("year") <= YEAR_HI))

    stop_set = set(stop)
    pieces = []
    years = sorted(df["year"].unique().to_list())
    for y in years:
        sub = df.filter(pl.col("year") == y).select("year", "goods_services")
        if sub.height == 0:
            continue
        tok = sub.with_columns(
            pl.col("goods_services").str.to_lowercase().str.extract_all(TOKEN_RE).alias("toks")
        ).select("year", "toks")
        exploded = tok.explode("toks").filter(
            pl.col("toks").is_not_null() & ~pl.col("toks").is_in(list(stop_set))
        )
        counts = (
            exploded.group_by(["year", "toks"])
            .len()
            .rename({"toks": "token", "len": "count"})
        )
        pieces.append(counts)
        del sub, tok, exploded
        gc.collect()

    out = (
        pl.concat(pieces, how="vertical_relaxed")
        .with_columns(pl.lit(c).cast(pl.Int32).alias("klass"))
    )
    return out


def _build_year_class_token_counts(stop: list[str]) -> pl.DataFrame:
    """Per-(class, year, token) unigram counts across all 45 classes.

    We write per-class shards to PROC/_cct_shards/, then read them back.
    Each class is processed independently so memory does not accumulate.
    We keep only rows where the token has nontrivial count in any year
    (>=10) to keep the next step tractable.
    """
    import gc

    shard_dir = PROC / "_cct_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    for c in CLASSES:
        out = shard_dir / f"yct_class{c:03d}.parquet"
        if out.exists():
            print(f"  class {c:03d}: cached -> {out.name}", file=sys.stderr)
            continue
        t0 = time.time()
        df = _year_class_tokens(c, stop)
        keep = df.group_by("token").agg(pl.col("count").max().alias("mx"))
        keep = keep.filter(pl.col("mx") >= 10).select("token")
        df = df.join(keep, on="token", how="inner")
        df.write_parquet(out)
        print(
            f"  class {c:03d}: {df.height:,} (year,token) rows after pruning "
            f"({time.time()-t0:.1f}s)",
            file=sys.stderr,
        )
        del df, keep
        gc.collect()

    # Now load all shards and concat
    print("  concatenating shards…", file=sys.stderr)
    shards = sorted(shard_dir.glob("yct_class*.parquet"))
    out = pl.concat([pl.read_parquet(s) for s in shards], how="vertical_relaxed")
    print(f"[counts] total rows: {out.height:,}", file=sys.stderr)
    return out


def _build_class_year_totals(yct: pl.DataFrame) -> pl.DataFrame:
    """Total unigram-token volume per (class, year)."""
    return yct.group_by(["klass", "year"]).agg(pl.col("count").sum().alias("year_total"))


def _rising_for_class(yct_c: pl.DataFrame, totals_c: pl.DataFrame, c: int) -> pl.DataFrame:
    """Compute rising tokens for a single class. Loops over T values."""
    yct_c = yct_c.select(["year", "token", "count"])
    totals_c = totals_c.select(["year", "year_total"])
    by_year_total = dict(zip(totals_c["year"].to_list(), totals_c["year_total"].to_list()))

    Ts = sorted(totals_c["year"].to_list())
    pieces = []
    for T in Ts:
        if T < YEAR_LO + 3:
            continue  # need T-3
        recent_years = [T - 1, T]
        old_years = [T - 3, T - 2]
        if not all(y in by_year_total for y in recent_years) or not all(
            y in by_year_total for y in old_years
        ):
            continue
        recent_total = sum(by_year_total[y] for y in recent_years)
        old_total = sum(by_year_total[y] for y in old_years)
        if recent_total == 0 or old_total == 0:
            continue
        recent = (
            yct_c.filter(pl.col("year").is_in(recent_years))
            .group_by("token")
            .agg(pl.col("count").sum().alias("recent_count"))
        )
        old = (
            yct_c.filter(pl.col("year").is_in(old_years))
            .group_by("token")
            .agg(pl.col("count").sum().alias("old_count"))
        )
        df = (
            recent.join(old, on="token", how="left")
            .with_columns(
                [
                    pl.col("old_count").fill_null(0),
                    (pl.col("recent_count") / recent_total).alias("recent_rate"),
                    ((pl.col("old_count").fill_null(0) + 1) / (old_total + 1)).alias("old_rate"),
                ]
            )
            .with_columns(
                (pl.col("recent_rate") / pl.col("old_rate")).alias("rise_ratio")
            )
            .filter(
                (pl.col("recent_count") >= RECENT_MIN_COUNT)
                & (pl.col("recent_rate") >= RECENT_MIN_RATE)
                & (pl.col("rise_ratio") >= RISE_RATIO)
            )
            .sort("recent_rate", descending=True)
            .head(TOP_K_RISING)
            .with_columns(
                [
                    pl.lit(c).cast(pl.Int32).alias("source_class"),
                    pl.lit(T).cast(pl.Int32).alias("year"),
                ]
            )
            .select(["source_class", "year", "token", "recent_rate", "rise_ratio"])
        )
        pieces.append(df)
    if not pieces:
        return pl.DataFrame(
            schema={
                "source_class": pl.Int32,
                "year": pl.Int32,
                "token": pl.Utf8,
                "recent_rate": pl.Float64,
                "rise_ratio": pl.Float64,
            }
        )
    return pl.concat(pieces, how="vertical_relaxed")


def _build_rising(yct: pl.DataFrame, totals: pl.DataFrame) -> pl.DataFrame:
    """Build the (source_class, year, token) rising table by iterating classes."""
    import gc

    pieces = []
    for c in CLASSES:
        t0 = time.time()
        yct_c = yct.filter(pl.col("klass") == c)
        totals_c = totals.filter(pl.col("klass") == c)
        rs = _rising_for_class(yct_c, totals_c, c)
        pieces.append(rs)
        print(
            f"  rising class {c:03d}: {rs.height:,} (T, token) entries "
            f"({time.time()-t0:.1f}s)",
            file=sys.stderr,
        )
        del yct_c, totals_c, rs
        gc.collect()

    rising = pl.concat(pieces, how="vertical_relaxed")
    print(
        f"[rising] {rising.height:,} (class, year, token) rising entries total "
        f"(top-{TOP_K_RISING} per class-year, RECENT_MIN_COUNT={RECENT_MIN_COUNT}, "
        f"RECENT_MIN_RATE={RECENT_MIN_RATE}, RISE_RATIO={RISE_RATIO}).",
        file=sys.stderr,
    )
    return rising


def _build_filing_panel(c: int, stop: list[str], rising: pl.DataFrame) -> pl.DataFrame:
    """Build the per-filing transfer panel for one destination class B = c.

    For each clean filing in class c at year T:
      - tokenize goods/services
      - compute tokens that are rising in some A != c at T but NOT rising in c at T
      - source_class = A with the largest recent_rate for that token
      - import_strength = sum of source-class recent_rate over imported tokens,
                          divided by total terms in the filing (so it's
                          intensity-per-term)
    Emit one row per filing, joined to surprise + outcome columns.
    """
    sp = pl.read_parquet(PROC / f"surprise_class{c:03d}.parquet").filter(
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
        & (pl.col("year") >= ANALYSIS_YEAR_LO)
        & (pl.col("year") <= ANALYSIS_YEAR_HI)
    ).select(
        "serial_number",
        "year",
        "n_terms",
        "prospective_kl",
        "retrospective_kl",
    )
    if sp.height == 0:
        return pl.DataFrame()

    oc = pl.read_parquet(
        PROC / f"outcomes_class{c:03d}.parquet",
        columns=["serial_number", "reached_registration", "survived_5y"],
    )
    tm = pl.read_parquet(
        PROC / f"tm_class{c:03d}.parquet",
        columns=["serial_number", "goods_services"],
    ).filter(pl.col("goods_services").str.len_chars() > 0)

    base = sp.join(tm, on="serial_number", how="inner").join(
        oc, on="serial_number", how="left"
    )

    # tokenize
    tk = base.with_columns(
        pl.col("goods_services").str.to_lowercase().str.extract_all(TOKEN_RE).alias("toks")
    ).select("serial_number", "year", "n_terms", "toks")
    long = tk.explode("toks").filter(
        pl.col("toks").is_not_null() & ~pl.col("toks").is_in(stop)
    ).rename({"toks": "token"})

    # rising in this destination class B = c at the same year (so we exclude these tokens)
    rising_dest = rising.filter(pl.col("source_class") == c).select(["year", "token"]).with_columns(
        pl.lit(True).alias("rising_in_dest")
    )

    # Megatrend filter: only consider tokens that are rising in <= MAX_SOURCE_CLASSES
    # classes simultaneously. A token rising in 30+ classes is not a cross-class
    # transfer, it's a corpus-wide megatrend.
    n_classes_per = (
        rising.group_by(["year", "token"])
        .len()
        .rename({"len": "n_source_classes"})
    )
    rising_filtered = rising.join(n_classes_per, on=["year", "token"], how="inner").filter(
        pl.col("n_source_classes") <= MAX_SOURCE_CLASSES
    )

    # rising in SOME other class A != c at year T (with megatrend filter applied)
    rising_others = (
        rising_filtered.filter(pl.col("source_class") != c)
        .sort(["year", "token", "recent_rate"], descending=[False, False, True])
        .group_by(["year", "token"])
        .agg(
            [
                pl.col("source_class").first().alias("source_class"),
                pl.col("recent_rate").first().alias("source_recent_rate"),
            ]
        )
    )

    # join: for each filing-token, mark as rising elsewhere AND not rising in dest
    joined = (
        long.join(rising_others, on=["year", "token"], how="left")
        .join(rising_dest, on=["year", "token"], how="left")
        .filter(pl.col("source_class").is_not_null() & pl.col("rising_in_dest").is_null())
    )

    # collapse per-filing. Take the source_class with the largest contribution.
    # n_imported_tokens counts DISTINCT imported tokens (not occurrences) so the
    # measure isn't dominated by repeated bigrams like 'machines, machines'.
    imports = (
        joined.sort(["serial_number", "source_recent_rate"], descending=[False, True])
        .group_by("serial_number")
        .agg(
            [
                pl.col("source_recent_rate").sum().alias("import_strength_raw"),
                pl.col("token").n_unique().alias("n_imported_tokens"),
                pl.col("source_class").first().alias("dominant_source_class"),
                pl.col("token").unique().head(8).alias("example_tokens"),
            ]
        )
    )

    out = (
        base.select(
            ["serial_number", "year", "n_terms", "prospective_kl", "retrospective_kl",
             "reached_registration", "survived_5y"]
        )
        .join(imports, on="serial_number", how="left")
        .with_columns(
            [
                pl.lit(c).cast(pl.Int32).alias("dest_class"),
                pl.col("n_imported_tokens").fill_null(0),
                pl.col("import_strength_raw").fill_null(0.0),
                pl.col("dominant_source_class").fill_null(-1).cast(pl.Int32),
                (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
                # intensity normalized by number of terms in the filing
                (pl.col("import_strength_raw").fill_null(0.0)
                 / pl.col("n_terms").cast(pl.Float64)).alias("import_intensity"),
                # A "transfer" filing imports at least 2 distinct rising tokens
                # from another class. n=1 is too noisy (single accidental match).
                (pl.col("n_imported_tokens").fill_null(0) >= 2).alias("is_transfer"),
            ]
        )
    )
    return out


def _summary_table(panel: pl.DataFrame) -> dict:
    """Compute headline numbers."""
    n_tot = panel.height
    n_tr = panel.filter(pl.col("is_transfer")).height
    rate = n_tr / max(n_tot, 1)

    def mean_se(df: pl.DataFrame, col: str) -> tuple[float, float, int]:
        x = df.select(pl.col(col).cast(pl.Float64)).drop_nulls()
        if x.height == 0:
            return (float("nan"), float("nan"), 0)
        vals = x[col].to_numpy()
        return (float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals))), int(len(vals)))

    rows = []
    for name, sub in [
        ("non_transfer", panel.filter(~pl.col("is_transfer"))),
        ("transfer", panel.filter(pl.col("is_transfer"))),
    ]:
        m, se, n = mean_se(sub, "reached_registration")
        m5, se5, n5 = mean_se(sub, "survived_5y")
        md, sed, nd = mean_se(sub, "dkl")
        rows.append(
            dict(
                group=name,
                n=sub.height,
                mean_reg=m,
                se_reg=se,
                n_reg=n,
                mean_5y=m5,
                se_5y=se5,
                n_5y=n5,
                mean_dkl=md,
                se_dkl=sed,
                n_dkl=nd,
            )
        )

    return dict(
        n_total=n_tot,
        n_transfer=n_tr,
        transfer_rate=rate,
        groups=rows,
    )


def _diff_z(mean_a: float, se_a: float, mean_b: float, se_b: float) -> tuple[float, float]:
    diff = mean_a - mean_b
    se = float(np.sqrt(se_a ** 2 + se_b ** 2))
    z = diff / se if se > 0 else float("nan")
    return diff, z


def _yearly_plot(panel: pl.DataFrame, outpath: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yr = (
        panel.group_by(["year", "is_transfer"])
        .agg(
            [
                pl.col("reached_registration").mean().alias("reg"),
                pl.col("survived_5y").mean().alias("sv5"),
                pl.col("dkl").mean().alias("dkl"),
                pl.len().alias("n"),
            ]
        )
        .sort(["year", "is_transfer"])
    )
    transfer_share = (
        panel.group_by("year")
        .agg(pl.col("is_transfer").mean().alias("share"))
        .sort("year")
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)

    # panel A: share over time
    ax = axes[0, 0]
    ax.plot(transfer_share["year"], transfer_share["share"], color="black", lw=1.6)
    ax.set_ylabel("share transfer filings")
    ax.set_title("A. Cross-class transfer share over time")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)

    # panel B: registration
    ax = axes[0, 1]
    for flag, color, label in [(False, "#888888", "non-transfer"), (True, "#c0392b", "transfer")]:
        sub = yr.filter(pl.col("is_transfer") == flag)
        ax.plot(sub["year"], sub["reg"], color=color, lw=1.6, label=label)
    ax.set_ylabel("reached registration")
    ax.set_title("B. Registration rate by year")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # panel C: survival
    ax = axes[1, 0]
    for flag, color, label in [(False, "#888888", "non-transfer"), (True, "#c0392b", "transfer")]:
        sub = yr.filter(pl.col("is_transfer") == flag)
        ax.plot(sub["year"], sub["sv5"], color=color, lw=1.6, label=label)
    ax.set_ylabel("5-year survival")
    ax.set_xlabel("filing year")
    ax.set_title("C. 5-year survival by year")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    # panel D: ΔKL
    ax = axes[1, 1]
    for flag, color, label in [(False, "#888888", "non-transfer"), (True, "#c0392b", "transfer")]:
        sub = yr.filter(pl.col("is_transfer") == flag)
        ax.plot(sub["year"], sub["dkl"], color=color, lw=1.6, label=label)
    ax.set_ylabel(r"$\Delta KL$ (prospective − retrospective)")
    ax.set_xlabel("filing year")
    ax.set_title(r"D. $\Delta KL$ by year")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Cross-class vocabulary transfer vs same-class filings (clean panel, 1995–2018)",
        fontsize=11,
        y=1.00,
    )
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _intensity_quintile_analysis(panel: pl.DataFrame) -> pl.DataFrame:
    """Bin filings by import_intensity quintile (within their dest_class × year cell)
    and compute mean outcomes per quintile.
    """
    p = panel.filter(pl.col("import_intensity").is_not_null())
    p = p.with_columns(
        pl.col("import_intensity").qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).alias("intensity_q")
    )
    summary = (
        p.group_by("intensity_q")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("reached_registration").mean().alias("reg_mean"),
                pl.col("survived_5y").mean().alias("sv5_mean"),
                pl.col("dkl").mean().alias("dkl_mean"),
                pl.col("import_intensity").mean().alias("intensity_mean"),
                pl.col("import_intensity").min().alias("intensity_min"),
                pl.col("import_intensity").max().alias("intensity_max"),
            ]
        )
        .sort("intensity_q")
    )
    return summary


def _within_class_year_test(panel: pl.DataFrame) -> dict:
    """Within (dest_class, year) average difference (transfer - nontransfer)."""
    panel = panel.with_columns(
        [
            pl.col("is_transfer").cast(pl.Int8),
            pl.col("reached_registration").cast(pl.Float64),
            pl.col("survived_5y").cast(pl.Float64),
            pl.col("dkl").cast(pl.Float64),
        ]
    )
    cell = (
        panel.group_by(["dest_class", "year", "is_transfer"])
        .agg(
            [
                pl.col("reached_registration").mean().alias("reg"),
                pl.col("survived_5y").mean().alias("sv5"),
                pl.col("dkl").mean().alias("dkl"),
                pl.len().alias("n"),
            ]
        )
    )
    # pivot
    pv_reg = cell.pivot(values="reg", index=["dest_class", "year"], on="is_transfer")
    pv_sv5 = cell.pivot(values="sv5", index=["dest_class", "year"], on="is_transfer")
    pv_dkl = cell.pivot(values="dkl", index=["dest_class", "year"], on="is_transfer")
    pv_n = cell.pivot(values="n", index=["dest_class", "year"], on="is_transfer")

    def diff(pv, pv_n):
        cols = pv.columns
        if "0" in cols and "1" in cols:
            non = pv["0"]; tr = pv["1"]; nn = pv_n["0"]; nt = pv_n["1"]
        else:
            non = pv["false"] if "false" in cols else pv[False]
            tr = pv["true"] if "true" in cols else pv[True]
            nn = pv_n["false"] if "false" in pv_n.columns else pv_n[False]
            nt = pv_n["true"] if "true" in pv_n.columns else pv_n[True]
        mask = (non.is_not_null()) & (tr.is_not_null()) & (nn > 30) & (nt > 30)
        d = (tr - non).filter(mask)
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.len())), int(d.len())

    res = {}
    res["reg_within"] = diff(pv_reg, pv_n)
    res["sv5_within"] = diff(pv_sv5, pv_n)
    res["dkl_within"] = diff(pv_dkl, pv_n)
    return res


def _examples(panel: pl.DataFrame, n: int = 30) -> pl.DataFrame:
    """Pull a handful of transfer filings with highest import intensity for face validity."""
    ex = (
        panel.filter(pl.col("is_transfer") & (pl.col("year") >= 2005) & (pl.col("year") <= 2015))
        .sort("import_intensity", descending=True)
        .head(n)
        .select(
            [
                "serial_number",
                "year",
                "dest_class",
                "dominant_source_class",
                "n_imported_tokens",
                "import_intensity",
                "dkl",
                "reached_registration",
                "survived_5y",
                "example_tokens",
            ]
        )
    )
    return ex


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    t0 = time.time()
    stop = _load_stopwords()
    cache_yct = PROC / "_cct_year_class_token_counts.parquet"
    cache_rising = PROC / "_cct_rising_tokens.parquet"

    if args.rebuild or not cache_yct.exists():
        print("[step 1/3] per-(class, year, token) counts", file=sys.stderr)
        yct = _build_year_class_token_counts(stop)
        yct.write_parquet(cache_yct)
    else:
        print(f"[step 1/3] cached -> {cache_yct}", file=sys.stderr)
        yct = pl.read_parquet(cache_yct)

    totals = _build_class_year_totals(yct)
    print(f"[totals] (class,year) cells: {totals.height:,}", file=sys.stderr)

    if args.rebuild or not cache_rising.exists():
        print("[step 2/3] rising-token table", file=sys.stderr)
        rising = _build_rising(yct, totals)
        rising.write_parquet(cache_rising)
    else:
        print(f"[step 2/3] cached -> {cache_rising}", file=sys.stderr)
        rising = pl.read_parquet(cache_rising)

    del yct  # free

    print("[step 3/3] per-filing transfer panel", file=sys.stderr)
    import gc

    panel_shard_dir = PROC / "_cct_panel_shards"
    panel_shard_dir.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        out_p = panel_shard_dir / f"panel_class{c:03d}.parquet"
        if out_p.exists() and not args.rebuild:
            print(f"  dest class {c:03d}: cached", file=sys.stderr)
            continue
        tc = time.time()
        p = _build_filing_panel(c, stop, rising)
        if p.height > 0:
            p.write_parquet(out_p)
        print(
            f"  dest class {c:03d}: {p.height:,} filings "
            f"({p.filter(pl.col('is_transfer')).height:,} transfer)  "
            f"({time.time()-tc:.1f}s)",
            file=sys.stderr,
        )
        del p
        gc.collect()

    shards = sorted(panel_shard_dir.glob("panel_class*.parquet"))
    panel = pl.concat([pl.read_parquet(s) for s in shards], how="vertical_relaxed")
    panel.write_parquet(PROC / "cross_class_transfer_panel.parquet")
    print(
        f"[panel] {panel.height:,} filings; transfer rate "
        f"{panel.filter(pl.col('is_transfer')).height/panel.height:.4f}",
        file=sys.stderr,
    )

    # Summary
    summ = _summary_table(panel)
    within = _within_class_year_test(panel)
    qsumm = _intensity_quintile_analysis(panel)
    ex = _examples(panel)

    # Write text summary
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "cross_class_transfer.txt").open("w") as f:
        f.write("Cross-class vocabulary transfer — headline summary\n")
        f.write("====================================================\n\n")
        f.write(f"Population: clean-panel filings, years {ANALYSIS_YEAR_LO}–{ANALYSIS_YEAR_HI}\n")
        f.write(f"  Total filings: {summ['n_total']:,}\n")
        f.write(f"  Transfer filings: {summ['n_transfer']:,}\n")
        f.write(f"  Transfer rate: {summ['transfer_rate']:.4f}\n\n")
        f.write("Rising-token construct:\n")
        f.write(f"  recent window: 2y; old window: 2y prior to that\n")
        f.write(f"  RECENT_MIN_COUNT={RECENT_MIN_COUNT}\n")
        f.write(f"  RECENT_MIN_RATE={RECENT_MIN_RATE}\n")
        f.write(f"  RISE_RATIO={RISE_RATIO}\n")
        f.write(f"  top-K rising per (class, year): {TOP_K_RISING}\n\n")
        f.write("Group means (raw, pooled across years and classes)\n")
        f.write("----------------------------------------------------\n")
        f.write(
            f"{'group':<14} {'n':>10} {'reg_mean':>10} {'reg_se':>9} "
            f"{'5y_mean':>10} {'5y_se':>9} {'dkl_mean':>10} {'dkl_se':>9}\n"
        )
        for r in summ["groups"]:
            f.write(
                f"{r['group']:<14} {r['n']:>10,} "
                f"{r['mean_reg']:>10.4f} {r['se_reg']:>9.5f} "
                f"{r['mean_5y']:>10.4f} {r['se_5y']:>9.5f} "
                f"{r['mean_dkl']:>10.4f} {r['se_dkl']:>9.5f}\n"
            )
        # raw pooled diffs
        nt = summ["groups"][0]
        tr = summ["groups"][1]
        d_reg, z_reg = _diff_z(tr["mean_reg"], tr["se_reg"], nt["mean_reg"], nt["se_reg"])
        d_5y, z_5y = _diff_z(tr["mean_5y"], tr["se_5y"], nt["mean_5y"], nt["se_5y"])
        d_dkl, z_dkl = _diff_z(tr["mean_dkl"], tr["se_dkl"], nt["mean_dkl"], nt["se_dkl"])
        f.write("\nPooled difference (transfer − non-transfer)\n")
        f.write("----------------------------------------------------\n")
        f.write(f"  registration: {d_reg:+.4f}  (z = {z_reg:+.2f})\n")
        f.write(f"  5y survival:  {d_5y:+.4f}  (z = {z_5y:+.2f})\n")
        f.write(f"  dkl:          {d_dkl:+.4f}  (z = {z_dkl:+.2f})\n\n")
        f.write("Import-intensity quintiles (pooled across destination classes and years)\n")
        f.write("----------------------------------------------------\n")
        f.write(
            f"{'Q':<5} {'n':>10} {'imp_mean':>10} {'reg_mean':>10} "
            f"{'sv5_mean':>10} {'dkl_mean':>10}\n"
        )
        for r in qsumm.iter_rows(named=True):
            f.write(
                f"{str(r['intensity_q']):<5} {r['n']:>10,} "
                f"{r['intensity_mean']:>10.5f} {r['reg_mean']:>10.4f} "
                f"{r['sv5_mean']:>10.4f} {r['dkl_mean']:>10.4f}\n"
            )
        f.write("\n")
        f.write("Within (dest_class, year) cell averages of (transfer − non-transfer)\n")
        f.write("Cells require n>=30 in each arm.\n")
        f.write("----------------------------------------------------\n")
        for k in ("reg_within", "sv5_within", "dkl_within"):
            m, se, n = within[k]
            z = m / se if se > 0 else float("nan")
            f.write(f"  {k}:  mean diff = {m:+.4f}  (se = {se:.5f}, z = {z:+.2f}, n_cells = {n})\n")
        f.write("\nTop transfer examples (highest import intensity, 2005–2015):\n")
        f.write("----------------------------------------------------\n")
        for r in ex.iter_rows(named=True):
            f.write(
                f"  sn={r['serial_number']} y={r['year']} "
                f"B={r['dest_class']:02d}<-A={r['dominant_source_class']:02d} "
                f"n_imp={r['n_imported_tokens']} intensity={r['import_intensity']:.4f} "
                f"dkl={r['dkl']:+.3f} reg={r['reached_registration']} "
                f"sv5={r['survived_5y']} tokens={r['example_tokens']}\n"
            )

    # Plot
    _yearly_plot(panel, RESULTS / "cross_class_transfer.png")

    # Examples in a separate file
    with (RESULTS / "cross_class_transfer_examples.txt").open("w") as f:
        f.write("Top 30 cross-class transfer filings by import_intensity (2005-2015)\n")
        f.write("---------------------------------------------------------------\n")
        for r in ex.iter_rows(named=True):
            f.write(
                f"sn={r['serial_number']} year={r['year']} dest={r['dest_class']:02d} "
                f"src={r['dominant_source_class']:02d} "
                f"n_imported={r['n_imported_tokens']} intensity={r['import_intensity']:.4f} "
                f"dkl={r['dkl']:+.3f} reg={r['reached_registration']} sv5={r['survived_5y']}\n"
            )
            f.write(f"   example tokens: {r['example_tokens']}\n")

    print(f"[done] {time.time()-t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
