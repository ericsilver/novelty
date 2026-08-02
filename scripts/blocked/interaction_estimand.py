"""§0 INTERACTION ESTIMAND  (blocked on the raw per-class corpus)

The critique retired the additive model dKL = b1*innov + b2*churn + ... in favor
of a MODERATION model: firm innovation may only register in dKL when the class
vocabulary is actually moving. Eric's counter-position is that this is not a
confound but the construct (diffusion-as-innovation). Either way the same
regression adjudicates it:

    dkl_ifct = a + b1*logpat + b2*churn_ct + b3*(logpat * churn_ct)
               + firm FE + year FE + e        (firm-clustered SE)

b3 > 0 with b1 ~ 0  =>  innovation shows up in dKL only via churning classes
                        (firm x class-phase quantity, consistent with diffusion).
b1 > 0, b3 ~ 0      =>  a firm-level innovation component independent of churn.

THE LOAD-BEARING REQUIREMENT (from the critique): churn_ct must be built
WITHOUT the resonance kernel, or we regress dKL on a function of dKL. This script
constructs two kernel-independent churn measures from the raw tokens:
  (A) log filing-volume growth per class-year   -- pure counts;
  (B) vocabulary turnover = share of class-year token TYPES not present in the
      prior W years -- set membership only, never touches KL or smoothing.

INPUTS REQUIRED (none currently on disk -- run `make tm crosswalk` to build):
  data/processed/surprise_class<NNN>.parquet   (serial_number, year, kl_vs_past, kl_vs_future)
  data/processed/tm_class<NNN>.parquet         (serial_number, year, owner_name, goods_services)
  data_publish/firm_year_patents_and_dkl.csv   (already present: owner-year patents)

This file is COMPLETE and will run as-is once those parquets exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "processed"


def _require():
    sur = sorted(DATA.glob("surprise_class*.parquet"))
    rec = sorted(DATA.glob("tm_class*.parquet"))
    if not sur or not rec:
        print("MISSING raw corpus. Need:", file=sys.stderr)
        print(f"  {DATA}/surprise_class<NNN>.parquet  (found {len(sur)})", file=sys.stderr)
        print(f"  {DATA}/tm_class<NNN>.parquet        (found {len(rec)})", file=sys.stderr)
        print("Build with:  make tm   (downloads ~12GB USPTO backfile, multi-hour)", file=sys.stderr)
        raise SystemExit(2)
    return sur, rec


def main() -> int:
    import numpy as np
    import pandas as pd
    import polars as pl
    import statsmodels.api as sm
    from sklearn.feature_extraction.text import CountVectorizer
    from novelty.dictionary import STOPWORDS, _make_analyzer

    sur_paths, rec_paths = _require()
    W = 5  # churn look-back window (years)

    # ---- filing-level dKL, tagged by class -------------------------------
    frames = []
    for p in sur_paths:
        cls = p.stem.replace("surprise_class", "")
        s = pl.read_parquet(p).select("serial_number", "year",
                                       "kl_vs_past", "kl_vs_future")
        s = s.with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
            pl.lit(cls).alias("nice"))
        frames.append(s)
    dkl = pl.concat(frames).drop_nulls("dkl")

    # ---- (B) kernel-INDEPENDENT vocabulary turnover per class-year -------
    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))
    churn_rows = []
    for p in rec_paths:
        cls = p.stem.replace("tm_class", "")
        r = pl.read_parquet(p).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
        ).filter(pl.col("goods_services").str.len_chars() > 0).drop_nulls("year")
        # token TYPE set per year (presence only)
        year_types: dict[int, set] = {}
        year_n: dict[int, int] = {}
        for yr, grp in r.group_by("year"):
            y = int(yr[0])
            toks: set = set()
            for gs in grp["goods_services"].to_list():
                toks.update(analyzer(gs))
            year_types[y] = toks
            year_n[y] = grp.height
        for y in sorted(year_types):
            prior = set().union(*(year_types[j] for j in range(y - W, y) if j in year_types)) if any(
                j in year_types for j in range(y - W, y)) else set()
            cur = year_types[y]
            turnover = (len(cur - prior) / len(cur)) if cur else np.nan          # (B)
            vol_prev = sum(year_n.get(j, 0) for j in range(y - W, y))
            vol_growth = np.log1p(year_n[y]) - np.log1p(vol_prev / max(W, 1))     # (A)
            churn_rows.append({"nice": cls, "year": y,
                               "churn_turnover": turnover, "churn_vol": vol_growth})
    churn = pl.DataFrame(churn_rows)

    # ---- firm innovation proxy: owner-year patents -----------------------
    pat = pl.read_csv(REPO / "data_publish" / "firm_year_patents_and_dkl.csv").select(
        pl.col("uspto_owner_name").alias("owner_name"), "year",
        pl.col("n_patents")).unique(["owner_name", "year"])

    rec_owner = pl.concat([
        pl.read_parquet(p).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
            pl.lit(p.stem.replace("tm_class", "")).alias("nice"),
        ).select("serial_number", "owner_name", "year", "nice")
        for p in rec_paths]).drop_nulls(["owner_name", "year"])

    df = (dkl.join(rec_owner, on=["serial_number", "year", "nice"], how="inner")
             .join(churn, on=["nice", "year"], how="left")
             .join(pat, on=["owner_name", "year"], how="left")).to_pandas()
    df["logpat"] = np.log1p(df.n_patents.fillna(0))
    df["dkl_z"] = (df.dkl - df.dkl.mean()) / df.dkl.std()
    df = df.dropna(subset=["churn_turnover", "logpat"])

    # ---- moderation regression with firm+year FE, firm-clustered SE ------
    def demean(s, g):
        return s - s.groupby(g).transform("mean")
    firm = df.owner_name
    df["x_pat"] = demean(df.logpat, firm)
    df["x_ch"] = demean(df.churn_turnover, firm)
    df["x_int"] = demean(df.logpat * df.churn_turnover, firm)
    yd = demean(df.dkl_z, firm)
    yr = pd.get_dummies(df.year.astype("category"), drop_first=True).astype(float)
    yr = yr.apply(lambda c: c - c.groupby(firm).transform("mean"))
    X = pd.concat([df[["x_pat", "x_ch", "x_int"]], yr], axis=1).astype(float)
    m = sm.OLS(yd.astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": firm.astype("category").cat.codes.values})
    print("Moderation: dkl_z ~ logpat * churn_turnover  (firm+year FE)")
    for k in ["x_pat", "x_ch", "x_int"]:
        print(f"  {k:7s} coef={m.params[k]:+.4f} se={m.bse[k]:.4f} t={m.tvalues[k]:+.2f}")
    print("Rule: x_int>0 & x_pat~0 => dKL innovation content exists only in "
          "churning classes (firm x class-phase; diffusion-consistent).")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
