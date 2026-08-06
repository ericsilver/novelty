"""Debut listing on vocabulary position: the paper's design-based listing table.

Produces the coefficients in the paper's listing table and panel B of the
quintile-profile figure. It exists to test whether two raw patterns in the
debut-outcome figure survive design: those raw patterns, from
debut_outcome_topic*.json and robust at T=50/200/500, are

  (1) P(EDGAR | registration) RISES monotonically in KL level
      (past-facing and future-facing alike): ~0.20% -> ~0.50%, ~2.5x.
  (2) P(EDGAR | registration) is U-SHAPED in lead: both tails beat the
      middle (~0.41/0.27/0.34% at T=50).

The tests are class x filing-year FE, log description length, and each other
-- is the U just the unsigned level in disguise? It is: spec C resolves the U
into a positive unsigned component and a negative signed one, which is the
paper's decomposition.

Sample: debut filings (owner's first-ever filing) 1995-2018 that reached
registration; outcome = owner ever in SEC EDGAR. One row per owner, so
heteroskedasticity-robust (HC1) SEs are enough -- there is no within-owner
clustering left to correct. The registration conditioning is deliberate:
registration is itself selected on language, so an unconditional gradient
mixes what the examiner does to a description with what the market does to
the product.

Specs (LPM, class x year FE absorbed by demeaning):
  A1 atypicality quintile dummies (within class x year)      [raw-in-FE]
  A2 A1 + log_len                                            [length]
  A1/A2 repeated on the past level alone, the robustness comparison
  B1 lead quintile dummies                                   [raw-in-FE]
  B2 B1 + log_len
  B3 B2 + past and future KL z-levels   <- does the U survive levels?
  C  decomposition: |lead z| and signed lead z + log_len

Reads   data/processed/tm_class{CLS}.parquet
        data/processed/{SRC}_surprise_class{CLS}.parquet   SRC = SURPRISE_SRC
        data/processed/uspto_sec_crosswalk.parquet
Output: paper/results/debut_edgar_substantiate.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"

# Default is "rolling": the per-filing reference windows every estimate in
# the paper uses. Set SURPRISE_SRC=topic to reproduce the retired
# per-calendar-year scoring, which the comparison appendix reports.
SRC = os.environ.get("SURPRISE_SRC", "rolling")
OUT = REPO / "paper" / "results"
FILE_LO, FILE_HI = 1995, 2018


def all_classes() -> list[str]:
    return sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def build() -> pl.DataFrame:
    print("[debut owners]", file=sys.stderr, flush=True)
    parts = []
    for cls in all_classes():
        d = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet", columns=["owner_name", "filing_date"]
        ).filter(pl.col("owner_name").is_not_null()
                 & (pl.col("filing_date").str.len_chars() == 8)
        ).group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        parts.append(d)
    debut = pl.concat(parts).group_by("owner_name").agg(
        pl.col("debut_date").min().alias("debut_date"))
    del parts
    gc.collect()

    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))

    rows = []
    for cls in all_classes():
        tp = PROC / f"{SRC}_surprise_class{cls}.parquet"
        if not tp.exists():
            continue
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "owner_name", "filing_date",
                     "registration_date", "goods_services"],
        ).filter(pl.col("owner_name").is_not_null()
                 & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False)
            .alias("fy"),
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
            .alias("registered"),
            pl.col("goods_services").fill_null("").str.len_chars().alias("gs_len"),
        ).filter(pl.col("fy").is_between(FILE_LO, FILE_HI))
        j = tm.join(debut, on="owner_name", how="inner").filter(
            pl.col("filing_date") == pl.col("debut_date")
        ).join(
            pl.read_parquet(tp).filter(
                pl.col("topic_dkl").is_finite()
                & pl.col("topic_kl_vs_past").is_finite()
                & pl.col("topic_kl_vs_future").is_finite()).select(
                "serial_number", "topic_dkl", "topic_kl_vs_past", "topic_kl_vs_future"),
            on="serial_number", how="inner",
        ).join(sec, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False),
            pl.lit(cls).alias("cls"),
        ).select("serial_number", "owner_name", "cls", "fy", "registered",
                 "gs_len", "topic_dkl", "topic_kl_vs_past", "topic_kl_vs_future", "in_sec")
        rows.append(j)
        del tm, j
        gc.collect()
    df = pl.concat(rows)
    del rows
    gc.collect()
    # debut filings can tie across classes on the same day: unique serials,
    # then one filing per owner (first serial) to keep one row per owner
    df = df.unique(subset="serial_number", keep="first").sort(
        "serial_number").unique(subset="owner_name", keep="first")
    print(f"[frame] {df.height:,} debut owners", file=sys.stderr, flush=True)
    return df


def lpm_fe(df: pl.DataFrame, xcols: list[str], label: str) -> dict:
    import pandas as pd
    y = df["in_sec"].cast(pl.Float64).to_numpy()
    X = df.select(xcols).cast(pl.Float64).to_numpy()
    cells = df["cell"].to_numpy()
    pdf = pd.DataFrame(X, columns=xcols)
    pdf["y"] = y
    pdf["cell"] = cells
    dem = pdf.groupby("cell")[xcols + ["y"]].transform("mean")
    Xd = (pdf[xcols] - dem[xcols]).to_numpy()
    yd = (pdf["y"] - dem["y"]).to_numpy()
    XtX = Xd.T @ Xd
    b = np.linalg.solve(XtX, Xd.T @ yd)
    e = yd - Xd @ b
    # HC1
    meat = (Xd * (e**2)[:, None]).T @ Xd
    n, k = Xd.shape
    XtX_inv = np.linalg.inv(XtX)
    V = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    out = {"label": label, "n": int(n),
           "p_sec": float(df["in_sec"].cast(pl.Float64).mean()),
           "coef": {c: {"b": float(bi), "se": float(si),
                        "t": float(bi / si)}
                    for c, bi, si in zip(xcols, b, se)}}
    disp = {c: f"{v['b']*100:+.3f}pp(t={v['t']:.1f})"
            for c, v in out["coef"].items()}
    print(f"  {label}: n={n:,} " + " ".join(f"{c}={s}" for c, s in disp.items()),
          file=sys.stderr, flush=True)
    return out


def main() -> int:
    df = build()
    reg = df.filter(pl.col("registered")).with_columns(
        pl.format("{}_{}", pl.col("cls"), pl.col("fy")).alias("cell"),
        (pl.col("gs_len").cast(pl.Float64) + 1).log().alias("log_len"),
        # Class x year cells thinner than 100 cannot support five quintiles and
        # are absorbed to nothing by the FE, so they are dropped.
    ).filter(pl.len().over("cell") >= 100)

    # Within-cell quintiles and z-scores. rank("ordinal") resolves ties by frame
    # row order, which is not stable across runs, and roughly 27% of scored
    # filings are tied (boilerplate goods/services text produces identical topic
    # distributions). Sort on the score then a unique key so the cut points are
    # reproducible.
    # Atypicality is the AVERAGE of the two levels, not the past level alone.
    # The paper's headline is estimated on this column; the past-level spec is
    # retained below as the robustness comparison (the two correlate at 0.979).
    reg = reg.with_columns(
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")))
        .alias("topic_atyp"))

    for col, tag in (("topic_kl_vs_past", "p"), ("topic_kl_vs_future", "r"),
                     ("topic_dkl", "d"), ("topic_atyp", "a")):
        reg = reg.sort(["cell", col, "owner_name"])
        reg = reg.with_columns(
            ((pl.col(col).rank("ordinal").over("cell") - 1) * 5
             // pl.len().over("cell")).cast(pl.Int8).alias(f"q_{tag}"),
            ((pl.col(col) - pl.col(col).mean().over("cell"))
             / pl.col(col).std().over("cell")).alias(f"z_{tag}"),
        )
    for tag in ("p", "d", "a"):
        for qq in range(1, 5):
            reg = reg.with_columns(
                (pl.col(f"q_{tag}") == qq).cast(pl.Float64).alias(f"{tag}q{qq+1}"))
    reg = reg.with_columns(
        pl.col("z_d").abs().alias("abs_zd"),
        (pl.col("z_d").sign() * pl.col("z_d").abs()).alias("signed_zd"),  # = z_d
    )
    print(f"[registered debuts] {reg.height:,} base p_sec="
          f"{float(reg['in_sec'].cast(pl.Float64).mean())*100:.3f}%",
          file=sys.stderr, flush=True)

    res = {"n_registered_debuts": reg.height, "specs": []}
    pq = ["pq2", "pq3", "pq4", "pq5"]
    dq = ["dq2", "dq3", "dq4", "dq5"]
    aq = ["aq2", "aq3", "aq4", "aq5"]
    res["specs"].append(lpm_fe(reg, aq, "A1_atyp_quintiles_FE"))
    res["specs"].append(lpm_fe(reg, aq + ["log_len"], "A2_atyp_len"))
    res["specs"].append(lpm_fe(reg, pq, "A1_prosKL_quintiles_FE"))
    res["specs"].append(lpm_fe(reg, pq + ["log_len"], "A2_prosKL_len"))
    res["specs"].append(lpm_fe(reg, dq, "B1_dKL_quintiles_FE"))
    res["specs"].append(lpm_fe(reg, dq + ["log_len"], "B2_dKL_len"))
    res["specs"].append(lpm_fe(reg, dq + ["log_len", "z_p", "z_r"],
                               "B3_dKL_len_levels"))
    res["specs"].append(lpm_fe(reg, ["abs_zd", "z_d", "log_len"],
                               "C_absdKL_sign_len"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "debut_edgar_substantiate.json").write_text(json.dumps(res, indent=1))
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
