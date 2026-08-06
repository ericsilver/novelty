"""The composition-vs-construct decider (adversarial-panel Tier 1, item 1).

Gate failure ~ topic-dKL, estimated properly for the first time:
  - unique serials (multi-class registrations deduped to one row)
  - dKL quintiles cut WITHIN class x cohort cells (kills pooled-composition mixing)
  - LPM with class x cohort FE (absorbed by demeaning), covariates:
      counsel, ITU basis, log goods/services length, log owner filing count,
      owner domicile (US / CN / other-unknown), foreign basis (44e / 66a)
  - SEs clustered on normalized owner name
Specs: (1) FE only; (2) FE + controls; (3) counsel-represented only;
       (4) counsel + US-domiciled only; (5) self-filed only;
       (6) cohort thirds within counsel+US (does the "modern" penalty survive
           in the clean stratum?)

Output: paper/results/gate_decisive_regression.json
"""
from __future__ import annotations

import gc
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
OWNER_ADDR = REPO / "_archive_mac_syndication_2026-05-30" / "data-extras" / "owner_address.parquet"

REG_LO, REG_HI = 2002, 2018
CLASSES = [f"{i:03d}" for i in range(1, 46)]

SUFFIX_RE = re.compile(
    r"\b(INC|INCORPORATED|LLC|L L C|CORP|CORPORATION|LTD|LIMITED|CO|COMPANY|"
    r"GMBH|S A|SA|AG|B V|BV|PTY|PLC|LP|LLP|KK|K K|CO LTD)\b\.?")


def norm_owner(col: pl.Expr) -> pl.Expr:
    c = col.fill_null("").str.to_uppercase()
    c = c.str.replace_all(r"[^A-Z0-9 ]", " ")
    c = c.str.replace_all(SUFFIX_RE.pattern, " ")
    return c.str.replace_all(r"\s+", " ").str.strip_chars()


def build_frame() -> pl.DataFrame:
    # The first maintenance gate is Section 8 for domestic registrations and
    # Section 71 for Madrid Protocol Section 66(a) extensions of protection.
    # Both fall due between the fifth and sixth anniversary of registration, so
    # they are the same gate under two statutes. Reading only C8.. scores every
    # 66(a) registration as a non-failure whatever became of it, which is what
    # produced the -46pp basis_66a coefficient; C71T is the corresponding
    # cancellation code and is already in case_events.
    print("[load] gate cancellation events (C8.. and C71T)", file=sys.stderr, flush=True)
    c8 = pl.scan_parquet(PROC / "case_events.parquet").filter(
        (pl.col("code").is_in(["C8..", "C71T"])) & (pl.col("date") > 19000000)).select(
        "serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("c8_d")).drop_nulls("c8_d").group_by("serial_number").agg(
        pl.col("c8_d").min())

    print("[load] extras", file=sys.stderr, flush=True)
    extras = pl.scan_parquet(PROC / "case_extras.parquet").select(
        "serial_number", "attorney_name", "intent_to_use_in",
        "filing_basis_current_44e_in", "filing_basis_current_66a_in").collect(
    ).with_columns(
        (pl.col("attorney_name").fill_null("").str.len_chars() > 0).alias("has_attorney"),
        (pl.col("intent_to_use_in") == "T").alias("itu"),
        (pl.col("filing_basis_current_44e_in") == "T").alias("basis_44e"),
        (pl.col("filing_basis_current_66a_in") == "T").alias("basis_66a"),
    ).select("serial_number", "has_attorney", "itu", "basis_44e", "basis_66a")

    print("[load] owner_address", file=sys.stderr, flush=True)
    addr = pl.scan_parquet(OWNER_ADDR).select(
        "serial_number", "owner_country").collect().unique(
        subset="serial_number", keep="first")

    parts = []
    for cls in CLASSES:
        tp = PROC / f"topic_surprise_class{cls}.parquet"
        if not tp.exists():
            continue
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "registration_date", "owner_name", "goods_services"],
        ).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8
        ).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
            .alias("reg_d"),
            pl.col("registration_date").str.slice(0, 4)
            .cast(pl.Int32, strict=False).alias("reg_year"),
            pl.col("goods_services").fill_null("").str.len_chars().alias("gs_len"),
        ).filter(
            pl.col("reg_year").is_between(REG_LO, REG_HI)
        ).drop_nulls("reg_d").select(
            "serial_number", "reg_d", "reg_year", "owner_name", "gs_len")
        topic = pl.read_parquet(tp).filter(
            pl.col("topic_dkl").is_finite()).select("serial_number", "topic_dkl")
        j = tm.join(topic, on="serial_number", how="inner").with_columns(
            pl.lit(cls).alias("cls"))
        parts.append(j)
        del tm, topic
        gc.collect()
    df = pl.concat(parts)
    del parts
    gc.collect()
    n_rows = df.height

    # unique serials: keep first class occurrence (sorted -> lowest class code)
    df = df.sort("cls").unique(subset="serial_number", keep="first")
    print(f"[dedupe] {n_rows:,} class-rows -> {df.height:,} unique serials",
          file=sys.stderr, flush=True)

    df = df.join(c8, on="serial_number", how="left").join(
        extras, on="serial_number", how="left").join(
        addr, on="serial_number", how="left").with_columns(
        pl.col("has_attorney").fill_null(False),
        pl.col("itu").fill_null(False),
        pl.col("basis_44e").fill_null(False),
        pl.col("basis_66a").fill_null(False),
        ((pl.col("c8_d") - pl.col("reg_d")).dt.total_days() / 365.25).alias("c8_age"),
        norm_owner(pl.col("owner_name")).alias("owner_key"),
        pl.col("owner_country").fill_null("").str.strip_chars().alias("ctry"),
    ).with_columns(
        ((pl.col("c8_age") >= 4.0) & (pl.col("c8_age") < 8.5))
        .fill_null(False).alias("failed1"),
        (pl.col("ctry") == "US").alias("dom_us"),
        (pl.col("ctry") == "CN").alias("dom_cn"),
        (pl.format("{}_{}", pl.col("cls"), pl.col("reg_year"))).alias("cell"),
    )
    # owner filing count within frame
    df = df.with_columns(pl.len().over("owner_key").alias("owner_n"))
    # Quintiles within class x cohort.
    # rank("ordinal") breaks ties by the frame's row order, which polars does not
    # guarantee across runs; and ties are not rare here -- roughly 27% of scored
    # filings share a topic_dkl value with another filing, because boilerplate
    # goods/services text yields identical topic distributions (the largest tie
    # groups run to five figures). Sorting on the score and then on the serial
    # number makes the tie-break deterministic and reproducible.
    df = df.sort(["cell", "topic_dkl", "serial_number"])
    _rank = os.environ.get("TIE_RULE", "ordinal")   # "min" keeps ties together
    df = df.with_columns(
        ((pl.col("topic_dkl").rank(_rank).over("cell") - 1) * 5
         // pl.len().over("cell")).cast(pl.Int8).alias("q"),
        ((pl.col("topic_dkl") - pl.col("topic_dkl").mean().over("cell"))
         / pl.col("topic_dkl").std().over("cell")).alias("z"),
        (pl.col("gs_len").cast(pl.Float64) + 1).log().alias("log_len"),
        (pl.col("owner_n").cast(pl.Float64)).log1p().alias("log_owner_n"),
    ).filter(pl.len().over("cell") >= 100)
    return df


def lpm_fe_cluster(df: pl.DataFrame, xcols: list[str], label: str) -> dict:
    """LPM of failed1 on xcols, class x cohort FE absorbed by demeaning,
    SEs clustered on owner_key."""
    y = df["failed1"].cast(pl.Float64).to_numpy()
    X = df.select(xcols).cast(pl.Float64).to_numpy()
    cells = df["cell"].to_numpy()
    owners = df["owner_key"].to_numpy()

    import pandas as pd
    pdf = pd.DataFrame(X, columns=xcols)
    pdf["y"] = y
    pdf["cell"] = cells
    dem = pdf.groupby("cell")[xcols + ["y"]].transform("mean")
    Xd = (pdf[xcols] - dem[xcols]).to_numpy()
    yd = (pdf["y"] - dem["y"]).to_numpy()

    XtX = Xd.T @ Xd
    b = np.linalg.solve(XtX, Xd.T @ yd)
    e = yd - Xd @ b

    # clustered meat
    Xe = pd.DataFrame(Xd * e[:, None])
    Xe["g"] = owners
    S = Xe.groupby("g").sum().to_numpy()
    meat = S.T @ S
    G = S.shape[0]
    n, k = Xd.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    XtX_inv = np.linalg.inv(XtX)
    V = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    out = {"label": label, "n": int(n), "n_owners": int(G),
           "p_fail1": float(df["failed1"].cast(pl.Float64).mean()),
           "coef": {c: {"b": float(bi), "se": float(si), "t": float(bi / si)}
                    for c, bi, si in zip(xcols, b, se)}}
    q5 = out["coef"].get("q5")
    print(f"  {label}: n={n:,} owners={G:,} base={out['p_fail1']:.3f}"
          + (f"  Q5-Q1={q5['b']*100:+.2f}pp (t={q5['t']:.1f})" if q5 else ""),
          file=sys.stderr, flush=True)
    return out


def main() -> int:
    df = build_frame()
    # quintile dummies (Q1 omitted)
    for qq in range(1, 5):
        df = df.with_columns((pl.col("q") == qq).cast(pl.Float64).alias(f"q{qq+1}"))
    for bcol in ("has_attorney", "itu", "basis_44e", "basis_66a", "dom_us", "dom_cn"):
        df = df.with_columns(pl.col(bcol).cast(pl.Float64))

    qcols = ["q2", "q3", "q4", "q5"]
    controls = ["has_attorney", "itu", "log_len", "log_owner_n",
                "dom_us", "dom_cn", "basis_44e", "basis_66a"]

    print("\n=== country mix (top) ===", file=sys.stderr)
    print(df.group_by("ctry").len().sort("len", descending=True).head(8),
          file=sys.stderr, flush=True)

    results = {"reg_window": [REG_LO, REG_HI], "specs": []}
    print("\n=== SPECS ===", file=sys.stderr, flush=True)
    results["specs"].append(lpm_fe_cluster(df, qcols, "1_FE_only"))
    results["specs"].append(lpm_fe_cluster(df, qcols + controls, "2_FE_controls"))

    couns = df.filter(pl.col("has_attorney") == 1.0)
    results["specs"].append(lpm_fe_cluster(
        couns, qcols + ["itu", "log_len", "log_owner_n", "dom_us", "dom_cn",
                        "basis_44e", "basis_66a"], "3_counsel_only"))
    cus = couns.filter(pl.col("dom_us") == 1.0)
    results["specs"].append(lpm_fe_cluster(
        cus, qcols + ["itu", "log_len", "log_owner_n"], "4_counsel_US_only"))
    selff = df.filter(pl.col("has_attorney") == 0.0)
    results["specs"].append(lpm_fe_cluster(
        selff, qcols + ["itu", "log_len", "log_owner_n", "dom_us", "dom_cn",
                        "basis_44e", "basis_66a"], "5_self_filed_only"))

    # 6: cohort thirds within counsel+US
    for lo, hi in ((2002, 2007), (2008, 2012), (2013, 2018)):
        sub = cus.filter(pl.col("reg_year").is_between(lo, hi))
        results["specs"].append(lpm_fe_cluster(
            sub, qcols + ["itu", "log_len", "log_owner_n"],
            f"6_counselUS_{lo}_{hi}"))

    # 9: drop Madrid 66(a) outright rather than absorbing it in a dummy.
    # 66(a) registrations file Section 71 declarations, not Section 8, so they
    # cannot generate the C8.. event this outcome is built from and enter the
    # sample as mechanical non-failures. The C71T cancellations ARE in
    # case_events (118,838 of them) but no analysis reads them; until they do,
    # the honest robustness check is to estimate without these records.
    no66 = df.filter(pl.col("basis_66a") == 0.0)
    results["specs"].append(lpm_fe_cluster(
        no66, qcols + ["itu", "log_len", "log_owner_n", "dom_us", "dom_cn",
                       "basis_44e"], "9_excl_madrid_66a"))

    # continuous-z variants of specs 2 and 4
    results["specs"].append(lpm_fe_cluster(df, ["z"] + controls, "7_z_FE_controls"))
    results["specs"].append(lpm_fe_cluster(
        cus, ["z", "itu", "log_len", "log_owner_n"], "8_z_counsel_US"))

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_decisive_regression.json").write_text(
        json.dumps(results, indent=1))
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
