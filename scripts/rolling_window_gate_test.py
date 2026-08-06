"""Does the gate result survive a per-filing reference window?

Re-runs the first-gate contrast on class 009 under both scorings on an
identical sample: the production per-calendar-year reference, and the
per-filing rolling reference built by rolling_window_rescore.py. Same
registrations, same cohort fixed effects, same owner clustering, same quintile
construction -- only the reference window differs.

Output: paper/results/rolling_window_gate_test.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

CLS = "009"
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
SUFFIX = re.compile(r"\b(INC|LLC|LTD|CORP|CORPORATION|COMPANY|CO|LP|LLP|PLC|GMBH|SA|NV|BV|AB|AG|PTY|PTE|KK|SRL|SPA)\b")


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def norm_owner(s: pl.Series) -> pl.Series:
    return (s.str.to_uppercase().str.replace_all(r"[^A-Z0-9 ]", " ")
            .str.replace_all(SUFFIX.pattern, "").str.replace_all(r"\s+", " ")
            .str.strip_chars())


def lpm_cluster(d: pd.DataFrame, y: str, xs: list[str], cell: str, grp: str) -> dict:
    d = d[[y, cell, grp] + xs].dropna()
    g = d.groupby(cell)
    yd = (d[y] - g[y].transform("mean")).to_numpy()
    X = np.column_stack([(d[x] - g[x].transform("mean")).to_numpy() for x in xs])
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ yd)
    e = yd - X @ b
    df_ = pd.DataFrame(X * e[:, None]); df_["g"] = d[grp].to_numpy()
    S = df_.groupby("g").sum().to_numpy()
    G, (n, k) = S.shape[0], X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    Xi = np.linalg.inv(XtX)
    V = adj * Xi @ (S.T @ S) @ Xi
    se = np.sqrt(np.diag(V))
    return {"n": int(n), "n_owners": int(G), "base": float(d[y].mean()),
            "coef": {x: {"b_pp": float(bi) * 100, "se_pp": float(si) * 100,
                         "t": float(bi / si)} for x, bi, si in zip(xs, b, se)}}


def main() -> int:
    tm = pl.read_parquet(PROC / f"tm_class{CLS}.parquet",
                         columns=["serial_number", "registration_date",
                                  "owner_name", "goods_services"]).filter(
        pl.col("registration_date").fill_null("").str.len_chars() >= 8)
    tm = tm.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("reg_d"),
        pl.col("goods_services").str.len_chars().alias("gs_len"),
    ).drop_nulls("reg_d").with_columns(
        pl.col("reg_d").dt.year().alias("reg_year")).filter(
        pl.col("reg_year").is_between(REG_LO, REG_HI))
    tm = tm.with_columns(norm_owner(pl.col("owner_name")).alias("own"))
    log(f"[reg] {tm.height:,} class-{CLS} registrations {REG_LO}-{REG_HI}")

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("c8_d")
    ).drop_nulls("c8_d").group_by("serial_number").agg(pl.col("c8_d").min())

    tm = tm.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("c8_d") - pl.col("reg_d")).dt.total_days() / 365.25).alias("age")
    ).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed1"))

    yr = pl.read_parquet(PROC / f"topic_surprise_class{CLS}_T200.parquet",
                         columns=["serial_number", "topic_dkl"])
    ro = pl.read_parquet(PROC / f"rolling_surprise_class{CLS}_T200.parquet",
                         columns=["serial_number", "roll_dkl"])
    d = tm.join(yr, on="serial_number", how="inner").join(
        ro, on="serial_number", how="inner").filter(
        pl.col("topic_dkl").is_finite() & pl.col("roll_dkl").is_finite())
    log(f"[sample] {d.height:,} registrations scored BOTH ways, "
        f"{d['own'].n_unique():,} owners")

    # within-cohort quintiles under each scoring, deterministic tie order
    for col, tag in (("topic_dkl", "y"), ("roll_dkl", "r")):
        d = d.sort(["reg_year", col, "serial_number"]).with_columns(
            ((pl.col(col).rank("ordinal").over("reg_year") - 1) * 5
             // pl.len().over("reg_year")).cast(pl.Int8).alias(f"q_{tag}"),
            ((pl.col(col) - pl.col(col).mean().over("reg_year"))
             / pl.col(col).std().over("reg_year")).alias(f"z_{tag}"))
    for tag in ("y", "r"):
        for q in range(1, 5):
            d = d.with_columns(
                (pl.col(f"q_{tag}") == q).cast(pl.Float64).alias(f"{tag}q{q+1}"))

    pdf = d.with_columns(
        (pl.col("gs_len").cast(pl.Float64) + 1).log().alias("log_len"),
        pl.col("reg_year").cast(pl.Utf8).alias("cell")).to_pandas()

    out = {"class": CLS, "reg_window": [REG_LO, REG_HI], "n": int(len(pdf)),
           "specs": {}}
    for tag, name in (("y", "per_calendar_year"), ("r", "per_filing_rolling")):
        qs = [f"{tag}q{q}" for q in range(2, 6)]
        out["specs"][name + "_quintiles"] = lpm_cluster(
            pdf, "failed1", qs + ["log_len"], "cell", "own")
        out["specs"][name + "_continuous"] = lpm_cluster(
            pdf, "failed1", [f"z_{tag}", "log_len"], "cell", "own")

    for k, v in out["specs"].items():
        key = [c for c in v["coef"] if c.endswith("q5") or c.startswith("z_")][0]
        c = v["coef"][key]
        log(f"  {k:34s} {key:6s} {c['b_pp']:+.3f}pp (se {c['se_pp']:.3f}, "
            f"t {c['t']:+.1f})  base={v['base']*100:.1f}%  n={v['n']:,}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "rolling_window_gate_test.json").write_text(json.dumps(out, indent=1))
    log("[done] rolling_window_gate_test.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
