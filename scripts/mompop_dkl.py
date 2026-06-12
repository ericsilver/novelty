"""The mom-and-pop question: among owners who are NOT on a counsel/venture
trajectory, what does signed dKL predict for survival?

Strata (proxying 'credible counsel early' without the attorney field):
  owner scale : small = owner's lifetime corpus filings <= 2
                large = >= 5
  distinctness: low / high = below / above the pooled median of the
                topic prospective-KL level (length-robust atypicality)

Within each cell, signed dKL (topic and token) quintile gradients on:
  - first Section 8 gate survival (registrations 2016-2018)
  - registration completion (filings 1995-2018)
plus EDGAR base rates per cell for context.

Output: paper/results/mompop_dkl.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

PASS = {"701", "702", "703", "704", "705", "800"}
FAIL = {"710"}


def all_classes() -> list[str]:
    return sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def owner_counts() -> pl.DataFrame:
    parts = []
    for cls in all_classes():
        d = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                            columns=["owner_name"]).filter(
            pl.col("owner_name").is_not_null())
        parts.append(d.group_by("owner_name").len().rename({"len": "n"}))
        del d
        gc.collect()
    return pl.concat(parts).group_by("owner_name").agg(
        pl.col("n").sum().alias("lifetime_filings"))


def main() -> int:
    oc = owner_counts()
    print(f"[owners] {oc.height:,}; share <=2 filings: "
          f"{float((oc['lifetime_filings'] <= 2).mean()):.3f}",
          file=sys.stderr, flush=True)
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))

    parts = []
    for cls in all_classes():
        tp = PROC / f"topic_surprise_class{cls}.parquet"
        sp = PROC / f"surprise_class{cls}.parquet"
        if not tp.exists():
            continue
        topic = pl.read_parquet(tp).filter(pl.col("topic_dkl").is_finite())
        tok = pl.read_parquet(
            sp, columns=["serial_number", "n_terms", "prospective_kl",
                         "retrospective_kl", "n_ref_prospective",
                         "n_ref_retrospective"],
        ).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
        ).with_columns(
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("token_dkl"))
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "owner_name", "registration_date",
                     "status_code"],
        ).with_columns(
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8
             ).alias("registered"),
            pl.col("registration_date").str.slice(0, 4)
              .cast(pl.Int32, strict=False).alias("reg_year"),
            pl.col("status_code").is_in(list(PASS)).alias("st_pass"),
            pl.col("status_code").is_in(list(FAIL)).alias("st_fail"),
        )
        j = topic.join(tok.select("serial_number", "token_dkl"),
                       on="serial_number", how="inner").join(
            tm, on="serial_number", how="inner").join(
            oc, on="owner_name", how="left").join(
            sec, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False),
            pl.col("lifetime_filings").fill_null(1),
        ).select("year", "topic_pros", "topic_dkl", "token_dkl",
                 "registered", "reg_year", "st_pass", "st_fail",
                 "lifetime_filings", "in_sec")
        parts.append(j)
        del topic, tok, tm, j
        gc.collect()
    df = pl.concat(parts)
    del parts
    gc.collect()
    print(f"[pool] {df.height:,} filings", file=sys.stderr, flush=True)

    med_pros = float(df["topic_pros"].median())
    df = df.with_columns(
        (pl.col("topic_pros") > med_pros).alias("high_distinct"),
        (pl.col("lifetime_filings") <= 2).alias("small_owner"),
        (pl.col("lifetime_filings") >= 5).alias("large_owner"),
    )

    def quintiles(frame, var, outcome):
        arr = frame[var].to_numpy()
        cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(cuts, arr)
        y = frame[outcome].cast(pl.Float64).to_numpy()
        out = {}
        for q in range(5):
            m = qi == q
            out[f"q{q+1}"] = float(y[m].mean()) if m.any() else None
            out[f"q{q+1}_n"] = int(m.sum())
        out["lift_q5_q1"] = out["q5"] - out["q1"]
        p1, n1, p5, n5 = out["q1"], out["q1_n"], out["q5"], out["q5_n"]
        out["lift_se"] = float(np.sqrt(p1*(1-p1)/max(n1,1) + p5*(1-p5)/max(n5,1)))
        return out

    cells = {
        "mompop_lowdist": (pl.col("small_owner") & ~pl.col("high_distinct")),
        "mompop_highdist": (pl.col("small_owner") & pl.col("high_distinct")),
        "large_lowdist": (pl.col("large_owner") & ~pl.col("high_distinct")),
        "large_highdist": (pl.col("large_owner") & pl.col("high_distinct")),
    }
    out = {"median_topic_pros": med_pros, "cells": {}}
    for name, cond in cells.items():
        sub = df.filter(cond)
        gate = sub.filter(pl.col("reg_year").is_between(2016, 2018)
                          & (pl.col("st_pass") | pl.col("st_fail")))
        regp = sub.filter(pl.col("year").is_between(1995, 2018))
        cell = {
            "n_filings": sub.height,
            "n_gated": gate.height,
            "p_edgar": float(sub["in_sec"].cast(pl.Float64).mean()),
            "p_registered": float(regp["registered"].cast(pl.Float64).mean()),
            "p_gate_pass": float(gate["st_pass"].cast(pl.Float64).mean())
            if gate.height else None,
        }
        if gate.height >= 5000:
            cell["gate_by_topic_dkl"] = quintiles(gate, "topic_dkl", "st_pass")
            cell["gate_by_token_dkl"] = quintiles(gate, "token_dkl", "st_pass")
        if regp.height >= 5000:
            cell["reg_by_topic_dkl"] = quintiles(regp, "topic_dkl", "registered")
        out["cells"][name] = cell
        g = cell.get("gate_by_topic_dkl", {})
        print(f"[{name}] n={sub.height:,} gated={gate.height:,} "
              f"P(reg)={cell['p_registered']:.3f} P(pass)={cell['p_gate_pass']} "
              f"P(edgar)={cell['p_edgar']:.5f} "
              f"gate_topic_lift={g.get('lift_q5_q1')}",
              file=sys.stderr, flush=True)

    (RES / "mompop_dkl.json").write_text(json.dumps(out, indent=1, default=float))
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
