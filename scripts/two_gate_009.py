"""The software two-gate diagnostic: does the penalty attenuate after year six?

Rebuilt because the original producer of paper/results/two_gate_009.json is no
longer in the tree and its numbers were computed on the retired per-calendar-year
scoring. Class 009, registrations 2010-2013 (both gates elapsed).

  gate 1: a dated cancellation for non-use at registration age 4.0-8.5 years,
          under Section 8 or, for Madrid 66(a) registrations, Section 71
  gate 2: among gate-1 survivors, whether the registration was renewed (status
          800) or died (710 / 900) -- the two gates share a terminal status and
          are separable only this way

The cohort is narrow by necessity: both gates must have elapsed, so it starts
after the year-ten window opens for 2010 registrations and ends where the
first-gate window closes. Class 009 is the diagnostic class throughout the
paper because it is the largest and the deepest work was done there; the
attenuation reported here is a single-class result, not a corpus estimate.

Contrasts are raw quintiles cut on the whole class with no fixed effects and
no owner clustering -- this is a shape diagnostic, not an estimate. The paper's
gate estimate comes from gate_decisive_regression.py.

Honours SURPRISE_SRC=rolling like the other gate scripts.

Reads   data/processed/tm_class009.parquet
        data/processed/case_events.parquet
        data/processed/{SRC}_surprise_class009.parquet
Output: paper/results/two_gate_009.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

# Default is "rolling": the per-filing reference windows every estimate in
# the paper uses. Set SURPRISE_SRC=topic to reproduce the retired
# per-calendar-year scoring, which the comparison appendix reports.
SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLS = "009"
LO, HI = 2010, 2013
GATE_LO, GATE_HI = 4.0, 8.5


def quint(df: pl.DataFrame, score: str, outcome: str) -> dict:
    d = df.drop_nulls([score, outcome]).filter(pl.col(score).is_finite())
    # Sort on the score then the serial number before ranking: rank("ordinal")
    # otherwise breaks ties by frame row order, which polars does not guarantee
    # across runs, and roughly a quarter of scored filings are tied because
    # boilerplate text yields identical topic distributions.
    d = d.sort([score, "serial_number"]).with_columns(
        ((pl.col(score).rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
    g = d.group_by("q").agg(pl.col(outcome).mean().alias("p")).sort("q")
    out = {f"q{int(r['q'])+1}": float(r["p"]) for r in g.to_dicts()}
    out["lift_q5_q1"] = out.get("q5", float("nan")) - out.get("q1", float("nan"))
    out["n"] = int(d.height)
    return out


def main() -> int:
    tm = pl.read_parquet(PROC / f"tm_class{CLS}.parquet",
                         columns=["serial_number", "registration_date", "status_code"]
                         ).filter(pl.col("registration_date").fill_null("").str.len_chars() >= 8)
    tm = tm.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("reg_d")
    ).drop_nulls("reg_d").with_columns(pl.col("reg_d").dt.year().alias("ry")
                                       ).filter(pl.col("ry").is_between(LO, HI))

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())

    tm = tm.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("reg_d")).dt.total_days() / 365.25).alias("age")
    ).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed1"))

    sc = pl.read_parquet(PROC / f"{SRC}_surprise_class{CLS}.parquet",
                         columns=["serial_number", "topic_dkl"])
    d = tm.join(sc, on="serial_number", how="inner")

    # Gate 2 is conditional on having cleared gate 1, and is only identified
    # for registrations whose terminal status has resolved: 800 renewed, 710
    # cancelled, 900 expired. Anything else is still mid-life and is dropped
    # rather than coded as a survivor.
    g2 = d.filter((pl.col("failed1") == 0.0)
                  & pl.col("status_code").is_in(["800", "710", "900"])).with_columns(
        pl.col("status_code").is_in(["710", "900"]).cast(pl.Float64).alias("failed2"))

    out = {"scoring": SRC, "cohort": f"class{CLS} reg {LO}-{HI}",
           "n_reg": int(d.height),
           "p_fail1": float(d["failed1"].mean()),
           "n_gate2_resolved": int(g2.height),
           "p_fail2": float(g2["failed2"].mean()),
           "fail1_by_topic_dkl": quint(d, "topic_dkl", "failed1"),
           "fail2_by_topic_dkl": quint(g2, "topic_dkl", "failed2")}

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "two_gate_009.json").write_text(json.dumps(out, indent=1))
    print(f"[{SRC}] n_reg={out['n_reg']:,} p_fail1={out['p_fail1']*100:.1f}% "
          f"gate1 lift={out['fail1_by_topic_dkl']['lift_q5_q1']*100:+.1f}pp | "
          f"n2={out['n_gate2_resolved']:,} p_fail2={out['p_fail2']*100:.1f}% "
          f"gate2 lift={out['fail2_by_topic_dkl']['lift_q5_q1']*100:+.1f}pp",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
