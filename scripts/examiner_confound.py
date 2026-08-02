"""Bound the examiner-selection channel.

Two pieces:
1. Failure-mode decomposition of non-registered applications, all classes
   pooled (filings 1995-2018): what share of registration failure is
   adversarial/examiner-driven vs applicant abandonment?
2. The direct test: P(owner ever in SEC EDGAR) among ALL debut filings,
   unconditional on registration, by dKL quintile. If examiner removal of
   weak firms created the firm-level gradient, it should vanish here.

Output: paper/results/examiner_confound.json
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

# 6xx abandonment vs adversarial decomposition
NO_SOU = {"606", "607"}            # never filed / defective statement of use
NO_RESPONSE = {"600", "602"}       # incomplete / failure to respond
EXPRESS = {"601"}                  # express abandonment by applicant
ADVERSARIAL = {"603", "604", "605"}  # ex parte appeal / inter partes / after publication


def all_classes() -> list[str]:
    return sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def main() -> int:
    # ---------- piece 1: failure-mode decomposition ----------
    counts = {"no_sou": 0, "no_response": 0, "express": 0,
              "adversarial": 0, "other": 0, "total": 0}
    for cls in all_classes():
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["registration_date", "status_code", "filing_date"],
        ).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8
             ).alias("registered"),
        ).filter(~pl.col("registered") & pl.col("year").is_between(1995, 2018))
        sc = tm["status_code"]
        counts["no_sou"] += int(sc.is_in(list(NO_SOU)).sum())
        counts["no_response"] += int(sc.is_in(list(NO_RESPONSE)).sum())
        counts["express"] += int(sc.is_in(list(EXPRESS)).sum())
        counts["adversarial"] += int(sc.is_in(list(ADVERSARIAL)).sum())
        counts["total"] += tm.height
        del tm
        gc.collect()
    counts["other"] = counts["total"] - sum(
        counts[k] for k in ("no_sou", "no_response", "express", "adversarial"))
    shares = {k: counts[k] / counts["total"] for k in counts if k != "total"}
    print("[failure modes]", {k: round(v, 4) for k, v in shares.items()},
          file=sys.stderr, flush=True)

    # ---------- piece 2: unconditional EDGAR gradient on debut filings ----------
    debut_parts = []
    for cls in all_classes():
        d = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet", columns=["owner_name", "filing_date"]
        ).filter(pl.col("owner_name").is_not_null()
                 & (pl.col("filing_date").str.len_chars() == 8)
        ).group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        debut_parts.append(d)
        del d
        gc.collect()
    debut = pl.concat(debut_parts).group_by("owner_name").agg(
        pl.col("debut_date").min().alias("debut_date"))
    del debut_parts
    gc.collect()

    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))

    parts = []
    for cls in all_classes():
        sp = PROC / f"surprise_class{cls}.parquet"
        if not sp.exists():
            continue
        s = pl.read_parquet(
            sp, columns=["serial_number", "year", "kl_vs_past",
                         "kl_vs_future", "n_ref_past",
                         "n_ref_future", "n_terms"],
        ).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
            & pl.col("year").is_between(1995, 2018))
        t = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "owner_name", "filing_date", "registration_date"],
        ).with_columns(
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8
             ).alias("registered"),
        ).join(debut, on="owner_name", how="left").filter(
            pl.col("filing_date") == pl.col("debut_date"))
        j = s.join(t, on="serial_number", how="inner").join(
            sec, on="owner_name", how="left").with_columns(
            pl.col("in_sec").fill_null(False),
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        ).select("dkl", "registered", "in_sec")
        if j.height:
            parts.append(j)
        del s, t, j
        gc.collect()
    df = pl.concat(parts)
    del parts
    gc.collect()

    def quintiles(frame: pl.DataFrame, outcome: str) -> dict:
        arr = frame["dkl"].to_numpy()
        cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(cuts, arr)
        g = frame.with_columns(pl.Series("q", qi)).group_by("q").agg(
            pl.len().alias("n"),
            pl.col(outcome).cast(pl.Float64).mean().alias("rate")).sort("q")
        d = {f"q{int(r['q'])+1}": r["rate"] for r in g.iter_rows(named=True)}
        d["lift_q5_q1"] = d["q5"] - d["q1"]
        return d

    reg = df.filter(pl.col("registered"))
    out = {
        "failure_modes": {"counts": counts, "shares": shares},
        "n_debut": df.height,
        "edgar_unconditional": quintiles(df, "in_sec"),
        "edgar_given_registration": quintiles(reg, "in_sec"),
        "p_edgar_uncond": float(df["in_sec"].cast(pl.Float64).mean()),
        "p_edgar_reg": float(reg["in_sec"].cast(pl.Float64).mean()),
    }
    (RES / "examiner_confound.json").write_text(json.dumps(out, indent=1, default=float))
    print("[edgar uncond]", {k: round(v, 5) for k, v in out["edgar_unconditional"].items()},
          file=sys.stderr, flush=True)
    print("[edgar | reg ]", {k: round(v, 5) for k, v in out["edgar_given_registration"].items()},
          file=sys.stderr, flush=True)
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
