"""Persist the numbers the audit found untraceable to any artifact.

The v3 draft quotes several quantities computed ad hoc in sessions and never
written to paper/results. This computes and persists them so every number in
the representation subsection and the clustering note traces to a file.

  1. Counsel share of filings by year, 1995-2018, and overall.
  2. Registration rate and event-dated first-gate failure by counsel status,
     unique serials (registration: filings 1995-2018; gate: regs 2002-2018).
  3. Median distinct-term counts by counsel status, classes 009 and 035
     (from surprise_class n_terms).
  4. Pooled within-owner correlation of gate failure on the full gate sample
     (one-way ANOVA ICC on owners with >= 2 registrations).

Output: paper/results/representation_stats.json
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
CLASSES = [f"{i:03d}" for i in range(1, 46)]
GATE_LO, GATE_HI = 4.0, 8.5


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    att = pl.read_parquet(PROC / "case_extras.parquet", columns=["serial_number", "attorney_name"]).with_columns(
        (pl.col("attorney_name").fill_null("").str.len_chars() > 0).alias("counsel")).select("serial_number", "counsel")
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    parts = []
    for c in CLASSES:
        p = PROC / f"tm_class{c}.parquet"
        if not p.exists():
            continue
        parts.append(pl.read_parquet(p, columns=["serial_number", "filing_date", "registration_date", "owner_name"]).filter(
            pl.col("filing_date").fill_null("").str.len_chars() >= 8))
    d = pl.concat(parts).unique("serial_number"); del parts; gc.collect()
    d = d.with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
                       pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
                       ).join(att, on="serial_number", how="left").with_columns(
        pl.col("counsel").fill_null(False), pl.col("rd").is_not_null().alias("registered"),
        pl.col("rd").dt.year().alias("ry"))
    reg = d.filter(pl.col("fy").is_between(1995, 2018))
    out = {"n_filings_1995_2018": int(reg.height),
           "counsel_share_overall": float(reg["counsel"].mean()),
           "counsel_share_by_year": {str(k): round(float(v), 4) for k, v in
                                     reg.group_by("fy").agg(pl.col("counsel").mean()).sort("fy").iter_rows()},
           "registration_by_counsel": {str(k): {"n": int(n), "rate": float(r)} for k, n, r in
                                       reg.group_by("counsel").agg(pl.len(), pl.col("registered").mean()).sort("counsel").iter_rows()}}
    g = reg.filter(pl.col("ry").is_between(2002, 2018)).join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64).alias("failed"))
    out["gate_by_counsel"] = {str(k): {"n": int(n), "fail": float(f)} for k, n, f in
                              g.group_by("counsel").agg(pl.len(), pl.col("failed").mean()).sort("counsel").iter_rows()}
    # 3. median distinct-term counts by counsel, 009 and 035
    med = {}
    for c in ("009", "035"):
        s = pl.read_parquet(PROC / f"surprise_class{c}.parquet", columns=["serial_number", "n_terms", "year"]).filter(
            pl.col("year").is_between(2002, 2018)).join(att, on="serial_number", how="inner")
        med[c] = {str(k): float(v) for k, v in s.group_by("counsel").agg(pl.col("n_terms").median()).sort("counsel").iter_rows()}
    out["median_terms_by_counsel"] = med
    # 4. pooled owner ICC of gate failure (one-way ANOVA estimator)
    go = g.with_columns(pl.col("owner_name").str.to_uppercase().str.replace_all(r"[^A-Z0-9 ]", "")
                        .str.replace_all(r"\s+", " ").str.strip_chars().alias("own")).filter(
        pl.col("own").str.len_chars() >= 3)
    grp = go.group_by("own").agg(pl.col("failed").sum().alias("s"), pl.len().alias("k")).filter(pl.col("k") >= 2)
    N = int(grp["k"].sum()); G = int(grp.height)
    ybar = float(go.join(grp.select("own"), on="own", how="inner")["failed"].mean())
    kk = grp["k"].to_numpy().astype(float); ss = grp["s"].to_numpy().astype(float)
    mean_g = ss / kk
    ssb = float((kk * (mean_g - ybar) ** 2).sum())
    j = go.join(grp.select("own"), on="own", how="inner")
    ssw_df = j.join(pl.DataFrame({"own": grp["own"], "m": mean_g}), on="own")
    ssw = float(((ssw_df["failed"] - ssw_df["m"]) ** 2).sum())
    msb = ssb / (G - 1); msw = ssw / (N - G)
    k0 = (N - float((kk ** 2).sum()) / N) / (G - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    out["owner_icc_gate_failure"] = {"icc": round(icc, 4), "n_regs": N, "n_owners": G, "kbar": round(k0, 2)}
    log(json.dumps({k: v for k, v in out.items() if k != 'counsel_share_by_year'}, indent=1)[:1200])
    (RES / "representation_stats.json").write_text(json.dumps(out, indent=1))
    log("[done] representation_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
