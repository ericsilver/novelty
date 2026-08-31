"""Timing-resolved outcomes for imitated originals, with the survivorship checks.

Population: first filers ("originals") of small text clusters (2-99 identical
filings, >= 2 distinct owners) and, as baseline, unique-text filings.
For each cluster the first DIFFERENT-owner filing date is the copy event;
originals are classed by copy lag (first copy within 2 years of the
original's filing; 2-6 years; over 6 years).

Survivorship checks (does copying condition on grant?):
  a. the registration rate of copied originals' clusters vs the base --
     i.e., do abandoned originals get copied at all;
  b. among registered originals, the share whose first copy PREDATES their
     registration date (pre-grant copying of a pending application).

Timing-clean outcomes (predictor fully realized before the outcome):
  1. five-year proof failure of registered originals (cohorts 2002-2018)
     whose first copy came within 2 years of filing -- the copy predates
     the proof window for essentially all -- vs unique-text registrations,
     within class x registration year;
  2. late SEC entry: for every registered debut in the frame, the outcome
     is appearance in SEC reporting with first SEC year at least three
     years after the filing year (so the outcome clock starts after the
     <=2y copy window closes for everyone), within class x filing year;
     owners already in SEC on or before the filing year are dropped;
  3. IPO marker with ipo_date after the copy date (originals) / after
     filing year + 2 (uniques).

Output: paper/results/timed_copy_outcomes.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
FY_LO, FY_HI = 1995, 2018


def log(m): print(m, file=sys.stderr, flush=True)


def load_class(c):
    tp = PROC / f"tm_class{c}.parquet"
    if not tp.exists():
        return None
    tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date",
                                      "owner_name", "goods_services"]).filter(
        pl.col("filing_date").fill_null("").str.len_chars() >= 8).with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8).alias("reg"),
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        pl.col("goods_services").fill_null("").str.to_lowercase()
          .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ")
          .str.strip_chars().alias("norm"),
        pl.col("owner_name").fill_null("").str.to_uppercase()
          .str.replace_all(r"[^A-Z0-9 ]", "").str.replace_all(r"\s+", " ")
          .str.strip_chars().alias("own"))
    tm = tm.filter(pl.col("fy").is_between(FY_LO, FY_HI) & pl.col("fd").is_not_null()
                   & (pl.col("norm").str.len_chars() > 0)).with_columns(
        pl.col("norm").hash(seed=7).alias("h")).drop("norm", "goods_services",
                                                     "filing_date", "registration_date",
                                                     "owner_name")
    cl1 = tm.sort(["fd", "serial_number"]).group_by("h").agg(
        pl.len().alias("dupN"), pl.col("own").n_unique().alias("dup_owners"),
        pl.col("fd").first().alias("fd0"), pl.col("own").first().alias("own0"),
        pl.col("serial_number").first().alias("serial0"))
    other = tm.join(cl1.select("h", "own0"), on="h").filter(
        pl.col("own") != pl.col("own0")).group_by("h").agg(
        pl.col("fd").min().alias("copy_fd"))
    cl1 = cl1.join(other, on="h", how="left")
    del other
    tm = tm.join(cl1, on="h", how="left")
    originals = tm.filter((pl.col("serial_number") == pl.col("serial0"))
                          & pl.col("dupN").is_between(2, 99) & (pl.col("dup_owners") >= 2)
                          & pl.col("copy_fd").is_not_null()).with_columns(
        ((pl.col("copy_fd") - pl.col("fd")).dt.total_days() / 365.25).alias("copy_lag_y"))
    uniques = tm.filter(pl.col("dupN") == 1).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("copy_lag_y"),
        pl.lit(None, dtype=pl.Date).alias("copy_fd_x"))
    out = pl.concat([
        originals.select("serial_number", "fy", "fd", "reg", "rd", "own",
                         pl.lit(True).alias("is_orig"), "copy_lag_y", "copy_fd"),
        uniques.select("serial_number", "fy", "fd", "reg", "rd", "own",
                       pl.lit(False).alias("is_orig"),
                       "copy_lag_y", pl.col("copy_fd_x").alias("copy_fd")),
    ]).with_columns(pl.lit(c).alias("cls"))
    del tm, cl1, originals, uniques
    gc.collect()
    return out


def main() -> int:
    parts = []
    for c in CLASSES:
        f = load_class(c)
        if f is not None:
            parts.append(f)
            log(f"[{c}] {f.height:,}")
    d = pl.concat(parts); del parts; gc.collect()
    d = d.with_columns(
        pl.when(~pl.col("is_orig")).then(pl.lit("unique"))
          .when(pl.col("copy_lag_y") <= 2).then(pl.lit("orig_copied_le2y"))
          .when(pl.col("copy_lag_y") <= 6).then(pl.lit("orig_copied_2_6y"))
          .otherwise(pl.lit("orig_copied_gt6y")).alias("cat"))
    log(f"[frame] {d.height:,} rows "
        + json.dumps({k[0]: v for k, v in
                      sorted(d.group_by("cat").len().rows())}, default=str))

    out = {"survivorship": {}, "registration": {}, "gate": {}, "sec": {}}

    # --- survivorship checks ---
    orig = d.filter(pl.col("is_orig"))
    out["survivorship"]["n_originals"] = orig.height
    out["survivorship"]["original_reg_rate"] = float(orig["reg"].mean())
    out["survivorship"]["unique_reg_rate"] = float(d.filter(~pl.col("is_orig"))["reg"].mean())
    ro = orig.filter(pl.col("reg") & pl.col("rd").is_not_null())
    out["survivorship"]["share_first_copy_before_registration"] = float(
        (ro["copy_fd"] < ro["rd"]).mean())
    ab = orig.filter(~pl.col("reg"))
    out["survivorship"]["n_abandoned_originals_copied"] = ab.height
    out["survivorship"]["share_originals_abandoned"] = float(ab.height / orig.height)
    for lag, sub in (("le2y", orig.filter(pl.col("copy_lag_y") <= 2)),
                     ("gt2y", orig.filter(pl.col("copy_lag_y") > 2))):
        out["survivorship"][f"orig_reg_rate_copy_{lag}"] = float(sub["reg"].mean())
    log("[survivorship] " + json.dumps(out["survivorship"]))

    # --- registration, within class x filing year ---
    d = d.with_columns((pl.col("reg").cast(pl.Float64)
                        - pl.col("reg").cast(pl.Float64).mean().over(["cls", "fy"])).alias("reg_dm"))
    for (cat,), g in sorted(d.group_by("cat"), key=lambda kv: kv[0][0]):
        out["registration"][cat] = {"n": g.height, "reg_rate": float(g["reg"].mean()),
                                    "delta_within": float(g["reg_dm"].mean())}
        log(f"[reg|{cat}] " + json.dumps(out["registration"][cat]))

    # --- five-year proof, timing-clean (copy within 2y of filing) ---
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    r = d.filter(pl.col("reg") & pl.col("rd").is_not_null()).with_columns(
        pl.col("rd").dt.year().alias("ry")).filter(pl.col("ry").is_between(2002, 2018))
    r = r.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        ((pl.col("age") >= 4.0) & (pl.col("age") < 8.5)).fill_null(False)
        .cast(pl.Float64).alias("fail")).with_columns(
        (pl.col("fail") - pl.col("fail").mean().over(["cls", "ry"])).alias("fail_dm"))
    for (cat,), g in sorted(r.group_by("cat"), key=lambda kv: kv[0][0]):
        se = (g["fail"].mean() * (1 - g["fail"].mean()) / g.height) ** 0.5
        out["gate"][cat] = {"n": g.height, "fail": float(g["fail"].mean()),
                            "delta_within": float(g["fail_dm"].mean()),
                            "se": float(se)}
        log(f"[gate|{cat}] " + json.dumps(out["gate"][cat]))
    del r, ev
    gc.collect()

    # --- late SEC entry and post-copy IPO, on registered debuts ---
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet",
                         columns=["owner_name", "first_year"]).group_by("owner_name").agg(
        pl.col("first_year").min())
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet",
                         columns=["owner_name", "ipo_date"]).filter(
        pl.col("ipo_date").is_not_null()).group_by("owner_name").agg(pl.col("ipo_date").min())
    debut = d.group_by("own").agg(pl.col("fy").min().alias("debut_fy"))
    first = d.join(debut, on="own", how="inner").filter(
        (pl.col("fy") == pl.col("debut_fy")) & pl.col("reg")).sort("serial_number").unique(
        "own", keep="first")
    first = first.join(cw, left_on="own", right_on="owner_name", how="left").join(
        fm, left_on="own", right_on="owner_name", how="left")
    # drop owners already reporting on/before their filing year
    first = first.filter(pl.col("first_year").is_null() | (pl.col("first_year") > pl.col("fy")))
    first = first.with_columns(
        (pl.col("first_year").is_not_null()
         & (pl.col("first_year") >= pl.col("fy") + 3)).cast(pl.Float64).alias("late_sec"),
        (pl.col("ipo_date").is_not_null()
         & (pl.col("ipo_date").dt.year() >= pl.col("fy") + 3)).cast(pl.Float64).alias("late_ipo"))
    first = first.with_columns(
        (pl.col("late_sec") - pl.col("late_sec").mean().over(["cls", "fy"])).alias("sec_dm"),
        (pl.col("late_ipo") - pl.col("late_ipo").mean().over(["cls", "fy"])).alias("ipo_dm"))
    for (cat,), g in sorted(first.group_by("cat"), key=lambda kv: kv[0][0]):
        out["sec"][cat] = {"n": g.height,
                           "late_sec_rate": float(g["late_sec"].mean()),
                           "sec_delta_within": float(g["sec_dm"].mean()),
                           "late_ipo_rate": float(g["late_ipo"].mean()),
                           "ipo_delta_within": float(g["ipo_dm"].mean())}
        log(f"[sec|{cat}] " + json.dumps(out["sec"][cat]))

    (RES / "timed_copy_outcomes.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
