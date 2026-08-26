"""Build the per-class data files behind the online IPO viewer.

One JSON per Nice class. Each row of the arrays is a registration
(registration years 1996-2018, scored on the production rolling measure),
positioned by lead and atypicality standardized within class and
registration year, and tiered by the latest gate its record reaches:

  0  registered, cancelled at the five-year proof of continued use
     (dated C8../C71T cancellation at registration age 4.0-8.5)
  1  passed the five-year proof
  2  owner appears in SEC financial reporting records (name-matched)
  3  owner carries an IPO marker (8-A registration or IPO date)

Tiers 2 and 3 are kept in full; tiers 0 and 1 are down-sampled so a class
file stays around a few hundred kilobytes. Coordinates are rounded to two
decimals and clipped to +/-3.5 standard deviations.

Output: docs/online-appendix/ipo-viewer/data/class_{NNN}.json
        docs/online-appendix/ipo-viewer/data/manifest.json
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
OUT = REPO / "docs" / "online-appendix" / "ipo-viewer" / "data"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
RY_LO, RY_HI = 1996, 2018
GATE_LO, GATE_HI = 4.0, 8.5
CLIP = 3.5
MAX_BASE_DOTS = 22_000  # cap on tiers 0+1 per class
SEED = 20260825


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((RES / "per_industry_names.json").read_text())
    sec_owners = set(pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet",
                                     columns=["owner_name"])["owner_name"].unique())
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet",
                         columns=["owner_name", "in_8a", "ipo_date"])
    ipo_owners = set(fm.filter((pl.col("in_8a").fill_null(0) == 1)
                               | pl.col("ipo_date").is_not_null())["owner_name"])
    log(f"[owners] SEC {len(sec_owners):,}  IPO {len(ipo_owners):,}")

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())

    manifest = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "registration_date", "owner_name"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
        ).drop_nulls("rd").unique("serial_number")
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        d = tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.col("rd").dt.year().alias("ry"),
            pl.col("topic_dkl").alias("L"),
            ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
        d = d.filter(pl.col("ry").is_between(RY_LO, RY_HI))
        d = d.join(ev, on="serial_number", how="left").with_columns(
            ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
            ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).alias("failed"))
        d = d.with_columns(
            ((pl.col("L") - pl.col("L").mean().over("ry")) / pl.col("L").std().over("ry")).alias("zL"),
            ((pl.col("A") - pl.col("A").mean().over("ry")) / pl.col("A").std().over("ry")).alias("zA"),
            pl.col("owner_name").is_in(sorted(sec_owners)).alias("sec"),
            pl.col("owner_name").is_in(sorted(ipo_owners)).alias("ipo"))
        d = d.with_columns(
            pl.when(pl.col("ipo")).then(3)
              .when(pl.col("sec")).then(2)
              .when(~pl.col("failed")).then(1)
              .otherwise(0).cast(pl.Int8).alias("g"))
        top = d.filter(pl.col("g") >= 2)
        base = d.filter(pl.col("g") < 2)
        if base.height > MAX_BASE_DOTS:
            base = base.sample(MAX_BASE_DOTS, seed=SEED)
        keep = pl.concat([base, top]).sort("g")  # ascending; the page draws in array order
        rec = {
            "cls": c,
            "name": names.get(c, c),
            "n_total": int(d.height),
            "n_by_tier": {str(t): int((d["g"] == t).sum()) for t in range(4)},
            "base_sampled": int(base.height),
            "ry": [int(v) for v in keep["ry"]],
            "x": [round(max(-CLIP, min(CLIP, float(v))), 2) for v in keep["zL"]],
            "y": [round(max(-CLIP, min(CLIP, float(v))), 2) for v in keep["zA"]],
            "g": [int(v) for v in keep["g"]],
        }
        (OUT / f"class_{c}.json").write_text(json.dumps(rec, separators=(",", ":")))
        manifest.append({"cls": c, "name": rec["name"], "n_total": rec["n_total"],
                         "n_by_tier": rec["n_by_tier"]})
        log(f"[{c}] {d.height:,} regs  tiers {rec['n_by_tier']}  dots {keep.height:,}")
        del tm, sc, d, top, base, keep
        gc.collect()
    (OUT / "manifest.json").write_text(json.dumps(
        {"ry_lo": RY_LO, "ry_hi": RY_HI, "classes": manifest}, indent=1))
    log("[done] ipo-viewer data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
