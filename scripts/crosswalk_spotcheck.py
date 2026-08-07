"""Validate the USPTO -> SEC crosswalk against known companies.

The listing outcome is only as good as this linkage, and the linkage is a
normalized exact string match. This does three checks a human can read:

  1. KNOWN LISTED FIRMS. For a hand-picked set of companies that unambiguously
     went public, is the owner matched at all, is it matched to the RIGHT CIK,
     and do its goods/services descriptions describe what the company actually
     sells? A match to the right name but the wrong line of business is a false
     positive that no aggregate statistic will reveal.

  2. ADDRESS AGREEMENT. Where owner domicile is available, does the matched
     owner's state agree with the SEC registrant's? Disagreement at scale means
     the normalized names are colliding across distinct firms.

  3. RANDOM MATCHES. A random sample of matches, printed with the USPTO name,
     the SEC name and a description snippet, for eyeballing. Aggregate
     precision cannot be measured without labels; this at least surfaces the
     failure modes.

Read-only. Writes paper/results/crosswalk_spotcheck.json and prints a report.
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
OWNER_ADDR = REPO / "_archive_mac_syndication_2026-05-30" / "data-extras" / "owner_address.parquet"
CLASSES = [f"{i:03d}" for i in range(1, 46)]

# Companies that indisputably reached public markets, spread across sectors so
# the check is not just a software test. Matched on the normalized key.
KNOWN = [
    "MICROSOFT CORPORATION", "APPLE INC.", "NVIDIA CORPORATION",
    "PFIZER INC.", "HASBRO, INC.", "THE COCA-COLA COMPANY",
    "NIKE, INC.", "STARBUCKS CORPORATION", "TESLA, INC.",
    "NETFLIX, INC.", "AMAZON.COM, INC.", "INTEL CORPORATION",
    "SALESFORCE, INC.", "MODERNA, INC.", "AIRBNB, INC.",
    "UBER TECHNOLOGIES, INC.", "PELOTON INTERACTIVE, INC.",
    "BEYOND MEAT, INC.", "ZOOM VIDEO COMMUNICATIONS, INC.",
    "SNOWFLAKE INC.", "THE BOEING COMPANY", "GENERAL MOTORS LLC",
]


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sec_link import normalize

    xw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")
    log(f"[crosswalk] {xw.height:,} matched owner strings, "
        f"{xw['cik'].n_unique():,} CIKs")

    # ---------- 1. known listed firms -------------------------------------
    log("\n=== 1. KNOWN LISTED FIRMS ===")
    known_norm = {normalize(k): k for k in KNOWN}
    hit = xw.filter(pl.col("norm").is_in(list(known_norm)))
    found = set(hit["norm"].to_list())
    missing = [known_norm[n] for n in known_norm if n not in found]

    # pull a description for each matched owner
    want = set(hit["owner_name"].to_list())
    descs: dict[str, str] = {}
    for cls in CLASSES:
        p = PROC / f"tm_class{cls}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p, columns=["owner_name", "goods_services", "mark_identification"]
                            ).filter(pl.col("owner_name").is_in(list(want))
                                     & pl.col("goods_services").is_not_null())
        for r in d.iter_rows(named=True):
            descs.setdefault(r["owner_name"], f"[{r['mark_identification']}] {r['goods_services']}")
        del d
        gc.collect()

    rows = []
    for r in hit.sort("n_filings", descending=True).iter_rows(named=True):
        snip = (descs.get(r["owner_name"], "") or "")[:150].replace("\n", " ")
        rows.append({"uspto": r["owner_name"], "sec": r["sec_name"], "cik": r["cik"],
                     "n_filings": r["n_filings"], "desc": snip})
        log(f"  {r['owner_name'][:32]:32s} -> {r['sec_name'][:30]:30s} "
            f"cik={r['cik']:<9d} n={r['n_filings']:<5d}")
        log(f"      {snip}")
    log(f"\n  matched {len(found)}/{len(known_norm)} known firms")
    if missing:
        log(f"  NOT MATCHED: {', '.join(missing)}")

    # ---------- 2. address agreement --------------------------------------
    log("\n=== 2. DOMICILE AGREEMENT ===")
    addr_state = None
    if OWNER_ADDR.exists():
        addr = pl.read_parquet(OWNER_ADDR).select(
            "serial_number", "owner_state", "owner_country").unique(
            subset="serial_number", keep="first")
        own_ser = []
        for cls in CLASSES:
            p = PROC / f"tm_class{cls}.parquet"
            if not p.exists():
                continue
            own_ser.append(pl.read_parquet(p, columns=["serial_number", "owner_name"]
                                           ).filter(pl.col("owner_name").is_in(
                                               xw["owner_name"].to_list())))
            gc.collect()
        os_ = pl.concat(own_ser).unique(subset="serial_number")
        j = os_.join(addr, on="serial_number", how="inner")
        by_owner = j.group_by("owner_name").agg(
            pl.col("owner_state").drop_nulls().mode().first().alias("uspto_state"),
            pl.col("owner_country").drop_nulls().mode().first().alias("uspto_country"))
        xw2 = xw.join(by_owner, on="owner_name", how="left")
        us = xw2.filter(pl.col("uspto_country") == "US").drop_nulls("uspto_state")
        log(f"  {us.height:,} matched owners have a US state on file")
        addr_state = int(us.height)
    else:
        log("  owner_address.parquet not present; skipped")

    # ---------- 3. random matches -----------------------------------------
    log("\n=== 3. RANDOM MATCHES (eyeball) ===")
    samp = xw.filter(pl.col("n_filings") >= 3).sample(
        n=min(25, xw.height), seed=11).sort("n_filings", descending=True)
    want2 = set(samp["owner_name"].to_list())
    d2: dict[str, str] = {}
    for cls in CLASSES:
        p = PROC / f"tm_class{cls}.parquet"
        if not p.exists():
            continue
        dd = pl.read_parquet(p, columns=["owner_name", "goods_services"]).filter(
            pl.col("owner_name").is_in(list(want2)) & pl.col("goods_services").is_not_null())
        for r in dd.iter_rows(named=True):
            d2.setdefault(r["owner_name"], r["goods_services"])
        del dd
        gc.collect()
    rand = []
    for r in samp.iter_rows(named=True):
        snip = (d2.get(r["owner_name"], "") or "")[:110].replace("\n", " ")
        rand.append({"uspto": r["owner_name"], "sec": r["sec_name"],
                     "n_filings": r["n_filings"], "desc": snip})
        log(f"  {r['owner_name'][:30]:30s} -> {r['sec_name'][:28]:28s} n={r['n_filings']:<4d} | {snip}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "crosswalk_spotcheck.json").write_text(json.dumps(
        {"n_matched_owner_strings": int(xw.height),
         "n_ciks": int(xw["cik"].n_unique()),
         "known_firms_matched": len(found), "known_firms_tested": len(known_norm),
         "known_firms_missing": missing, "known": rows,
         "us_state_available": addr_state, "random_sample": rand}, indent=1))
    log("\n[done] paper/results/crosswalk_spotcheck.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
