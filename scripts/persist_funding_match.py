"""Persist the trademark-owner -> Form D issuer match set for reuse.

scripts/atypicality_and_funding.py derives this linkage inline and throws it away
at the end of the run.  Other scripts that want "did this owner ever raise a
priced Reg D round, and did it subsequently list?" should not have to re-derive
the Form D filtering, the collision handling, or the EDGAR 8-A pass.  This script
builds that table once and writes it to

    data/processed/funding_owner_match.parquet

ONE ROW PER MATCHED TRADEMARK OWNER.  The owner universe is every distinct
owner_name in data/raw/owner_debut.parquet (i.e. every owner appearing in any of
the 45 NICE-class files), NOT just the 2009-2018 debut cohort that the funding
analysis regresses on -- a persisted artifact should be the superset, and callers
can restrict it.

Construction
------------
FORM D.  data/raw/formd/form_d_issuers.parquet, parsed from the 73 quarterly SEC
Form D data sets (2009q1-2026q2) by scripts/funding_lag_prototype.py.  Kept rows
are OPERATING COMPANIES only:
    ~ISPOOLEDINVESTMENTFUNDTYPE   drops pooled funds and SPVs (54.7% of rows)
    INVESTMENTFUNDTYPE == ""      drops the residual hedge/PE/VC fund vehicles
    IS_PRIMARYISSUER_FLAG         keeps the primary issuer of each offering
Amendments and offerings with no usable round date are dropped, then the earliest
remaining notice per normalized issuer name is taken.  Any normalized name that
resolves to more than one CIK is a collision that cannot be assigned to a single
trademark owner, and is dropped outright rather than guessed at.

MATCH.  Owner names are normalized with scripts/sec_link.normalize (imported, not
re-implemented) and joined to the issuer names by exact match on the normalized
key.  This is the same aggressive-exact matcher the USPTO-SEC crosswalk uses.

LISTING MARKERS.  Three, deliberately independent in what they depend on:
    in_sec   owner_name appears in data/processed/uspto_sec_crosswalk.parquet.
             This is the paper's own listed-universe marker.  NOTE it inherits
             the crosswalk's fuzzy-pass rule, which requires >= 10 USPTO filings,
             so it is partly a filing-volume marker.  Prefer in_fsds for clean work.
    in_fsds  the matched Form D CIK appears in data/processed/sec_firm_year.parquet
             (SEC Financial Statement Data Sets = the XBRL filer universe).  Needs
             no owner-name matching beyond the Form D link.
    in_8a    the CIK has a first-ever Form 8-A12B or 8-A12G on EDGAR = Exchange
             Act Sec. 12 registration = the actual moment of exchange listing.
             ipo_date carries that date.  The prototype's S-1 / 424B / 10-K
             fallbacks are EXCLUDED: S-1 and 10-K are filed by many companies that
             never list, and including them inflates the outcome.

!! in_8a IS SCREENED, NOT ENUMERATED.  The EDGAR submissions API is queried only
!! for matched CIKs that already appear in the FSDS or crosswalk universes.  A
!! company that registered a class of securities under Sec. 12 but files no XBRL
!! financials and is absent from both universes would be missed.  That is close to
!! a null set, and the screen is what keeps the pass to ~3.5k lookups instead of
!! ~50k, but in_8a == 0 outside the screen means "not verified", not "checked and
!! found absent".  in_fsds and in_sec are complete over the matched set.
!! (169 rows do carry in_8a == 1 while outside the current screen: the cache also
!! holds CIKs looked up by earlier runs, so coverage is a superset of the screen,
!! never a subset.)

Column semantics that will bite you if you assume otherwise
-----------------------------------------------------------
first_formd_date  is the ROUND date, i.e. the offering's date of first sale,
                  falling back to the notice filing date when the sale date is
                  missing or implausible (the parser rejects a sale date that
                  postdates the filing or precedes it by more than 1095 days).
                  Because a 2009 notice may report a 2006 first sale, 592 rows
                  carry a first_formd_date before 2009 even though no Form D was
                  e-filed before 2009q1.  It is NOT the notice filing date.
first_amount      TOTALAMOUNTSOLD on that first notice.  Never null, but 8,942
                  rows are 0.0, which means the issuer reported nothing sold yet,
                  not that the value is missing.  Treat 0 as a reported zero and
                  filter it explicitly if you want realized proceeds.
ipo_date          the 8-A date, which may PRECEDE first_formd_date.  Of the 6,593
                  rows with in_8a == 1, 3,688 (56%) have ipo_date < first_formd_date:
                  these are already-listed companies doing Reg D private placements
                  (PIPEs), NOT startups that raised and then went public.  Only
                  2,905 rows are "funded, then listed".  Any funding -> IPO analysis
                  MUST compare the two dates rather than reading in_8a as an
                  outcome that follows the round.
owner_name        one row per distinct USPTO owner STRING, per the requested
                  schema.  USPTO casing and punctuation vary, so a single firm can
                  appear several times ("10X GENOMICS, INC." / "10X Genomics, Inc."
                  / "10x Genomics, Inc." are three rows on one CIK; the worst case
                  is 32 rows).  91,029 rows cover only 49,667 distinct CIKs.  Group
                  on cik (or norm) for firm-level work; counting rows counts names.
in_sec            keyed on the exact owner_name string, so it is INCONSISTENT across
                  those casing variants: 1,912 CIKs have some name variants flagged
                  and others not.  in_fsds and in_8a key on CIK and are perfectly
                  consistent within a firm (0 inconsistent CIKs).  Prefer them.

Known biases, carried forward from the match itself
---------------------------------------------------
*  Form D is electronic-only from 2009-03-16.  An owner that raised its rounds
   before 2009 is invisible here regardless of how large it became.
*  Normalized-exact matching lands ~1.6% of all owners and strongly favours
   formally suffixed corporate entities (in the 2009-2018 debut cohort, owners
   whose name carries a corporate suffix match at 4.21% against 0.34% for those
   that do not -- a 12x gradient).  Absence from this table is NOT evidence that
   an owner never raised money.
*  Holding-company renames break the CIK link at listing (e.g. "Coinbase, Inc."
   raising under one CIK and "Coinbase Global, Inc." listing under another), so
   in_8a and in_fsds are biased toward zero for firms that restructured at IPO.

Inputs
------
data/raw/owner_debut.parquet                owner universe
data/raw/formd/form_d_issuers.parquet       parsed Form D
data/processed/uspto_sec_crosswalk.parquet  in_sec
data/processed/sec_firm_year.parquet        in_fsds
data/raw/formd/edgar_8a_funded.json         8-A cache (extended in place)

Usage
-----
    PYTHONPATH=src .venv-analysis/Scripts/python.exe scripts/persist_funding_match.py
    ... --skip-edgar        # use only what is already cached
    ... --max-ciks 500      # cap the EDGAR pass
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RAWD = REPO / "data" / "raw"
OUT = PROC / "funding_owner_match.parquet"
CACHE = RAWD / "formd" / "edgar_8a_funded.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sec_link import normalize  # noqa: E402

UA = "Eric Silver epsilver@gmail.com (novelty research)"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
IPO_FORMS = ("8-A12B", "8-A12G")   # S-1 / 424B / 10-K fallbacks deliberately excluded
SLEEP_API = 0.12


def formd_operating_issuers() -> tuple[pl.DataFrame, dict]:
    """Earliest operating-company Reg D notice per unambiguous normalized name."""
    fd = pl.read_parquet(RAWD / "formd" / "form_d_issuers.parquet")
    ops = fd.filter(~pl.col("is_pooled") & (pl.col("fund_type") == "")
                    & pl.col("is_primary"))
    amb = (ops.group_by("norm").agg(pl.col("cik").n_unique().alias("n_cik"))
           .filter(pl.col("n_cik") > 1)["norm"].implode())
    fr = (
        ops.filter(~pl.col("is_amend") & pl.col("round_date").is_not_null()
                   & (pl.col("norm") != ""))
        .sort("round_date")
        .group_by("norm", maintain_order=True)
        .agg(pl.col("cik").first(),
             pl.col("round_date").first().alias("first_formd_date"),
             pl.col("total_sold").first().alias("first_amount"),
             pl.len().alias("n_formd"))
        .filter(~pl.col("norm").is_in(amb))
    )
    meta = {
        "formd_issuer_offerings_total": int(fd.height),
        "formd_pooled_share": round(float(fd["is_pooled"].mean()), 4),
        "formd_operating_offerings": int(ops.height),
        "formd_names_dropped_as_cik_collisions": int(amb.explode().len()),
        "formd_unambiguous_operating_issuers": int(fr.height),
    }
    print(f"[formd] {fd.height:,} issuer-offerings, {fd['is_pooled'].mean()*100:.1f}% pooled "
          f"-> {ops.height:,} operating -> {fr.height:,} unambiguous issuer names",
          file=sys.stderr, flush=True)
    return fr, meta


def edgar_8a(ciks: list[int], skip: bool, max_ciks: int) -> dict[str, str]:
    """First-ever 8-A12B/8-A12G date per CIK; "" means looked up and not found."""
    cache: dict[str, str] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [c for c in ciks if str(c) not in cache][:max_ciks]
    if skip or not todo:
        print(f"[edgar] {len(cache):,} cached; {0 if skip else len(todo):,} to look up",
              file=sys.stderr, flush=True)
        return cache
    import requests
    print(f"[edgar] {len(todo):,} CIKs to look up "
          f"({len(ciks)-len(todo):,} already cached)", file=sys.stderr, flush=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    for i, cik in enumerate(todo, 1):
        best = ""
        try:
            r = s.get(SUBMISSIONS.format(cik=cik), timeout=60)
            if r.status_code == 200:
                j = r.json()
                blocks = [j["filings"]["recent"]]
                for extra in j["filings"].get("files", []):
                    time.sleep(SLEEP_API)
                    r2 = s.get("https://data.sec.gov/submissions/" + extra["name"],
                               timeout=60)
                    if r2.status_code == 200:
                        blocks.append(r2.json())
                for b in blocks:
                    for f, dt in zip(b.get("form", []), b.get("filingDate", [])):
                        if f in IPO_FORMS and (not best or dt < best):
                            best = dt
        except Exception as exc:
            print(f"[edgar] cik {cik}: {type(exc).__name__}", file=sys.stderr)
            continue          # leave uncached so a later run retries it
        cache[str(cik)] = best
        if i % 200 == 0:
            CACHE.write_text(json.dumps(cache))
            print(f"[edgar] {i:,}/{len(todo):,}", file=sys.stderr, flush=True)
        time.sleep(SLEEP_API)
    CACHE.write_text(json.dumps(cache))
    return cache


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-edgar", action="store_true",
                    help="use only the cached 8-A lookups, make no network calls")
    ap.add_argument("--max-ciks", type=int, default=8000,
                    help="cap on new EDGAR submissions-API lookups")
    args = ap.parse_args()
    t0 = time.time()

    fr, meta = formd_operating_issuers()

    # ---- owner universe -> normalized key ---------------------------------
    own = (pl.read_parquet(RAWD / "owner_debut.parquet")
           .with_columns(pl.col("owner_name")
                         .map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))
           .filter(pl.col("norm") != ""))
    n_owners = own.height
    print(f"[owners] {n_owners:,} distinct trademark owners normalized "
          f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    m = own.join(fr, on="norm", how="inner").drop("debut")
    print(f"[match] {m.height:,} owners matched to a Form D operating-company issuer "
          f"({m.height/n_owners*100:.2f}% of owners), {m['cik'].n_unique():,} distinct CIKs",
          file=sys.stderr, flush=True)

    # ---- listing markers --------------------------------------------------
    cw = (pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")
          .select("owner_name").unique().with_columns(pl.lit(True).alias("_sec")))
    fs = (pl.read_parquet(PROC / "sec_firm_year.parquet")
          .select("cik").unique().with_columns(pl.lit(True).alias("_fsds")))
    m = (m.join(cw, on="owner_name", how="left")
          .join(fs, on="cik", how="left")
          .with_columns(pl.col("_sec").fill_null(False),
                        pl.col("_fsds").fill_null(False)))

    cand = m.filter(pl.col("_sec") | pl.col("_fsds"))["cik"].unique().to_list()
    print(f"[edgar] screen: {len(cand):,} matched CIKs are in the FSDS or crosswalk "
          f"universe and get an 8-A lookup", file=sys.stderr, flush=True)
    cache = edgar_8a(cand, args.skip_edgar, args.max_ciks)

    hits = pl.DataFrame(
        {"cik": [int(k) for k, v in cache.items() if v],
         "ipo_date": [v for v in cache.values() if v]},
        schema={"cik": pl.Int64, "ipo_date": pl.Utf8},
    ).with_columns(pl.col("ipo_date").str.to_date("%Y-%m-%d", strict=False))

    out = (
        m.join(hits, on="cik", how="left")
         .with_columns(
             pl.col("_sec").cast(pl.Int8).alias("in_sec"),
             pl.col("_fsds").cast(pl.Int8).alias("in_fsds"),
             pl.col("ipo_date").is_not_null().cast(pl.Int8).alias("in_8a"),
             pl.col("first_amount").cast(pl.Float64),
             pl.col("n_formd").cast(pl.Int32),
             pl.col("cik").cast(pl.Int64),
         )
         .select("owner_name", "norm", "cik", "first_formd_date", "n_formd",
                 "first_amount", "in_sec", "in_fsds", "in_8a", "ipo_date")
         .sort("owner_name")
    )
    assert out.height == out["owner_name"].n_unique(), "owner_name must be unique"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)

    summary = {
        "rows": int(out.height),
        "distinct_cik": int(out["cik"].n_unique()),
        "owner_universe": int(n_owners),
        "match_rate": round(out.height / n_owners, 5),
        "in_sec": int(out["in_sec"].sum()),
        "in_fsds": int(out["in_fsds"].sum()),
        "in_8a": int(out["in_8a"].sum()),
        "edgar_screen_ciks": len(cand),
        "edgar_cached_ciks": len(cache),
        "with_first_amount": int(out["first_amount"].is_not_null().sum()),
        "first_formd_date_range": [str(out["first_formd_date"].min()),
                                   str(out["first_formd_date"].max())],
        **meta,
    }
    print("\n" + "=" * 74, file=sys.stderr)
    for k, v in summary.items():
        print(f"  {k:<42s} {v}", file=sys.stderr)
    print("=" * 74, file=sys.stderr)
    print(f"[done] -> {OUT}  ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    print("in_8a is SCREENED to CIKs already in the FSDS/crosswalk universes; "
          "in_8a==0\noutside that screen means 'not verified', not 'checked and absent'.",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
