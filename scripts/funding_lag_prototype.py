"""Trademark -> venture funding -> IPO lags for novel commercial vocabulary.

Proof-of-concept for two terms (default "blockchain" and "telehealth"), using
only free public sources. Crunchbase is NOT used: see the header note in
paper/results/funding_lag_prototype.json and the discussion in the run log.

Pipeline
--------
1. TERM EMERGENCE. Regex the raw goods_services text of every NICE class
   (data/processed/tm_class*.parquet) -- the same mechanism diff2_theme_transit.py
   uses -- to get every filing whose description carries the term. Two emergence
   dates are recorded: the first-ever term-bearing filing, and the "sustained"
   date at which the cumulative count of term-bearing filings reaches
   MIN_SUSTAINED (a single 1987 stray should not define a term's birth).

2. FIRMS. Each distinct owner_name of a term-bearing filing is a candidate firm,
   dated at that owner's first term-bearing filing. Owners are then split into two
   strata, because pooling them makes the lag meaningless: DEBUT CARRIERS, whose
   term-bearing filing is (within DEBUT_WINDOW_DAYS of) their first-ever USPTO
   filing in any class -- firms that entered the record already speaking the
   vocabulary -- and INCUMBENT ADOPTERS, established filers who picked the term up
   later. For an incumbent, "first term-bearing filing" is a mid-life event that
   routinely postdates both its financing and its IPO, so its lags run negative and
   measure nothing about the propagation pipeline.

3. FUNDING. SEC Form D (Reg D notice of exempt offering) quarterly structured
   data sets, 2008q1-present. Owner names are matched to Form D ISSUERS.ENTITYNAME
   by the aggressive normalized exact match of scripts/sec_link.py (imported, not
   re-implemented). Pooled investment funds, amendments, and TEST filings are
   dropped. Round date = date of first sale, falling back to the filing date.

4. IPO. Form D carries the issuer's EDGAR CIK, so the funding -> IPO link needs no
   name matching at all. For each matched CIK the EDGAR submissions API gives the
   full filing history; the IPO is dated at the first-ever Form 8-A12B/8-A12G
   (Exchange Act Sec. 12 registration = the moment of exchange listing), with
   424B4/424B1 (final pricing prospectus), S-1, and first 10-K recorded as
   fallbacks and cross-checks.

5. LAGS. Per firm: tm -> first Form D, first Form D -> IPO, tm -> IPO. Reported as
   medians, because the distributions are right-skewed and right-censored.

Caching
-------
data/raw/formd/*.zip                 quarterly Form D data sets (~250 MB full history)
data/raw/formd/form_d_issuers.parquet   parsed one-row-per-(accession, issuer)
data/raw/formd/edgar_first_forms.json   per-CIK first-filing dates from the submissions API
data/raw/owner_debut.parquet         each owner's first-ever filing date, all 45 classes

Usage
-----
    PYTHONPATH=src .venv-analysis/Scripts/python.exe scripts/funding_lag_prototype.py
    ... --terms blockchain telehealth --start 2009q1 --max-ciks 600

Outputs: paper/results/funding_lag_prototype.json (+ stdout summary table)
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from datetime import date, datetime
from pathlib import Path

import polars as pl
import requests

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RAW = REPO / "data" / "raw" / "formd"
OUT = REPO / "paper" / "results" / "funding_lag_prototype.json"

# reuse the crosswalk normalizer verbatim rather than forking it
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sec_link import normalize  # noqa: E402

UA = "Eric Silver epsilver@gmail.com (novelty research)"
FORMD_INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-d-data-sets"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# term label -> regex on lowercased goods_services (diff2_theme_transit.py style)
TERMS = {
    "blockchain": r"blockchain|distributed ledger",
    "telehealth": r"telehealth|telemedicine|tele-?health",
    "as-a-service": r"as a service|\bsaas\b|\bpaas\b|\biaas\b",
    "machine-learning": r"machine learning|artificial intelligence",
    "non-fungible-token": r"non[\s-]?fungible token|\bnft\b",
    "cbd": r"\bcbd\b|cannabidiol",
    "vape": r"\bvap(e|ing|orizer)\b|e-?cigarette",
}

MIN_SUSTAINED = 10          # cumulative term-bearing filings defining "sustained" emergence
MIN_OWNER_FILINGS = 1       # owners are kept from their first term-bearing filing
DEBUT_WINDOW_DAYS = 365     # term filing this close to the owner's first-ever filing = debut carrier
IPO_FORMS = ["8-A12B", "8-A12G", "424B4", "424B1", "S-1", "10-K"]
IPO_PRIMARY = ("8-A12B", "8-A12G")
IPO_FALLBACK = ("424B4", "424B1", "S-1", "10-K")
SLEEP_BULK = 1.1            # seconds between SEC bulk-file requests
SLEEP_API = 0.15            # seconds between data.sec.gov calls (SEC allows 10/s)


# ---------------------------------------------------------------- helpers ---
def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    return s


def parse_date(s: str) -> date | None:
    """Form D uses ISO timestamps in early quarters and DD-MON-YYYY in recent ones."""
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def tm_date(s: str) -> date | None:
    """USPTO filing_date is YYYYMMDD."""
    s = (s or "").strip()
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return float(xs[n // 2]) if n % 2 else float((xs[n // 2 - 1] + xs[n // 2]) / 2)


# ------------------------------------------------------ step 1: Form D dl ---
def formd_links(s: requests.Session) -> dict[str, str]:
    """Scrape the Form D data-sets page. The filename stem is stable (YYYYqN_d)
    but the directory prefix and a trailing _0 are not, so never construct URLs."""
    cache = RAW / "_formd_links.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = s.get(FORMD_INDEX, timeout=120)
    r.raise_for_status()
    out: dict[str, str] = {}
    for href in re.findall(r'href="([^"]+\.zip)"', r.text):
        m = re.search(r"/(\d{4}q\d)_d(?:_\d+)?\.zip$", href, flags=re.I)
        if not m:
            continue
        out[m.group(1).lower()] = href if href.startswith("http") else "https://www.sec.gov" + href
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


def download_formd(s: requests.Session, start: str, end: str | None) -> list[Path]:
    links = formd_links(s)
    qs = sorted(q for q in links if q >= start and (end is None or q <= end))
    print(f"[formd] {len(qs)} quarters available {qs[0]}..{qs[-1]}", file=sys.stderr, flush=True)
    paths = []
    for q in qs:
        dest = RAW / f"{q}_d.zip"
        paths.append(dest)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        for attempt in range(4):
            try:
                r = s.get(links[q], timeout=300)
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(dest)
                print(f"[get ] {q} {len(r.content)/1e6:.1f} MB", file=sys.stderr, flush=True)
                break
            except (requests.RequestException, OSError) as e:
                print(f"[retry {attempt+1}] {q}: {e}", file=sys.stderr)
                time.sleep(2**attempt)
        time.sleep(SLEEP_BULK)
    return [p for p in paths if p.exists()]


# ------------------------------------------------------ step 2: Form D parse ---
def _tsv(zf: zipfile.ZipFile, root: str, name: str):
    with zf.open(f"{root}/{name}") as fh:
        tw = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        hdr = next(tw).lstrip("﻿").rstrip("\r\n").split("\t")
        ix = {h: i for i, h in enumerate(hdr)}
        for line in tw:
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) < len(hdr):
                continue
            yield ix, cols


def parse_formd(zips: list[Path]) -> pl.DataFrame:
    """One row per (accession, issuer) with the round date and offering size."""
    cache = RAW / "form_d_issuers.parquet"
    if cache.exists():
        df = pl.read_parquet(cache)
        print(f"[formd] cached parse: {df.height:,} issuer-offerings", file=sys.stderr, flush=True)
        return df

    rows: list[dict] = []
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            root = zf.namelist()[0].split("/")[0]
            sub: dict[str, dict] = {}
            for ix, c in _tsv(zf, root, "FORMDSUBMISSION.tsv"):
                if c[ix["TESTORLIVE"]].upper() != "LIVE":
                    continue
                sub[c[ix["ACCESSIONNUMBER"]]] = {
                    "filing_date": parse_date(c[ix["FILING_DATE"]]),
                    "submission_type": c[ix["SUBMISSIONTYPE"]],
                    "sic": c[ix["SIC_CODE"]],
                }
            off: dict[str, dict] = {}
            for ix, c in _tsv(zf, root, "OFFERING.tsv"):
                off[c[ix["ACCESSIONNUMBER"]]] = {
                    "industry": c[ix["INDUSTRYGROUPTYPE"]],
                    "is_pooled": c[ix["ISPOOLEDINVESTMENTFUNDTYPE"]].lower() == "true",
                    "fund_type": c[ix["INVESTMENTFUNDTYPE"]],
                    "is_amend": c[ix["ISAMENDMENT"]].lower() == "true",
                    "sale_date": parse_date(c[ix["SALE_DATE"]]),
                    "total_offering": c[ix["TOTALOFFERINGAMOUNT"]],
                    "total_sold": c[ix["TOTALAMOUNTSOLD"]],
                    "revenue_range": c[ix["REVENUERANGE"]],
                }
            for ix, c in _tsv(zf, root, "ISSUERS.tsv"):
                acc = c[ix["ACCESSIONNUMBER"]]
                if acc not in sub or acc not in off:
                    continue
                cik_raw = c[ix["CIK"]].strip()
                if not cik_raw.isdigit():
                    continue
                s_, o_ = sub[acc], off[acc]
                fd = s_["filing_date"]
                sd = o_["sale_date"]
                # first-sale dates are sometimes the company's inception, decades
                # before the notice; only trust one that sits just behind the filing
                if fd and sd and (sd > fd or (fd - sd).days > 1095):
                    sd = None
                rows.append(
                    {
                        "accession": acc,
                        "cik": int(cik_raw),
                        "entity_name": c[ix["ENTITYNAME"]],
                        "state": c[ix["STATEORCOUNTRY"]],
                        "entity_type": c[ix["ENTITYTYPE"]],
                        "year_inc": c[ix["YEAROFINC_VALUE_ENTERED"]],
                        "is_primary": c[ix["IS_PRIMARYISSUER_FLAG"]].upper() == "YES",
                        "filing_date": fd,
                        "round_date": sd or fd,
                        "submission_type": s_["submission_type"],
                        "sic": s_["sic"],
                        **o_,
                    }
                )
        print(f"[parse] {zp.name}: {len(rows):,} cumulative issuer rows", file=sys.stderr, flush=True)

    df = pl.DataFrame(rows, infer_schema_length=None).drop("sale_date")
    df = df.with_columns(
        pl.col("entity_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"),
        pl.col("total_offering").cast(pl.Float64, strict=False),
        pl.col("total_sold").cast(pl.Float64, strict=False),
    )
    df.write_parquet(cache)
    print(f"[formd] parsed {df.height:,} issuer-offerings -> {cache}", file=sys.stderr, flush=True)
    return df


# -------------------------------------------------- step 3: term -> owners ---
def all_classes() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def owner_debut() -> pl.DataFrame:
    """Every owner's first-ever filing date across all 45 NICE classes."""
    cache = REPO / "data" / "raw" / "owner_debut.parquet"
    if cache.exists():
        d = pl.read_parquet(cache)
        print(f"[debut] cached: {d.height:,} owners", file=sys.stderr, flush=True)
        return d
    parts = []
    for cls in all_classes():
        parts.append(
            pl.scan_parquet(PROC / f"tm_class{cls}.parquet")
            .select("owner_name", "filing_date")
            .filter(pl.col("owner_name").is_not_null()
                    & (pl.col("filing_date").str.len_chars() == 8))
            .group_by("owner_name")
            .agg(pl.col("filing_date").min().alias("debut"))
            .collect()
        )
        print(f"[debut] class {cls}", file=sys.stderr, flush=True)
    d = pl.concat(parts).group_by("owner_name").agg(pl.col("debut").min())
    d.write_parquet(cache)
    print(f"[debut] {d.height:,} owners -> {cache}", file=sys.stderr, flush=True)
    return d


def term_filings(term: str, pattern: str) -> pl.DataFrame:
    """Every filing in any NICE class whose goods_services carries the term."""
    parts = []
    for cls in all_classes():
        d = (
            pl.scan_parquet(PROC / f"tm_class{cls}.parquet")
            .select("serial_number", "owner_name", "filing_date", "goods_services")
            .filter(
                pl.col("owner_name").is_not_null()
                & (pl.col("filing_date").str.len_chars() == 8)
                & pl.col("goods_services").str.to_lowercase().str.contains(pattern)
            )
            .select("serial_number", "owner_name", "filing_date")
            .with_columns(pl.lit(cls).alias("nice"))
            .collect()
        )
        if d.height:
            parts.append(d)
    df = pl.concat(parts).unique(subset="serial_number")
    print(f"[term] {term}: {df.height:,} filings, {df['owner_name'].n_unique():,} owners",
          file=sys.stderr, flush=True)
    return df


# ---------------------------------------------------- step 4: IPO from EDGAR ---
def edgar_first_forms(s: requests.Session, ciks: list[int], cache_path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    todo = [c for c in ciks if str(c) not in cache]
    print(f"[edgar] {len(todo):,} CIKs to look up ({len(ciks)-len(todo):,} cached)",
          file=sys.stderr, flush=True)
    for i, cik in enumerate(todo, 1):
        rec: dict[str, str] = {}
        try:
            r = s.get(SUBMISSIONS.format(cik=cik), timeout=60)
            if r.status_code == 200:
                j = r.json()
                rec["_name"] = j.get("name", "")
                blocks = [j["filings"]["recent"]]
                for extra in j["filings"].get("files", []):
                    time.sleep(SLEEP_API)
                    r2 = s.get("https://data.sec.gov/submissions/" + extra["name"], timeout=60)
                    if r2.status_code == 200:
                        blocks.append(r2.json())
                for b in blocks:
                    for f, d in zip(b.get("form", []), b.get("filingDate", [])):
                        if f in IPO_FORMS and (f not in rec or d < rec[f]):
                            rec[f] = d
        except (requests.RequestException, ValueError) as e:
            rec["_error"] = str(e)[:120]
        cache[str(cik)] = rec
        if i % 100 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"[edgar] {i:,}/{len(todo):,}", file=sys.stderr, flush=True)
        time.sleep(SLEEP_API)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return cache


def ipo_date(rec: dict) -> tuple[date | None, str | None]:
    for f in IPO_PRIMARY:
        if rec.get(f):
            return parse_date(rec[f]), f
    for f in IPO_FALLBACK:
        if rec.get(f):
            return parse_date(rec[f]), f
    return None, None


# --------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terms", nargs="+", default=["blockchain", "telehealth"],
                    choices=sorted(TERMS), help="term labels to run")
    ap.add_argument("--start", default="2009q1", help="first Form D quarter (mandatory e-filing 2009q1)")
    ap.add_argument("--end", default=None, help="last Form D quarter")
    ap.add_argument("--max-ciks", type=int, default=2000,
                    help="cap on EDGAR submissions-API lookups per run")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = session()
    t0 = time.time()

    zips = sorted(RAW.glob("*_d.zip")) if args.skip_download else download_formd(s, args.start, args.end)
    fd = parse_formd(zips)

    # operating-company universe: drop pooled funds, keep the primary issuer
    ops = fd.filter(~pl.col("is_pooled") & (pl.col("fund_type") == "") & pl.col("is_primary"))
    print(f"[formd] {ops.height:,} non-fund primary-issuer offerings, "
          f"{ops['norm'].n_unique():,} distinct normalized names", file=sys.stderr, flush=True)

    # first non-amendment offering per normalized issuer name
    first_round = (
        ops.filter(~pl.col("is_amend") & pl.col("round_date").is_not_null())
        .sort("round_date")
        .group_by("norm", maintain_order=True)
        .agg(
            pl.col("cik").first(),
            pl.col("entity_name").first(),
            pl.col("round_date").first().alias("first_round"),
            pl.col("total_sold").first().alias("first_round_sold"),
            pl.col("industry").first(),
            pl.len().alias("n_form_d"),
        )
        .filter(pl.col("norm") != "")
    )
    # a normalized name that resolves to several CIKs is ambiguous; drop it
    amb = (
        ops.group_by("norm").agg(pl.col("cik").n_unique().alias("n_cik"))
        .filter(pl.col("n_cik") > 1)["norm"]
    )
    first_round = first_round.filter(~pl.col("norm").is_in(amb))
    print(f"[formd] {first_round.height:,} unambiguous issuer names "
          f"({len(amb):,} dropped as name collisions)", file=sys.stderr, flush=True)

    debut = owner_debut()
    results: dict[str, dict] = {}
    matched_all: list[pl.DataFrame] = []

    for term in args.terms:
        tf = term_filings(term, TERMS[term])
        by_date = tf.group_by("filing_date").len().sort("filing_date")
        first_ever = by_date["filing_date"][0]
        cum = by_date.with_columns(pl.col("len").cum_sum().alias("cum"))
        sus = cum.filter(pl.col("cum") >= MIN_SUSTAINED)
        sustained = sus["filing_date"][0] if sus.height else None

        owners = (
            tf.group_by("owner_name")
            .agg(pl.col("filing_date").min().alias("tm_first"), pl.len().alias("n_tm"))
            .filter(pl.col("n_tm") >= MIN_OWNER_FILINGS)
            .with_columns(pl.col("owner_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))
            .filter(pl.col("norm") != "")
            .join(debut, on="owner_name", how="left")
            .with_columns(
                (
                    pl.col("tm_first").map_elements(tm_date, return_dtype=pl.Date)
                    - pl.col("debut").map_elements(tm_date, return_dtype=pl.Date)
                ).dt.total_days().alias("days_since_debut")
            )
            .with_columns(
                (pl.col("days_since_debut") <= DEBUT_WINDOW_DAYS)
                .fill_null(False).alias("is_debut_carrier")
            )
        )
        n_debut = int(owners["is_debut_carrier"].sum())
        m = owners.join(first_round, on="norm", how="inner").with_columns(pl.lit(term).alias("term"))
        print(f"[match] {term}: {m.height:,}/{owners.height:,} owners matched to a Form D issuer "
              f"({m.height/max(owners.height,1)*100:.2f}%); "
              f"{n_debut:,} owners are debut carriers, {int(m['is_debut_carrier'].sum()):,} of them matched",
              file=sys.stderr, flush=True)
        matched_all.append(m)
        results[term] = {
            "n_owners_debut_carriers": n_debut,
            "debut_window_days": DEBUT_WINDOW_DAYS,
            "regex": TERMS[term],
            "n_filings": int(tf.height),
            "n_owners": int(owners.height),
            "first_term_filing": first_ever,
            "sustained_emergence": sustained,
            "min_sustained": MIN_SUSTAINED,
            "n_owners_matched_formd": int(m.height),
            "match_rate": round(m.height / max(owners.height, 1), 5),
        }

    matched = pl.concat(matched_all)
    ciks = matched["cik"].unique().to_list()[: args.max_ciks]
    forms = edgar_first_forms(s, ciks, RAW / "edgar_first_forms.json")

    ipo_rows = []
    for cik_str, rec in forms.items():
        d, src = ipo_date(rec)
        if d:
            ipo_rows.append({"cik": int(cik_str), "ipo_date": d, "ipo_src": src,
                             "edgar_name": rec.get("_name", "")})
    ipos = pl.DataFrame(ipo_rows, schema={"cik": pl.Int64, "ipo_date": pl.Date,
                                          "ipo_src": pl.Utf8, "edgar_name": pl.Utf8}) \
        if ipo_rows else pl.DataFrame(schema={"cik": pl.Int64, "ipo_date": pl.Date,
                                              "ipo_src": pl.Utf8, "edgar_name": pl.Utf8})

    panel = (
        matched.with_columns(
            pl.col("tm_first").map_elements(tm_date, return_dtype=pl.Date).alias("tm_date")
        )
        .join(ipos, on="cik", how="left")
        .with_columns(
            ((pl.col("first_round") - pl.col("tm_date")).dt.total_days() / 365.25).alias("lag_tm_fd"),
            ((pl.col("ipo_date") - pl.col("first_round")).dt.total_days() / 365.25).alias("lag_fd_ipo"),
            ((pl.col("ipo_date") - pl.col("tm_date")).dt.total_days() / 365.25).alias("lag_tm_ipo"),
        )
    )
    panel.write_parquet(REPO / "data" / "processed" / "funding_lag_prototype_panel.parquet")

    def stratum_stats(p: pl.DataFrame) -> dict:
        a = p["lag_tm_fd"].drop_nulls().to_list()
        b = p["lag_fd_ipo"].drop_nulls().to_list()
        c = p["lag_tm_ipo"].drop_nulls().to_list()
        return {
            "n_firms": int(p.height),
            "n_with_ipo_marker": int(p["ipo_date"].drop_nulls().len()),
            "median_lag_tm_to_formd_yrs": median(a), "n_lag_tm_to_formd": len(a),
            "median_lag_formd_to_ipo_yrs": median(b), "n_lag_formd_to_ipo": len(b),
            "median_lag_tm_to_ipo_yrs": median(c), "n_lag_tm_to_ipo": len(c),
            "mean_lag_tm_to_formd_yrs": (sum(a) / len(a)) if a else None,
            "mean_lag_formd_to_ipo_yrs": (sum(b) / len(b)) if b else None,
            "mean_lag_tm_to_ipo_yrs": (sum(c) / len(c)) if c else None,
            "ipo_marker_sources": dict(
                p.filter(pl.col("ipo_src").is_not_null()).group_by("ipo_src").len().iter_rows()
            ),
        }

    def cell(v: float | None, n: int) -> str:
        return f"{v:+.2f}y/{n}" if v is not None else "--"

    print("\n" + "=" * 100)
    print(f"{'term':<13}{'stratum':<12}{'matched':>9}{'w/ IPO':>8}{'emerge':>9}"
          f"{'TM->FormD':>15}{'FormD->IPO':>15}{'TM->IPO':>15}")
    print("-" * 100)
    for term in args.terms:
        r = results[term]
        p_all = panel.filter(pl.col("term") == term)
        r["all_carriers"] = stratum_stats(p_all)
        r["debut_carriers"] = stratum_stats(p_all.filter(pl.col("is_debut_carrier")))
        r["incumbent_adopters"] = stratum_stats(p_all.filter(~pl.col("is_debut_carrier")))
        em = (r["sustained_emergence"] or "")[:4]
        for label, key in (("debut", "debut_carriers"), ("incumbent", "incumbent_adopters"),
                           ("all", "all_carriers")):
            st = r[key]
            print(f"{term if label=='debut' else '':<13}{label:<12}{st['n_firms']:>9,}"
                  f"{st['n_with_ipo_marker']:>8,}{(em if label=='debut' else ''):>9}"
                  f"{cell(st['median_lag_tm_to_formd_yrs'], st['n_lag_tm_to_formd']):>15}"
                  f"{cell(st['median_lag_formd_to_ipo_yrs'], st['n_lag_formd_to_ipo']):>15}"
                  f"{cell(st['median_lag_tm_to_ipo_yrs'], st['n_lag_tm_to_ipo']):>15}")
    print("=" * 100)
    print("Medians in years / n completed transitions. Only DEBUT carriers are interpretable:")
    print("for incumbent adopters the term-bearing filing is a mid-life act that postdates")
    print("their financing and often their listing, so their lags run negative by construction.")
    print("Even the debut column is right-censored, survivorship-selected, and conditional on")
    print("a normalized-exact name match that favours large, formally-named entities.")

    meta = {
        "run_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "formd_quarters": len(zips),
        "formd_rows": int(fd.height),
        "formd_operating_offerings": int(ops.height),
        "formd_unambiguous_issuers": int(first_round.height),
        "n_ciks_looked_up": len(ciks),
        "ipo_primary_forms": list(IPO_PRIMARY),
        "ipo_fallback_forms": list(IPO_FALLBACK),
        "elapsed_s": round(time.time() - t0, 1),
        "terms": results,
    }
    OUT.write_text(json.dumps(meta, indent=1, default=str))
    print(f"\n[done] -> {OUT}  ({time.time()-t0:.1f}s)", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
