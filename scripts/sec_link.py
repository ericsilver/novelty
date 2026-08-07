"""Build a USPTO owner_name -> SEC CIK crosswalk by normalized exact match.

Both sides are aggressively normalized (uppercase, strip punctuation, drop
common legal-entity suffixes) and then exact-matched. SEC's own company-ticker
file is folded in as authoritative aliases alongside the Financial Statement
Data Sets names.

REBUILT 2026-08-07. The previous version had three defects, all of which
shrank and distorted coverage:

  1. The owner universe was assembled by globbing firm_year_class*.parquet,
     and only five of those tables exist (009, 032, 035, 039, 042). An owner
     filing solely in, say, clothing could not be matched at all. That put
     2,040,349 owners in the universe against 5,674,909 in the corpus, so 57%
     of the debut panel was coded non-listed BY CONSTRUCTION -- and, worse,
     eligibility was correlated with the text being scored (top-atypicality
     quintile 47.3% eligible against 40.3% in the bottom). The universe is now
     every owner in all 45 class files.

  2. The exact-match stage did group_by("norm").first(), keeping ONE owner
     string per normalized key. 652,353 norms carry more than one owner string,
     so 944,309 owner strings were silently dropped even when they normalized
     onto a matched SEC name. Matches are now kept per owner_name; several
     owner strings may legitimately share a CIK.

  3. The fuzzy pass has never run: rapidfuzz is not installed in the analysis
     environment, so the try/except fell through to a no-op and every shipped
     crosswalk was 100% exact. Downstream scripts nonetheless reasoned about
     censoring induced by its >=10-filing threshold. The pass is retained but
     now reports loudly that it did nothing, rather than failing silently.

One deliberate tightening accompanies the expansion: normalized names that
resolve to more than one CIK (263 of 20,101 SEC norms) are dropped rather than
arbitrarily assigned, because widening the candidate pool from 2.0M to 5.7M
owners raises the cost of a false positive.

What this does NOT fix: the outcome remains "an owner's normalized name exactly
matches an SEC registrant's". Firms with distinctive names are more matchable
than firms with generic ones, and distinctiveness of the name plausibly
correlates with atypicality of the goods/services text. Expanding the universe
removes the class-coverage artifact; it does not remove that one.

Reads   data/processed/tm_class{001..045}.parquet   owner names, filing dates
        data/processed/sec_firm_year.parquet        FSDS company names by CIK
        data/raw/sec/company_tickers.json           SEC's own alias file
Writes  data/processed/uspto_sec_crosswalk.parquet
    owner_name      str   USPTO original
    norm            str   normalized key
    cik             int64
    sec_name        str
    n_filings       int   unique serials under this owner, all classes
    first_year      int
    last_year       int
    match_type      str   "exact" (or "fuzzy", never populated in practice)
    match_score     float
"""

from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
CLASSES = [f"{i:03d}" for i in range(1, 46)]

LEGAL_SUFFIX_RE = re.compile(
    r"(?:^|[\s,])(?:"
    r"INC|INCORPORATED|LLC|L\.L\.C\.|LLP|LP|L\.P\.|LTD|LIMITED|CORP|CORPORATION|"
    r"CO|COMPANY|GMBH|S\.A\.|SA|AG|N\.V\.|NV|PLC|PBC|"
    r"AB|OY|KK|KABUSHIKI KAISHA|S\.R\.L\.|SRL|S\.P\.A\.|SPA|S\.A\.S\.|SAS|"
    r"BV|B\.V\.|PTY|PTE"
    r")(?:\.|\b)"
)
PUNCT_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def normalize(name: str) -> str:
    if not name:
        return ""
    s = name.upper()
    s = s.replace("&", " AND ")
    # Repeatedly strip legal-entity suffixes: "ACME HOLDINGS CORP LTD" needs
    # more than one pass, and each pass can expose a new trailing suffix.
    for _ in range(4):
        new = LEGAL_SUFFIX_RE.sub(" ", s)
        new = PUNCT_RE.sub(" ", new)
        new = WHITESPACE_RE.sub(" ", new).strip()
        if new == s:
            break
        s = new
    s = s.removesuffix(" THE").removesuffix(" THE,")
    # USPTO writes "The Coca-Cola Company"; SEC writes "COCA COLA CO". A leading
    # article is never identifying, and leaving it in cost both Coca-Cola and
    # Boeing their match.
    s = s.removeprefix("THE ")
    return s.strip()


def owner_universe() -> pl.DataFrame:
    """Every owner in the corpus, with filing counts deduped across classes."""
    parts = []
    for cls in CLASSES:
        p = PROC / f"tm_class{cls}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(
            p, columns=["serial_number", "owner_name", "filing_date"]
        ).filter(
            pl.col("owner_name").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        ).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("y")
        ).select("serial_number", "owner_name", "y")
        parts.append(d)
        del d
        gc.collect()

    # A multi-class filing appears once per class; dedupe on serial so
    # n_filings counts applications, not class-rows.
    allf = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    owners = allf.group_by("owner_name").agg(
        pl.len().alias("n_filings"),
        pl.col("y").min().alias("first_year"),
        pl.col("y").max().alias("last_year"),
    ).filter(pl.col("owner_name").is_not_null())
    log(f"[universe] {owners.height:,} distinct owners across all 45 classes "
        f"({allf.height:,} unique serials)")
    return owners.with_columns(
        pl.col("owner_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))


def sec_side() -> pl.DataFrame:
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet")
    sec_names = sec.group_by(["cik", "name"]).agg(
        pl.col("fy").min().alias("first_fy"), pl.col("fy").max().alias("last_fy")
    ).with_columns(
        pl.col("name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))

    tk_path = REPO_ROOT / "data/raw/sec/company_tickers.json"
    if not tk_path.exists():
        raise SystemExit(
            "missing data/raw/sec/company_tickers.json -- fetch it with\n"
            "  curl -A '<name> <email>' -o data/raw/sec/company_tickers.json \\\n"
            "       https://www.sec.gov/files/company_tickers.json")
    tickers = json.load(tk_path.open())
    tk = pl.DataFrame([
        {"cik": int(v["cik_str"]), "sec_name": v["title"], "norm": normalize(v["title"])}
        for v in tickers.values()
    ])

    lookup = pl.concat([
        sec_names.select("cik", pl.col("name").alias("sec_name"), "norm"),
        tk,
    ]).filter(pl.col("norm") != "").unique(["cik", "norm"])

    # Drop normalized names that resolve to more than one CIK. With a 5.7M-owner
    # candidate pool an arbitrary pick is a false positive, not a coin flip.
    amb = lookup.group_by("norm").agg(
        pl.col("cik").n_unique().alias("n_cik")).filter(pl.col("n_cik") > 1)
    lookup = lookup.join(amb.select("norm"), on="norm", how="anti")
    log(f"[sec] {lookup.height:,} (cik, norm) pairs over "
        f"{lookup['norm'].n_unique():,} unambiguous norms; "
        f"dropped {amb.height:,} norms resolving to multiple CIKs")
    return lookup.unique(subset="norm")


def main() -> int:
    owners = owner_universe()
    lookup = sec_side()

    # Guard against degenerate keys before joining. HOLDINGS / GROUP / TRUST /
    # FOUNDATION are deliberately NOT stripped above -- they are substantive
    # words, and stripping them made "Q, LLC" collide with "Q HOLDINGS, INC."
    # With them retained, only keys of one or two characters are too weak to
    # carry identity. An earlier version dropped every single-token key shorter
    # than six characters, which removed APPLE, NIKE, TESLA and INTEL -- i.e.
    # precisely the large listed firms the outcome is meant to detect.
    short_key = pl.col("norm").str.len_chars() < 3
    n_before = owners.height
    owners = owners.filter(~short_key)
    lookup = lookup.filter(~short_key)
    log(f"[guard] dropped degenerate short keys: {n_before - owners.height:,} owners, "
        f"leaving {owners.height:,}")

    # One row per OWNER STRING. Distinct owner strings that normalize together
    # legitimately share a CIK; collapsing them loses matches.
    exact = owners.join(lookup, on="norm", how="inner").with_columns(
        pl.lit("exact").alias("match_type"), pl.lit(100.0).alias("match_score"))
    log(f"[exact] {exact.height:,} owner strings matched "
        f"({exact['cik'].n_unique():,} distinct CIKs)")

    fuzzy_candidates = owners.join(
        exact.select("owner_name"), on="owner_name", how="anti"
    ).filter(pl.col("n_filings") >= 10)
    fuzzy_rows: list[dict] = []
    try:
        from rapidfuzz import process, fuzz  # noqa: F401
    except ImportError:
        log(f"[fuzzy] SKIPPED -- rapidfuzz is not installed. "
            f"{fuzzy_candidates.height:,} unmatched owners with >=10 filings were "
            f"NOT considered. Every shipped crosswalk to date has been 100% exact; "
            f"downstream code that reasons about fuzzy-pass censoring is describing "
            f"a stage that has never run.")

    cols = ["owner_name", "norm", "cik", "sec_name", "n_filings",
            "first_year", "last_year", "match_type", "match_score"]
    casts = [pl.col("n_filings").cast(pl.Int64),
             pl.col("first_year").cast(pl.Int64),
             pl.col("last_year").cast(pl.Int64)]
    out_df = exact.select(cols).with_columns(casts)
    if fuzzy_rows:
        out_df = pl.concat([out_df, pl.DataFrame(fuzzy_rows).select(cols).with_columns(casts)])

    dest = PROC / "uspto_sec_crosswalk.parquet"
    out_df.write_parquet(dest)

    n_owners = owners.height
    n_match = out_df["owner_name"].n_unique()
    log(f"[done] {n_match:,} matched owner strings / {n_owners:,} owners "
        f"({n_match / n_owners * 100:.2f}%), {out_df['cik'].n_unique():,} CIKs; "
        f"covers {int(out_df['n_filings'].sum()):,} of "
        f"{int(owners['n_filings'].sum()):,} filings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
