"""For a curated list of canonical-winner firms (Amazon, Google, Facebook,
…), find their earliest trademark filing across all classes, and check
whether they used category-defining vocabulary at the time.

For each firm, for each candidate category-buzzword T, compute:
  - earliest_filing_date  (in any class, by any matching owner_name variant)
  - earliest_T_filing     (their first filing whose g/s contains T)
  - rank_if_T             (where they would have ranked in the global
                           T-adoption ladder, given earliest_T date)
  - first_gs_excerpt      (first 180 chars of their first filing's g/s)

Owner-name matching is intentionally permissive — it uses a regex over
canonical brand strings so subsidiary/holding/.com variants all count.

Outputs:
  paper/results/big_firm_backcast.csv
  paper/results/big_firm_backcast.txt
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

# (brand, regex over owner_name, primary category buzzword, founded_year)
FIRMS: list[tuple[str, str, str, int]] = [
    ("Amazon",     r"(?i)\bamazon(\.com)?(\s|,|$)",              "online retail",      1994),
    ("eBay",       r"(?i)\bebay\b",                              "online retail",      1995),
    ("Etsy",       r"(?i)\betsy\b",                              "online retail",      2005),
    ("Shopify",    r"(?i)\bshopify\b",                           "online retail",      2006),
    ("Wayfair",    r"(?i)\bwayfair\b",                           "online retail",      2002),
    ("Chewy",      r"(?i)\bchewy(,| inc)",                       "online retail",      2011),

    ("Google",     r"(?i)\bgoogle\b",                            "website",            1998),
    ("Yahoo",      r"(?i)yahoo!?\b",                             "website",            1994),
    ("Netflix",    r"(?i)\bnetflix\b",                           "website",            1997),
    ("Dropbox",    r"(?i)\bdropbox\b",                           "website",            2007),

    ("Facebook",   r"(?i)\b(facebook|meta platforms)\b",         "social media",       2004),
    ("Twitter",    r"(?i)\b(twitter,? inc|twitter\s|x corp)\b",  "social media",       2006),
    ("LinkedIn",   r"(?i)\blinkedin\b",                          "social media",       2002),
    ("Pinterest",  r"(?i)\bpinterest\b",                         "social media",       2010),
    ("Snap",       r"(?i)\b(snap,? inc|snapchat)\b",             "social media",       2011),

    ("Coinbase",   r"(?i)\bcoinbase\b",                          "blockchain",         2012),
    ("Block (Square)", r"(?i)\b(square,? inc|block,? inc)\b",    "blockchain",         2009),
    ("Stripe",     r"(?i)\bstripe,? inc\b",                      "online retail",      2010),

    ("Zoom",       r"(?i)\bzoom video\b",                        "saas",               2011),
    ("Slack",      r"(?i)\bslack technologies\b",                "saas",               2009),
    ("Salesforce", r"(?i)\bsalesforce(\.com)?,?\s*inc\b",        "saas",               1999),
    ("Workday",    r"(?i)\bworkday,? inc\b",                     "saas",               2005),
    ("ServiceNow", r"(?i)\bservicenow\b",                        "saas",               2004),
    ("Snowflake",  r"(?i)\bsnowflake (computing|inc)\b",         "saas",               2012),

    ("Tesla",      r"(?i)\btesla(,?\s*(motors|inc))\b",          "electric vehicle",   2003),
    ("Uber",       r"(?i)\buber technologies\b",                 "mobile app",         2009),
    ("Airbnb",     r"(?i)\bairbnb\b",                            "mobile app",         2008),
    ("Lyft",       r"(?i)\blyft,? inc\b",                        "mobile app",         2012),
    ("Spotify",    r"(?i)\bspotify\b",                           "website",            2006),
    ("Palantir",   r"(?i)\bpalantir\b",                          "saas",               2003),
]

# All candidate buzzwords (we check each firm's first filing against any of these)
BUZZWORDS = [
    "website", "wireless", "online retail", "social media", "blockchain",
    "cloud computing", "saas", "mobile app", "artificial intelligence",
    "machine learning", "streaming", "smartphones", "online", "internet",
    "platform", "electric vehicle", "ride-sharing",
]


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def find_firm_filings(regex: str) -> pl.DataFrame:
    """All filings whose owner_name matches the firm regex."""
    parts = []
    for cls in class_list():
        df = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "filing_date", "owner_name", "goods_services"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("owner_name").str.contains(regex)
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        )
        if df.height:
            parts.append(df.with_columns(pl.lit(cls).alias("nice_class")))
    return pl.concat(parts) if parts else pl.DataFrame()


def adoption_rank_for_token(token: str, target_date: str) -> int | None:
    """Among all firms in the corpus, what was the adoption rank of the
    Nth firm whose first T-mention was on or before target_date?"""
    body = token.replace(" ", r"\s+")
    pat = r"(?i)\b" + body + r"\b"
    parts = []
    for cls in class_list():
        df = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["filing_date", "owner_name", "goods_services"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
            & pl.col("goods_services").is_not_null()
            & pl.col("goods_services").str.contains(pat)
        ).select("filing_date", "owner_name")
        if df.height:
            parts.append(df)
    if not parts:
        return None
    df = pl.concat(parts).group_by("owner_name").agg(
        pl.col("filing_date").min().alias("adopt")
    ).sort("adopt")
    n_before = df.filter(pl.col("adopt") <= target_date).height
    return n_before


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    out_rows = []
    print(f"[backcast] {len(FIRMS)} firms; {len(BUZZWORDS)} candidate buzzwords\n", flush=True)
    for brand, regex, primary_buzz, founded in FIRMS:
        print(f"=== {brand} ===  primary='{primary_buzz}'  founded={founded}", flush=True)
        filings = find_firm_filings(regex)
        if filings.height == 0:
            print(f"  no filings matched\n")
            out_rows.append({"brand": brand, "founded": founded, "primary_buzz": primary_buzz,
                             "n_filings": 0})
            continue
        filings = filings.sort("filing_date")
        first = filings.row(0, named=True)
        first_gs = (first["goods_services"] or "")[:180]
        print(f"  earliest filing: {first['filing_date']} ({first['nice_class']})  "
              f"owner='{first['owner_name']}'\n"
              f"    g/s: {first_gs!r}", flush=True)

        # Did they ever use any buzzword?
        used = {}
        for bz in BUZZWORDS:
            body = bz.replace(" ", r"\s+")
            pat = r"(?i)\b" + body + r"\b"
            sub = filings.filter(pl.col("goods_services").str.contains(pat))
            if sub.height:
                used[bz] = sub.row(0, named=True)["filing_date"]
        if used:
            print(f"  buzzwords used: " + ", ".join(f"{k}={v[:4]}" for k,v in sorted(used.items(), key=lambda x: x[1])), flush=True)
        else:
            print(f"  buzzwords used: none of {len(BUZZWORDS)}", flush=True)

        # For the primary buzzword, compute their adoption rank
        rank_if_T = None
        first_T = used.get(primary_buzz)
        if first_T is not None:
            print(f"  computing adoption rank for '{primary_buzz}' as of {first_T[:4]}…", flush=True)
            rank_if_T = adoption_rank_for_token(primary_buzz, first_T)
            print(f"    rank: {rank_if_T:,}", flush=True)
        else:
            print(f"  NEVER used the category buzzword '{primary_buzz}'", flush=True)

        out_rows.append({
            "brand": brand, "founded": founded, "primary_buzz": primary_buzz,
            "n_filings": filings.height,
            "earliest_filing": first["filing_date"],
            "earliest_owner": first["owner_name"],
            "earliest_class": first["nice_class"],
            "first_gs": first_gs,
            "used_primary_buzz": first_T,
            "rank_in_primary_ladder": rank_if_T,
            "n_buzzwords_ever_used": len(used),
            "buzzwords_used": ", ".join(used.keys()),
        })
        print(flush=True)

    out = pl.from_dicts(out_rows)
    out.write_csv(OUT / "big_firm_backcast.csv")

    # Render readable text table
    lines = ["Big-firm category-buzzword backcast\n",
             f"{'brand':<18s}  {'fnd':>4s}  {'fst':>4s}  {'primary buzz':<18s}  "
             f"{'used?':<8s}  {'rank':>10s}  notes"]
    for r in out_rows:
        used_dt = (r.get('used_primary_buzz') or '')[:4] if r.get('used_primary_buzz') else '·'
        rank = (f"{r['rank_in_primary_ladder']:,}" if r.get('rank_in_primary_ladder') is not None else '·')
        first_y = (r.get('earliest_filing') or '')[:4] if r.get('earliest_filing') else '·'
        lines.append(
            f"{r['brand']:<18s}  {r['founded']:>4d}  {first_y:>4s}  "
            f"{r['primary_buzz']:<18s}  {used_dt:<8s}  {rank:>10s}"
        )
    (OUT / "big_firm_backcast.txt").write_text("\n".join(lines))
    print(f"\n[done] wrote {OUT/'big_firm_backcast.txt'} and .csv", flush=True)
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
