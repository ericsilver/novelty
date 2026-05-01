"""Build a USPTO owner_name -> SEC CIK crosswalk by normalized exact match.

We aggressively normalize legal-entity strings on both sides (uppercase, strip
punctuation, drop common suffixes), then exact-match. A second pass adds
SEC's official ticker file as authoritative aliases.

Outputs: data/processed/uspto_sec_crosswalk.parquet
    owner_name      str   USPTO original
    norm            str   normalized key
    cik             int64
    sec_name        str
    n_uspto_filings int   total Class 9+32 filings under this owner
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]

LEGAL_SUFFIX_RE = re.compile(
    r"(?:^|[\s,])(?:"
    r"INC|INCORPORATED|LLC|L\.L\.C\.|LLP|LP|L\.P\.|LTD|LIMITED|CORP|CORPORATION|"
    r"CO|COMPANY|HOLDINGS|HOLDING|GROUP|GMBH|S\.A\.|SA|AG|N\.V\.|NV|PLC|PBC|"
    r"AB|OY|KK|KABUSHIKI KAISHA|S\.R\.L\.|SRL|S\.P\.A\.|SPA|S\.A\.S\.|SAS|"
    r"BV|B\.V\.|PTY|PTE|TRUST|FOUNDATION"
    r")(?:\.|\b)"
)
PUNCT_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize(name: str) -> str:
    if not name:
        return ""
    s = name.upper()
    s = s.replace("&", " AND ")
    # repeatedly strip trailing legal-entity suffix tokens
    for _ in range(4):
        new = LEGAL_SUFFIX_RE.sub(" ", s)
        new = PUNCT_RE.sub(" ", new)
        new = WHITESPACE_RE.sub(" ", new).strip()
        if new == s:
            break
        s = new
    s = s.removesuffix(" THE").removesuffix(" THE,")
    return s.strip()


def main() -> int:
    sec = pl.read_parquet(REPO_ROOT / "data/processed/sec_firm_year.parquet")
    sec_names = (
        sec.group_by(["cik", "name"])
        .agg(pl.col("fy").min().alias("first_fy"), pl.col("fy").max().alias("last_fy"))
    ).with_columns(pl.col("name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))

    tickers = json.load((REPO_ROOT / "data/raw/sec/company_tickers.json").open())
    ticker_rows = []
    for v in tickers.values():
        ticker_rows.append({"cik": int(v["cik_str"]), "name": v["title"], "norm": normalize(v["title"])})
    tk = pl.DataFrame(ticker_rows)

    sec_lookup = pl.concat(
        [
            sec_names.select("cik", "name", "norm").rename({"name": "sec_name"}),
            tk.rename({"name": "sec_name"}),
        ]
    ).filter(pl.col("norm") != "").unique(["cik", "norm"])

    parts = []
    for cls in ("009", "032"):
        fy = pl.read_parquet(REPO_ROOT / f"data/processed/firm_year_class{cls}.parquet")
        parts.append(
            fy.group_by("owner_name").agg(
                pl.col("n_filings").sum().alias("n_filings"),
                pl.col("year").min().alias("first_year"),
                pl.col("year").max().alias("last_year"),
            ).with_columns(pl.lit(cls).alias("nice_class"))
        )
    owners = (
        pl.concat(parts)
        .group_by("owner_name")
        .agg(
            pl.col("n_filings").sum().alias("n_filings"),
            pl.col("first_year").min().alias("first_year"),
            pl.col("last_year").max().alias("last_year"),
        )
        .filter(pl.col("owner_name").is_not_null())
        .with_columns(pl.col("owner_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))
    )

    joined = owners.join(sec_lookup, on="norm", how="inner")
    joined = (
        joined.sort(["norm", "n_filings"], descending=[False, True])
        .group_by("norm", maintain_order=True)
        .agg(
            pl.col("owner_name").first(),
            pl.col("cik").first(),
            pl.col("sec_name").first(),
            pl.col("n_filings").first(),
            pl.col("first_year").first(),
            pl.col("last_year").first(),
        )
    )

    out = REPO_ROOT / "data/processed/uspto_sec_crosswalk.parquet"
    joined.write_parquet(out)
    n_owners = owners.height
    n_match = joined["owner_name"].n_unique()
    n_filings_match = int(joined["n_filings"].sum())
    n_filings_total = int(owners["n_filings"].sum())
    print(
        f"[done] {n_match:,} matches / {n_owners:,} owners "
        f"({n_match / n_owners * 100:.1f}%); "
        f"covers {n_filings_match:,} / {n_filings_total:,} filings "
        f"({n_filings_match / n_filings_total * 100:.1f}%)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
