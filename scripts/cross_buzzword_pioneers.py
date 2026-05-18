"""Are there owners (firms or humans) who showed up as early adopters
across MULTIPLE category-defining buzzwords?  If a founder really does
'fail forward,' you'd see them in the pioneer cohort of one category
and then again in a later one.

For each buzzword T, compute the set of owners ranked ≤1000 in the
T-adoption ladder (the 'sweet-spot' early-adopter pool from the prior
analysis).  Then for every pair (T1, T2), report overlap.

Also tag owners by whether their owner_name looks like an individual
human (no LLC/INC/CORP/etc. suffix) — these are the rough proxy for
'founder filed personally'.

Outputs:
  paper/results/cross_buzzword_pioneers.txt
  paper/results/cross_buzzword_overlap.csv
  paper/results/cross_buzzword_individual_serial.txt
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

BUZZWORDS = [
    "website", "wireless", "online retail", "social media",
    "blockchain", "cloud computing", "artificial intelligence",
    "mobile app", "streaming", "saas",
]
EARLY_RANK_CUTOFF = 1000

# Heuristic: owner_name looks like an individual if it has none of these business suffixes
BIZ_SUFFIX = re.compile(
    r"\b(INC|LLC|L\.L\.C\.|LTD|LIMITED|CORP|CORPORATION|CO\.?|COMPANY|GMBH|AG|S\.A\.|SA|"
    r"SAS|PLC|SE|GROUP|HOLDINGS|FOUNDATION|ASSOCIATION|TRUST|PARTNERS|PARTNERSHIP|"
    r"ENTERPRISES|VENTURES|CAPITAL|FUND|BANK|CONSORTIUM|UNIVERSITY|COLLEGE|INSTITUTE|"
    r"DEPARTMENT|MINISTRY|AUTHORITY|COMMISSION|BUREAU|AGENCY|ADMINISTRATION|"
    r"SOCIETY|GROUP|ALLIANCE|CONSORTIUM|COOPERATIVE|UNION|FEDERATION|GUILD|"
    r"LP|LLP|PLLC|PC|PA|N\.A\.|N\.V\.|S\.R\.L\.|SRL|BV|OY|AB|AS|KG|OHG|"
    r"BREWERIES?|MILLS?|WORKS?|PUBLISHING|PUBLICATIONS|MEDIA|ENTERTAINMENT|"
    r"PRODUCTIONS|STUDIOS|GAMES|TECHNOLOGIES|TECHNOLOGY|SYSTEMS|SOLUTIONS|"
    r"SERVICES|INTERNATIONAL|GLOBAL|WORLDWIDE|HOLDINGS?|ASSOCIATES|GROUP|FAMILY)\b",
    re.IGNORECASE,
)


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def early_adopter_owners(token: str) -> pl.DataFrame:
    """All firms that ever filed something containing T, ranked by their
    first T-mention date.  Return only the top-EARLY_RANK_CUTOFF."""
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
        del df
        gc.collect()
    if not parts:
        return pl.DataFrame()
    return (pl.concat(parts)
              .group_by("owner_name")
              .agg(pl.col("filing_date").min().alias("first_T_date"))
              .sort("first_T_date")
              .head(EARLY_RANK_CUTOFF)
              .with_row_index("rank", offset=1))


def is_individual(owner: str) -> bool:
    if not owner:
        return False
    if BIZ_SUFFIX.search(owner):
        return False
    # Few words, no all-caps shouting brand, no '&' (typical brand sep)
    words = owner.split()
    if not 2 <= len(words) <= 4:
        return False
    if "&" in owner:
        return False
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pools: dict[str, pl.DataFrame] = {}
    for bz in BUZZWORDS:
        print(f"[pool] {bz} …", flush=True)
        pools[bz] = early_adopter_owners(bz)
        print(f"        {pools[bz].height:,} early adopters (cutoff {EARLY_RANK_CUTOFF})", flush=True)

    # Pairwise overlap
    bz_list = list(pools.keys())
    overlap_rows = []
    for i, t1 in enumerate(bz_list):
        for t2 in bz_list[i+1:]:
            s1 = set(pools[t1]["owner_name"].to_list())
            s2 = set(pools[t2]["owner_name"].to_list())
            inter = s1 & s2
            overlap_rows.append({
                "t1": t1, "t2": t2,
                "n_t1": len(s1), "n_t2": len(s2),
                "overlap": len(inter),
                "examples": "; ".join(sorted(inter)[:6]),
            })
    overlap = pl.from_dicts(overlap_rows).sort("overlap", descending=True)
    overlap.write_csv(OUT / "cross_buzzword_overlap.csv")
    print("\n=== pairwise overlap (top 20) ===", flush=True)
    for r in overlap.head(20).iter_rows(named=True):
        print(f"  {r['t1']:<22s} ∩ {r['t2']:<22s}  overlap={r['overlap']:>4d}  "
              f"(of {r['n_t1']:,} × {r['n_t2']:,})  e.g. {r['examples'][:90]}")

    # Owners that appear in K ≥ 3 buzzword pools
    own_count: dict[str, list[tuple[str, int, str]]] = {}
    for bz, df in pools.items():
        for r in df.iter_rows(named=True):
            own_count.setdefault(r["owner_name"], []).append(
                (bz, int(r["rank"]), str(r["first_T_date"]))
            )
    multi = {o: ds for o, ds in own_count.items() if len(ds) >= 3}
    multi_sorted = sorted(multi.items(), key=lambda kv: -len(kv[1]))

    lines = [f"Owners appearing in ≥3 different buzzword pioneer pools "
             f"(rank ≤{EARLY_RANK_CUTOFF})\n",
             f"{'k':>3s}  {'owner':<54s}  buzzwords (rank, year)"]
    for owner, ds in multi_sorted[:80]:
        ds_str = ", ".join(f"{b}(r{r},y{d[:4]})" for b, r, d in sorted(ds, key=lambda x: x[2]))
        lines.append(f"  {len(ds):>2d}  {owner[:52]:<54s}  {ds_str[:170]}")
    txt = "\n".join(lines)
    (OUT / "cross_buzzword_pioneers.txt").write_text(txt)
    print(f"\n=== owners in ≥3 pools: {len(multi):,} firms ===")
    print(txt[:4000])

    # Individual-owner subset
    ind_multi = {o: ds for o, ds in multi.items() if is_individual(o)}
    ind_sorted = sorted(ind_multi.items(), key=lambda kv: -len(kv[1]))
    ilines = [f"INDIVIDUAL-LOOKING owners appearing in ≥3 buzzword pioneer pools\n",
              f"{'k':>3s}  {'owner':<54s}  buzzwords (rank, year)"]
    for owner, ds in ind_sorted[:80]:
        ds_str = ", ".join(f"{b}(r{r},y{d[:4]})" for b, r, d in sorted(ds, key=lambda x: x[2]))
        ilines.append(f"  {len(ds):>2d}  {owner[:52]:<54s}  {ds_str[:170]}")
    itxt = "\n".join(ilines)
    (OUT / "cross_buzzword_individual_serial.txt").write_text(itxt)
    print(f"\n=== individual owners in ≥3 pools: {len(ind_multi):,} humans ===")
    print(itxt[:3500])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
