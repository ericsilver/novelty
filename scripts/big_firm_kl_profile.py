"""For each canonical-winner firm, compute their prospective and
retrospective KL profile across all their (post-founding) trademark
filings.  Independent of whether they used category-defining vocabulary.

For each firm we report:
  - n_filings        (post-founding-year, across all classes)
  - mean_pros        firm-mean prospective_kl
  - mean_retr        firm-mean retrospective_kl
  - mean_dkl         firm-mean (pros - retr)
  - first_real_pros  prospective_kl on their first post-founding filing
  - first_real_retr  retrospective_kl on their first post-founding filing
  - first_real_gs    first 140 chars of their first post-founding g/s

Baseline reference values printed at top: median pros and retr across
the entire clean H=2 corpus.

Filters mirror the CLEAN cut used elsewhere:
  n_ref_pros >= 1000, n_ref_retr >= 1000, n_terms >= 3, finite KLs.

Outputs:
  paper/results/big_firm_kl_profile.csv
  paper/results/big_firm_kl_profile.txt
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

# Reuse the same firm regex map as big_firm_backcast.py
FIRMS: list[tuple[str, str, int]] = [
    ("Amazon",         r"(?i)\bamazon(\.com)?(\s|,|$)",              1994),
    ("eBay",           r"(?i)\bebay\b",                              1995),
    ("Etsy",           r"(?i)\betsy\b",                              2005),
    ("Shopify",        r"(?i)\bshopify\b",                           2006),
    ("Wayfair",        r"(?i)\bwayfair\b",                           2002),
    ("Chewy",          r"(?i)\bchewy(,| inc)",                       2011),
    ("Google",         r"(?i)\bgoogle\b",                            1998),
    ("Yahoo",          r"(?i)yahoo!?\b",                             1994),
    ("Netflix",        r"(?i)\bnetflix\b",                           1997),
    ("Dropbox",        r"(?i)\bdropbox\b",                           2007),
    ("Facebook",       r"(?i)\b(facebook|meta platforms)\b",         2004),
    ("Twitter",        r"(?i)\b(twitter,? inc|twitter\s|x corp)\b",  2006),
    ("LinkedIn",       r"(?i)\blinkedin\b",                          2002),
    ("Pinterest",      r"(?i)\bpinterest\b",                         2010),
    ("Snap",           r"(?i)\b(snap,? inc|snapchat)\b",             2011),
    ("Coinbase",       r"(?i)\bcoinbase\b",                          2012),
    ("Block (Square)", r"(?i)\b(square,? inc|block,? inc)\b",        2009),
    ("Stripe",         r"(?i)\bstripe,? inc\b",                      2010),
    ("Zoom",           r"(?i)\bzoom video\b",                        2011),
    ("Slack",          r"(?i)\bslack technologies\b",                2009),
    ("Salesforce",     r"(?i)\bsalesforce(\.com)?,?\s*inc\b",        1999),
    ("Workday",        r"(?i)\bworkday,? inc\b",                     2005),
    ("ServiceNow",     r"(?i)\bservicenow\b",                        2004),
    ("Snowflake",      r"(?i)\bsnowflake (computing|inc)\b",         2012),
    ("Tesla",          r"(?i)\btesla(,?\s*(motors|inc))\b",          2003),
    ("Uber",           r"(?i)\buber technologies\b",                 2009),
    ("Airbnb",         r"(?i)\bairbnb\b",                            2008),
    ("Lyft",           r"(?i)\blyft,? inc\b",                        2012),
    ("Spotify",        r"(?i)\bspotify\b",                           2006),
    ("Palantir",       r"(?i)\bpalantir\b",                          2003),
]


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def baseline_kls() -> tuple[float, float]:
    """Median pros / retr across the whole clean corpus (sample 6 big classes)."""
    parts = []
    for cls in ["009", "025", "035", "041", "042", "036"]:
        sp = PROC / f"surprise_class{cls}.parquet"
        if not sp.exists(): continue
        d = pl.read_parquet(sp, columns=["prospective_kl","retrospective_kl",
                                          "n_ref_prospective","n_ref_retrospective","n_terms"]).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
        ).select("prospective_kl", "retrospective_kl")
        parts.append(d)
    df = pl.concat(parts)
    return float(df["prospective_kl"].median()), float(df["retrospective_kl"].median())


def collect_firm_kls(regex: str, founded: int) -> pl.DataFrame | None:
    """All post-founding filings by this firm, joined to per-filing KL.
    Returns None if no qualifying filings."""
    parts = []
    for cls in class_list():
        tm_p = PROC / f"tm_class{cls}.parquet"
        sp = PROC / f"surprise_class{cls}.parquet"
        if not sp.exists():
            continue
        tm = pl.read_parquet(
            tm_p,
            columns=["serial_number", "filing_date", "owner_name", "goods_services"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("owner_name").str.contains(regex)
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
            & (pl.col("filing_date").str.slice(0, 4).cast(pl.Int32) >= founded)
        ).with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"))
        if tm.height == 0:
            continue
        kl = pl.read_parquet(
            sp,
            columns=["serial_number", "prospective_kl", "retrospective_kl",
                     "n_ref_prospective", "n_ref_retrospective", "n_terms"],
        ).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
        )
        j = tm.join(kl, on="serial_number", how="inner")
        if j.height:
            parts.append(j.with_columns(pl.lit(cls).alias("cls")))
        del tm, kl, j
        gc.collect()
    if not parts:
        return None
    return pl.concat(parts).sort("filing_date")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[baseline] computing corpus medians …", flush=True)
    base_p, base_r = baseline_kls()
    print(f"  corpus median pros={base_p:.3f}  retr={base_r:.3f}  "
          f"(ΔKL baseline={base_p-base_r:+.3f})\n", flush=True)

    rows = []
    for brand, regex, founded in FIRMS:
        print(f"=== {brand} (founded {founded}) ===", flush=True)
        df = collect_firm_kls(regex, founded)
        if df is None or df.height == 0:
            print(f"  no qualifying post-founding clean-KL filings\n", flush=True)
            rows.append({"brand": brand, "founded": founded, "n_clean_filings": 0})
            continue
        first = df.row(0, named=True)
        mp = float(df["prospective_kl"].mean())
        mr = float(df["retrospective_kl"].mean())
        md = mp - mr
        print(f"  n_clean_filings: {df.height:,}",  flush=True)
        print(f"  firm-mean  pros={mp:.3f}  retr={mr:.3f}  ΔKL={md:+.3f}", flush=True)
        print(f"  first-real {first['filing_date']} cls{first['cls']}  "
              f"pros={first['prospective_kl']:.3f}  retr={first['retrospective_kl']:.3f}", flush=True)
        gs = (first["goods_services"] or "")[:140]
        print(f"    g/s: {gs!r}\n", flush=True)
        rows.append({
            "brand": brand, "founded": founded,
            "n_clean_filings": df.height,
            "mean_pros": mp, "mean_retr": mr, "mean_dkl": md,
            "first_real_date": first["filing_date"],
            "first_real_class": first["cls"],
            "first_real_pros": float(first["prospective_kl"]),
            "first_real_retr": float(first["retrospective_kl"]),
            "first_real_owner": first["owner_name"],
            "first_real_gs": gs,
        })

    out = pl.from_dicts(rows)
    out.write_csv(OUT / "big_firm_kl_profile.csv")

    # Formatted table
    lines = [f"Big-firm KL profile  (corpus median pros={base_p:.2f}  retr={base_r:.2f}  ΔKL=0.00)\n",
             f"{'brand':<18s}  {'fnd':>4s}  {'n':>5s}  "
             f"{'mn_p':>6s}  {'mn_r':>6s}  {'ΔKL':>7s}  "
             f"{'1st_p':>6s}  {'1st_r':>6s}  {'1st_yr':>6s}"]
    for r in rows:
        if r["n_clean_filings"] == 0:
            lines.append(f"  {r['brand']:<16s}  {r['founded']:>4d}     0   "
                         f"{'—':>6s}  {'—':>6s}  {'—':>7s}  {'—':>6s}  {'—':>6s}  {'—':>6s}")
            continue
        lines.append(
            f"  {r['brand']:<16s}  {r['founded']:>4d}  {r['n_clean_filings']:>5,}  "
            f"{r['mean_pros']:>6.2f}  {r['mean_retr']:>6.2f}  {r['mean_dkl']:>+7.3f}  "
            f"{r['first_real_pros']:>6.2f}  {r['first_real_retr']:>6.2f}  "
            f"{r['first_real_date'][:4]:>6s}"
        )
    (OUT / "big_firm_kl_profile.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
