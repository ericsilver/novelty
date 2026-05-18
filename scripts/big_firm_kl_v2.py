"""Big-firm KL profile, v2:
  - Tightened owner_name regexes to suppress false-match historical firms
    (e.g. AMAZON COTTON MILLS, the 1924 wireless filers, etc.)
  - Extended canonical-winner list (now ~60 firms across eras)
  - Added a 'failed firm' negative control set (Pets.com, Webvan, Theranos,
    MoviePass, Quibi, WeWork, Enron, Lehman, FTX, Solyndra, Better.com,
    Bird, Casper, Peloton-as-pivoter, etc.)

For each firm:
  cohort        ('winner' or 'failure')
  founded_year  approximate
  n_filings     post-founding clean H=2y filings
  mean_pros, mean_retr, mean_dkl
  first_real_*  (date / KL values / g/s excerpt)

Outputs:
  paper/results/big_firm_kl_v2.csv
  paper/results/big_firm_kl_v2.txt
"""

from __future__ import annotations

import gc
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

# (cohort, brand, regex, founded_year)
# Tightened: brand-string is required to appear with INC/LLC/.com/CORP/etc. suffix
# OR the brand alone if the brand is sufficiently unique
FIRMS: list[tuple[str, str, str, int]] = [
    # ---- winners ----
    # Amazon: restrict to "Amazon.com" variants. "Amazon Technologies, Inc."
    # was used as a registered owner by a different firm before Amazon's IP
    # subsidiary of that name was created (~2001), and the trademark XML does
    # not record an entity ID that would distinguish them.  Using Amazon.com
    # variants loses some post-2001 IP-holder filings but is unambiguous.
    ("winner", "Amazon",       r"(?i)\bamazon\.com\b",                              1994),
    ("winner", "eBay",         r"(?i)\bebay\s+inc\b|\bebay,\s*inc\b",                1995),
    ("winner", "Etsy",         r"(?i)\betsy,?\s*inc\b",                              2005),
    ("winner", "Shopify",      r"(?i)\bshopify\b",                                   2006),
    ("winner", "Wayfair",      r"(?i)\bwayfair(\.com|\s+llc|,?\s*inc)\b",            2002),
    ("winner", "Chewy",        r"(?i)\bchewy,\s*inc\b",                              2011),
    ("winner", "Google",       r"(?i)\bgoogle\s+(inc|llc)\b",                        1998),
    ("winner", "Yahoo",        r"(?i)yahoo!?\s*inc\b",                               1994),
    ("winner", "Netflix",      r"(?i)\bnetflix,?\s*inc\b",                           1997),
    ("winner", "Dropbox",      r"(?i)\bdropbox,?\s*inc\b",                           2007),
    ("winner", "Facebook/Meta",r"(?i)\bfacebook,?\s*inc\b|\bmeta\s+platforms,?\s*inc\b", 2004),
    ("winner", "Twitter/X",    r"(?i)\btwitter,?\s*inc\b|\bx\s+corp\b",              2006),
    ("winner", "LinkedIn",     r"(?i)\blinkedin\s+corporation\b|\blinkedin\s+ireland\b|\blinkedin\b", 2002),
    ("winner", "Pinterest",    r"(?i)\bpinterest,?\s*inc\b",                         2010),
    ("winner", "Snap",         r"(?i)\bsnap,?\s*inc\b|\bsnapchat\b",                 2011),
    ("winner", "Coinbase",     r"(?i)\bcoinbase\b",                                  2012),
    ("winner", "Block/Square", r"(?i)\bsquare,?\s*inc\b|\bblock,?\s*inc\b",          2009),
    ("winner", "Stripe",       r"(?i)\bstripe,?\s*inc\b",                            2010),
    ("winner", "Zoom",         r"(?i)\bzoom\s+video\s+communications\b",             2011),
    ("winner", "Slack",        r"(?i)\bslack\s+technologies\b",                      2009),
    ("winner", "Salesforce",   r"(?i)\bsalesforce(\.com)?,?\s*inc\b",                1999),
    ("winner", "Workday",      r"(?i)\bworkday,?\s*inc\b",                           2005),
    ("winner", "ServiceNow",   r"(?i)\bservicenow,?\s*inc\b|\bservicenow\b",         2004),
    ("winner", "Snowflake",    r"(?i)\bsnowflake\s+(computing|inc)\b",               2012),
    ("winner", "Tesla",        r"(?i)\btesla\s+(motors|inc)\b|\btesla,\s*inc\b",     2003),
    ("winner", "Uber",         r"(?i)\buber\s+technologies\b",                       2009),
    ("winner", "Airbnb",       r"(?i)\bairbnb,?\s*inc\b",                            2008),
    ("winner", "Lyft",         r"(?i)\blyft,?\s*inc\b",                              2012),
    ("winner", "Spotify",      r"(?i)\bspotify\s+(usa|ab)\b",                        2006),
    ("winner", "Palantir",     r"(?i)\bpalantir\s+technologies\b",                   2003),
    # Older incumbents
    ("winner", "Cisco",        r"(?i)\bcisco\s+(systems|technology)\b",              1984),
    ("winner", "Oracle",       r"(?i)\boracle\s+(corporation|america|international)\b", 1977),
    ("winner", "SAP",          r"(?i)\bsap\s+(ag|se|america)\b",                     1972),
    ("winner", "NVIDIA",       r"(?i)\bnvidia\s+corporation\b",                      1993),
    ("winner", "AMD",          r"(?i)\badvanced\s+micro\s+devices\b",                1969),
    ("winner", "Adobe",        r"(?i)\badobe\s+(systems|inc)\b",                     1982),
    ("winner", "Intuit",       r"(?i)\bintuit\s+inc\b",                              1983),
    ("winner", "Apple",        r"(?i)\bapple\s+(inc|computer)\b",                    1976),
    ("winner", "Microsoft",    r"(?i)\bmicrosoft\s+corporation\b",                   1975),
    ("winner", "Intel",        r"(?i)\bintel\s+corporation\b",                       1968),
    ("winner", "Qualcomm",     r"(?i)\bqualcomm\s+(incorporated|inc)\b",             1985),

    # ---- failures (negative control) ----
    ("failure", "Pets.com",    r"(?i)\bpets\.com\b",                                 1998),
    ("failure", "Webvan",      r"(?i)\bwebvan\b",                                    1996),
    ("failure", "Theranos",    r"(?i)\btheranos\b",                                  2003),
    ("failure", "MoviePass",   r"(?i)\bmoviepass\b",                                 2011),
    ("failure", "Quibi",       r"(?i)\bquibi\b",                                     2018),
    ("failure", "WeWork",      r"(?i)\bwework\b|\bthe\s+we\s+company\b",             2010),
    ("failure", "FTX",         r"(?i)\bftx\s+(trading|us|exchange)\b",               2019),
    ("failure", "Solyndra",    r"(?i)\bsolyndra\b",                                  2005),
    ("failure", "Bird",        r"(?i)\bbird\s+rides\b|\bbird\s+global\b",            2017),
    ("failure", "Casper",      r"(?i)\bcasper\s+sleep\b",                            2014),
    ("failure", "Better.com",  r"(?i)\bbetter\s+holdco\b|\bbetter\.com\b",           2014),
    ("failure", "Lordstown",   r"(?i)\blordstown\s+motors\b",                        2018),
    ("failure", "Nikola",      r"(?i)\bnikola\s+(corporation|motor)\b",              2014),
    ("failure", "Beyond Meat", r"(?i)\bbeyond\s+meat,?\s*inc\b",                     2009),  # alive but distressed
    ("failure", "Peloton",     r"(?i)\bpeloton\s+interactive\b",                     2012),  # alive but post-IPO collapsed
    ("failure", "Pets.com (old)", r"(?i)\bpets\s+at\s+home\s+inc\b",                 1998),  # alt match
    ("failure", "Friendster",  r"(?i)\bfriendster\b",                                2002),
    ("failure", "Myspace",     r"(?i)\bmyspace\b",                                   2003),
    ("failure", "Vine",        r"(?i)\bvine\s+labs\b",                               2012),
    ("failure", "Juicero",     r"(?i)\bjuicero\b",                                   2013),
]


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def baseline_kls() -> tuple[float, float]:
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
    print("[baseline] …", flush=True)
    base_p, base_r = baseline_kls()
    print(f"  corpus median pros={base_p:.3f}  retr={base_r:.3f}\n", flush=True)

    rows = []
    for cohort, brand, regex, founded in FIRMS:
        print(f"=== [{cohort}] {brand} (founded {founded}) ===", flush=True)
        df = collect_firm_kls(regex, founded)
        if df is None or df.height == 0:
            print(f"  no qualifying filings\n", flush=True)
            rows.append({"cohort": cohort, "brand": brand, "founded": founded,
                         "n_clean_filings": 0})
            continue
        first = df.row(0, named=True)
        mp = float(df["prospective_kl"].mean())
        mr = float(df["retrospective_kl"].mean())
        md = mp - mr
        first_dkl = float(first["prospective_kl"]) - float(first["retrospective_kl"])
        print(f"  n={df.height:,}  mean pros={mp:.3f} retr={mr:.3f} ΔKL={md:+.3f}",
              flush=True)
        print(f"  first {first['filing_date']} cls{first['cls']} "
              f"pros={first['prospective_kl']:.2f} retr={first['retrospective_kl']:.2f} "
              f"ΔKL={first_dkl:+.2f}", flush=True)
        gs = (first["goods_services"] or "")[:120]
        print(f"    g/s: {gs!r}\n", flush=True)
        rows.append({
            "cohort": cohort, "brand": brand, "founded": founded,
            "n_clean_filings": df.height,
            "mean_pros": mp, "mean_retr": mr, "mean_dkl": md,
            "first_real_date": first["filing_date"],
            "first_real_class": first["cls"],
            "first_real_pros": float(first["prospective_kl"]),
            "first_real_retr": float(first["retrospective_kl"]),
            "first_real_dkl":  first_dkl,
            "first_real_owner": first["owner_name"],
            "first_real_gs": gs,
        })

    out = pl.from_dicts(rows)
    out.write_csv(OUT / "big_firm_kl_v2.csv")

    # Formatted text
    lines = [f"Big-firm KL profile v2  (corpus median pros={base_p:.2f} retr={base_r:.2f})\n"]
    for label, coh in [("=== WINNERS ===", "winner"), ("\n=== FAILURES ===", "failure")]:
        lines.append(label)
        lines.append(f"{'brand':<18s}  {'fnd':>4s}  {'n':>5s}  "
                     f"{'mn_p':>5s}  {'mn_r':>5s}  {'ΔKL':>6s}  "
                     f"{'1st_p':>5s}  {'1st_r':>5s}  {'1stΔ':>6s}  {'1st_yr':>6s}")
        for r in rows:
            if r["cohort"] != coh: continue
            if r["n_clean_filings"] == 0:
                lines.append(f"  {r['brand']:<16s}  {r['founded']:>4d}     0    —      —      —       —      —      —       —")
                continue
            lines.append(
                f"  {r['brand']:<16s}  {r['founded']:>4d}  {r['n_clean_filings']:>5,}  "
                f"{r['mean_pros']:>5.2f}  {r['mean_retr']:>5.2f}  {r['mean_dkl']:>+6.3f}  "
                f"{r['first_real_pros']:>5.2f}  {r['first_real_retr']:>5.2f}  "
                f"{r['first_real_dkl']:>+6.2f}  {r['first_real_date'][:4]:>6s}"
            )
    (OUT / "big_firm_kl_v2.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
