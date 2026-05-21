"""Apply Census 2010 surname → race/ethnicity probabilities to USPTO
trademark debut filers, where the owner parses as an individual person
(not an LLC/Corp/etc.).

Caveats acknowledged in the writeup:
- Restricted to ~individual-name filings (~20-30% of debuts); LLC haven
  states (DE/WY/NV) mask most individual founders.
- Census surname accuracy is high for non-Hispanic White, Black, and
  Hispanic; OK for Asian/Pacific Islander; very poor for South Asian,
  Middle Eastern, and African names (the Census categories don't break
  these out separately).
- Trademark filings under-represent businesses with low branding
  intensity (small services, immigrant-owned restaurants) -- selection,
  not detection.

Method:
1. Identify "personal-name-shaped" owner names: 2-4 alpha tokens, no
   entity markers (LLC, INC, CORP, etc.), no .com/.io/punctuation that
   indicates a brand string.
2. Extract surname: last token after stripping suffixes (JR, SR, II,
   III, MD, etc.).
3. Join to Names_2010Census.csv on surname; weight by Census race
   probabilities.
4. Tabulate ethnic-share by year and by NICE class on first-ever
   filings.

Outputs:
  paper/results/surname_ethnicity_panel.csv         (year x bucket counts)
  paper/results/surname_ethnicity_by_class.csv      (class x bucket shares)
  paper/results/surname_ethnicity.png               (time-series)
  paper/results/surname_ethnicity.txt               (summary)
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
# Census surname file is a small public reference dataset; we keep a copy
# in the repo under reference_data/ so this script is reproducible without
# requiring the user to re-download from census.gov.
RAW = REPO / "reference_data"
OUT = REPO / "paper" / "results"

# Entity-name markers. If any of these appear (as a whole token or with
# common punctuation), the filing is treated as a corporate entity and
# excluded from surname classification.
ENTITY_MARKERS = {
    "LLC", "L.L.C.", "LL.C.", "L.L.C",
    "INC", "INC.", "INCORPORATED",
    "CORP", "CORP.", "CORPORATION",
    "CO", "CO.", "COMPANY",
    "LTD", "LTD.", "LIMITED",
    "LP", "L.P.", "LLP", "L.L.P.",
    "PLC", "P.L.C.",
    "GMBH", "AG", "NV", "N.V.", "BV", "B.V.", "SA", "S.A.", "SAS",
    "OY", "AB", "KG", "S.R.L.", "SRL", "S.A.S.",
    "TRUST", "FOUNDATION", "FUND",
    "ASSOCIATION", "ASSN", "ASSN.",
    "SOCIETY", "INSTITUTE", "INSTITUTION",
    "GROUP", "HOLDINGS", "ENTERPRISES", "INDUSTRIES", "BRANDS",
    "INTERNATIONAL", "WORLDWIDE", "GLOBAL",
    "UNIVERSITY", "COLLEGE", "SCHOOL", "ACADEMY",
    "HOSPITAL", "CLINIC", "MEDICAL",
    "CHURCH", "MINISTRIES", "PARISH", "DIOCESE",
    "BANK", "INSURANCE", "CAPITAL", "VENTURES", "PARTNERS",
    "TECHNOLOGIES", "TECHNOLOGY", "SOLUTIONS", "SYSTEMS",
    "SERVICES", "PRODUCTS", "PRODUCTIONS",
    "STUDIOS", "MEDIA", "ENTERTAINMENT",
    "RESTAURANTS", "FOODS",
    "AGENCY", "BUREAU", "DEPARTMENT", "GOVERNMENT",
    "USA", "U.S.A.", "U.S.", "AMERICA",
    "OF", "AND", "AND/OR",  # filtered out before suffix check, but flagged for entity-like patterns
}

SUFFIX_TOKENS = {"JR", "JR.", "SR", "SR.", "II", "III", "IV", "V",
                 "ESQ", "ESQ.", "MD", "M.D.", "DO", "D.O.", "PHD", "PH.D.",
                 "DDS", "D.D.S.", "JD", "J.D.", "CPA", "C.P.A."}


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def is_entity(name: str) -> bool:
    """Heuristic: name contains a corporate/institutional marker."""
    upper = name.upper()
    # Brand/URL markers
    if any(ch in upper for ch in ("@", ".COM", ".NET", ".ORG", ".IO", "/", "&")):
        return True
    # Numeric content (e.g., "1-800-FLOWERS") -- treat as entity
    if any(c.isdigit() for c in upper):
        return True
    tokens = re.findall(r"[A-Z][A-Z.]*", upper)
    return any(tok in ENTITY_MARKERS for tok in tokens)


def parse_surname(name: str) -> str | None:
    """Return the candidate surname from a personal-name-shaped string.

    Strategy: tokenise on whitespace and punctuation, drop suffixes,
    take the last remaining token if length 2-4 tokens.  Returns None if
    the name doesn't look like a personal name.
    """
    raw = name.upper().strip()
    if not raw:
        return None
    if is_entity(raw):
        return None
    # Tokenise on whitespace and hyphens/commas
    tokens = re.split(r"[\s,]+", raw)
    tokens = [t.strip(".") for t in tokens if t.strip(".")]
    # Drop suffix tokens at the end
    while tokens and tokens[-1] in SUFFIX_TOKENS:
        tokens.pop()
    # Personal-name pattern: 2-4 tokens, each alpha
    if not (2 <= len(tokens) <= 4):
        return None
    if not all(re.match(r"^[A-Z'\-]+$", t) for t in tokens):
        return None
    # Handle "LAST, FIRST" pattern (very common in some applicant records).
    # We detect this by looking at the original string: if there is a
    # comma between first and second token, then token 0 is the surname.
    if "," in raw:
        # Find position of first comma
        head = raw.split(",", 1)[0].strip()
        head_tokens = re.split(r"\s+", head)
        head_tokens = [t.strip(".") for t in head_tokens if t.strip(".")]
        if 1 <= len(head_tokens) <= 2 and head_tokens and re.match(r"^[A-Z'\-]+$", head_tokens[0]):
            return head_tokens[0]
    return tokens[-1]


def debut_owner_names() -> pl.DataFrame:
    """For every owner, find their first-ever filing year and a NICE class
    of that filing.  We arbitrate ties by taking the alphabetically-first
    NICE class so the row is well-defined."""
    parts = []
    for cls in class_list():
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["owner_name", "filing_date", "nice_classes"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(pl.lit(cls).alias("class_id"))
        parts.append(tm.select(["owner_name", "filing_date", "class_id"]))
        del tm
        gc.collect()
    allf = pl.concat(parts)
    # First-ever per owner
    debut = (
        allf.sort(["owner_name", "filing_date", "class_id"])
        .group_by("owner_name", maintain_order=False)
        .agg(pl.col("filing_date").first().alias("debut_date"),
             pl.col("class_id").first().alias("class_id"))
        .with_columns(pl.col("debut_date").str.slice(0, 4).cast(pl.Int32).alias("year"))
    )
    return debut


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- 1. Census surname table
    print("[1/4] census surname lookup …", flush=True)
    surn = pl.read_csv(RAW / "census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    print(f"      {surn.height:,} surnames")

    # --- 2. Build debut panel
    print("[2/4] building debut panel …", flush=True)
    debut = debut_owner_names()
    print(f"      {debut.height:,} unique owners with debut date")

    # --- 3. Personal-name detection + surname extraction
    print("[3/4] parsing surnames from personal-name-shaped filers …", flush=True)
    surnames = []
    for nm in debut["owner_name"].to_list():
        surnames.append(parse_surname(nm) if nm else None)
    debut = debut.with_columns(pl.Series("surname", surnames))
    n_indiv = int(debut["surname"].is_not_null().sum())
    print(f"      {n_indiv:,} of {debut.height:,} owners parse as individuals "
          f"({100*n_indiv/debut.height:.1f}%)")

    # Join to Census
    debut_indiv = debut.filter(pl.col("surname").is_not_null())
    joined = debut_indiv.join(
        surn.select(["name_u", "count", "pctwhite", "pctblack", "pctapi",
                     "pctaian", "pct2prace", "pcthispanic"]),
        left_on="surname", right_on="name_u", how="left",
    )
    n_matched = int(joined.filter(pl.col("count").is_not_null()).height)
    print(f"      {n_matched:,} of {n_indiv:,} individual debuts match Census surname "
          f"({100*n_matched/n_indiv:.1f}%)")

    # Treat missing pct as 0 (count contribution still happens)
    pct_cols = ["pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic"]
    joined = joined.with_columns([pl.col(c).fill_null(0.0) for c in pct_cols])

    # Each debut contributes fractional weight to each bucket.
    # Divide by 100 so weights sum to ~1.
    joined = joined.with_columns([(pl.col(c) / 100.0).alias(f"w_{c[3:]}") for c in pct_cols])
    weight_cols = [f"w_{c[3:]}" for c in pct_cols]

    # --- 4. Aggregation by year
    print("[4/4] aggregating by year …", flush=True)
    has_match = joined.filter(pl.col("count").is_not_null())
    by_year = has_match.group_by("year").agg([
        pl.len().alias("n_matched"),
        *[pl.col(w).sum().alias(w) for w in weight_cols],
    ]).sort("year")
    # Share columns
    by_year = by_year.with_columns([
        (pl.col(w) / pl.col("n_matched")).alias(f"share_{w[2:]}") for w in weight_cols
    ])
    by_year.write_csv(OUT / "surname_ethnicity_panel.csv")
    print()
    print(by_year.filter(pl.col("year") >= 2000)
                 .select(["year", "n_matched"] + [f"share_{w[2:]}" for w in weight_cols])
                 .to_pandas().to_string(index=False))

    # By NICE class -- overall shares across all years
    by_class = has_match.group_by("class_id").agg([
        pl.len().alias("n_matched"),
        *[pl.col(w).sum().alias(w) for w in weight_cols],
    ]).sort("class_id")
    by_class = by_class.with_columns([
        (pl.col(w) / pl.col("n_matched")).alias(f"share_{w[2:]}") for w in weight_cols
    ])
    by_class.write_csv(OUT / "surname_ethnicity_by_class.csv")

    # --- Plot time-series
    pdf = by_year.filter((pl.col("year") >= 1990) & (pl.col("year") <= 2024)).to_pandas()
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = {
        "share_white": ("Non-Hispanic White", "#2b6cb0"),
        "share_black": ("Non-Hispanic Black", "#cc4444"),
        "share_api": ("Asian / Pacific Islander", "#229922"),
        "share_hispanic": ("Hispanic (any race)", "#cc7a00"),
        "share_2prace": ("Two or More Races", "#999999"),
        "share_aian": ("American Indian / Alaska Native", "#7a3eb2"),
    }
    for col, (label, color) in labels.items():
        ax.plot(pdf["year"], 100 * pdf[col], "-o", color=color, ms=4, lw=1.5, label=label)
    ax.set_xlabel("Debut filing year")
    ax.set_ylabel("Share of individual-name debut filers (%)")
    ax.set_title("Estimated race/ethnicity composition of USPTO individual-name\n"
                 "trademark debut filers via Census 2010 surname probabilities, 1990-2024")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "surname_ethnicity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] {OUT/'surname_ethnicity.png'}")

    # Summary
    lines = []
    lines.append("Surname-based race/ethnicity composition of USPTO debut filers")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Debut owners total:         {debut.height:>10,}")
    lines.append(f"Parse as individuals:       {n_indiv:>10,}  ({100*n_indiv/debut.height:.1f}% of debuts)")
    lines.append(f"Matched in Census surname:  {n_matched:>10,}  ({100*n_matched/n_indiv:.1f}% of individuals)")
    lines.append("")
    lines.append("Headline shares over time (% of individual-name matched debut filers):")
    lines.append("")
    for yr_band in [(1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024)]:
        lo, hi = yr_band
        sub = pdf[(pdf.year >= lo) & (pdf.year <= hi)]
        if not len(sub):
            continue
        total_n = sub["n_matched"].sum()
        lines.append(f"{lo}-{hi}  (n={total_n:>9,})")
        for col, (label, _) in labels.items():
            wsum = (sub[col] * sub["n_matched"]).sum()
            avg = wsum / sub["n_matched"].sum() if sub["n_matched"].sum() else 0
            lines.append(f"   {label:<35s} {100*avg:>6.2f}%")
        lines.append("")
    summary = "\n".join(lines)
    (OUT / "surname_ethnicity.txt").write_text(summary)
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
