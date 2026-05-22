"""Within-ethnicity NICE-class composition of USPTO debut filers,
1990-2024. Tests the Kerr (Gift of Global Talent, 2018) / Saxenian
(New Argonauts, 2006) hypothesis that immigrant-origin entrepreneurship
shifted from goods/low-skill services to high-skill/IP-intensive
sectors over recent decades.

For each ethnic group g (White, Black, API, Hispanic), compute:
  share_{g,t,c} = sum_{i in debuts(t)} pct_g(surname(i)) * I[class(i) = c]
                  / sum_{i in debuts(t)} pct_g(surname(i))

so the within-group composition sums to 1 across classes c for each
(g, t).  This is robust to the foreign-applicant level confound that
contaminates the cross-group level comparison: the SHIFT in
composition within a group reflects how that group's filers are
distributing across sectors, separately from the overall scale.

NICE classes are grouped into five skill-tier buckets:
  IP/HIGH:    009 (electronics/software), 041 (education/entertainment),
              042 (scientific & tech svc)
  PRO_SVC:    035 (advertising/retail biz svc), 036 (financial),
              045 (legal/security)
  MANUF:      001-008, 010-028 (manufactured goods other than food)
  FOOD_HYG:   029, 030, 032, 033 (food/beverage), 043 (restaurant),
              044 (medical/hygiene)
  OTHER:      everything else (construction 037, transport 039, telecom 038, ...)

Outputs:
  paper/results/within_ethnicity_classmix.csv
  paper/results/within_ethnicity_skill_shift.png
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
REF = REPO / "reference_data"
OUT = REPO / "paper" / "results"

# Same entity heuristics as surname_ethnicity.py
ENTITY_MARKERS = {
    "LLC", "L.L.C.", "LL.C.", "L.L.C",
    "INC", "INC.", "INCORPORATED",
    "CORP", "CORP.", "CORPORATION",
    "CO", "CO.", "COMPANY",
    "LTD", "LTD.", "LIMITED",
    "LP", "L.P.", "LLP", "L.L.P.", "PLC", "P.L.C.",
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
    "OF", "AND", "AND/OR",
}
SUFFIX_TOKENS = {"JR", "JR.", "SR", "SR.", "II", "III", "IV", "V",
                 "ESQ", "ESQ.", "MD", "M.D.", "DO", "D.O.", "PHD", "PH.D.",
                 "DDS", "D.D.S.", "JD", "J.D.", "CPA", "C.P.A."}


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class", "") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def is_entity(name: str) -> bool:
    upper = name.upper()
    if any(ch in upper for ch in ("@", ".COM", ".NET", ".ORG", ".IO", "/", "&")):
        return True
    if any(c.isdigit() for c in upper):
        return True
    tokens = re.findall(r"[A-Z][A-Z.]*", upper)
    return any(tok in ENTITY_MARKERS for tok in tokens)


def parse_surname(name: str) -> str | None:
    raw = name.upper().strip()
    if not raw or is_entity(raw):
        return None
    tokens = re.split(r"[\s,]+", raw)
    tokens = [t.strip(".") for t in tokens if t.strip(".")]
    while tokens and tokens[-1] in SUFFIX_TOKENS:
        tokens.pop()
    if not (2 <= len(tokens) <= 4):
        return None
    if not all(re.match(r"^[A-Z'\-]+$", t) for t in tokens):
        return None
    if "," in raw:
        head = raw.split(",", 1)[0].strip()
        head_tokens = re.split(r"\s+", head)
        head_tokens = [t.strip(".") for t in head_tokens if t.strip(".")]
        if 1 <= len(head_tokens) <= 2 and head_tokens and re.match(r"^[A-Z'\-]+$", head_tokens[0]):
            return head_tokens[0]
    return tokens[-1]


# Skill-tier mapping
SKILL_TIER = {}
for c in ["009", "041", "042"]:
    SKILL_TIER[c] = "IP_HIGH"
for c in ["035", "036", "045"]:
    SKILL_TIER[c] = "PRO_SVC"
for c in (["001","002","003","004","005","006","007","008",
           "010","011","012","013","014","015","016","017","018",
           "019","020","021","022","023","024","025","026","027","028"]):
    SKILL_TIER[c] = "MANUF"
for c in ["029", "030", "032", "033", "043", "044"]:
    SKILL_TIER[c] = "FOOD_HYG"
# everything else → OTHER (031 agriculture, 034 tobacco, 037 construction,
# 038 telecom, 039 transport, 040 materials treatment)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- 1. Build debut panel with first NICE class
    print("[1/4] building debut panel + first NICE class …", flush=True)
    parts = []
    for cls in class_list():
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["owner_name", "filing_date"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(pl.lit(cls).alias("class_id"))
        parts.append(tm)
        del tm
        gc.collect()
    allf = pl.concat(parts)
    debut = (
        allf.sort(["owner_name", "filing_date", "class_id"])
            .group_by("owner_name", maintain_order=False)
            .agg(pl.col("filing_date").first().alias("debut_date"),
                 pl.col("class_id").first().alias("class_id"))
            .with_columns(pl.col("debut_date").str.slice(0, 4).cast(pl.Int32).alias("year"))
    )
    print(f"      {debut.height:,} unique owners")

    # --- 2. Parse surnames
    print("[2/4] parsing surnames …", flush=True)
    surnames = [parse_surname(n) if n else None for n in debut["owner_name"].to_list()]
    debut = debut.with_columns(pl.Series("surname", surnames))
    debut_indiv = debut.filter(pl.col("surname").is_not_null())
    print(f"      {debut_indiv.height:,} parse as individuals")

    # --- 3. Join Census surname → race probabilities
    print("[3/4] joining Census surname probabilities …", flush=True)
    surn = pl.read_csv(REF / "census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    joined = debut_indiv.join(
        surn.select(["name_u", "pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic"]),
        left_on="surname", right_on="name_u", how="inner",
    )
    pct_cols = ["pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic"]
    joined = joined.with_columns([(pl.col(c).fill_null(0.0) / 100.0).alias(f"w_{c[3:]}") for c in pct_cols])
    print(f"      {joined.height:,} matched")

    # --- 4. Assign skill tier
    joined = joined.with_columns(
        pl.col("class_id").map_elements(lambda c: SKILL_TIER.get(c, "OTHER"),
                                         return_dtype=pl.Utf8).alias("tier")
    )

    # Aggregate weighted counts by (year, ethnic_group, tier)
    print("[4/4] computing within-ethnicity shares by tier …", flush=True)
    weight_cols = [f"w_{c[3:]}" for c in pct_cols]
    by_year_tier = joined.group_by(["year", "tier"]).agg(
        [pl.col(w).sum().alias(w) for w in weight_cols]
    ).sort(["year", "tier"])

    # Compute within-ethnic-group share by tier per year
    long_rows = []
    eth_map = {
        "w_white": "White", "w_black": "Black", "w_api": "API",
        "w_hispanic": "Hispanic", "w_2prace": "Two+", "w_aian": "AIAN",
    }
    pdf_by_yt = by_year_tier.to_pandas()
    for year, sub in pdf_by_yt.groupby("year"):
        for wc, eth in eth_map.items():
            total = sub[wc].sum()
            if total == 0:
                continue
            for _, row in sub.iterrows():
                long_rows.append({
                    "year": year,
                    "ethnic": eth,
                    "tier": row["tier"],
                    "weight_in_tier": row[wc],
                    "share_within_ethnic": row[wc] / total,
                })
    long = pl.DataFrame(long_rows)
    long.write_csv(OUT / "within_ethnicity_classmix.csv")

    # --- Plot: 4-panel (one per major ethnic group), stacked area of tier shares over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    tiers_order = ["IP_HIGH", "PRO_SVC", "MANUF", "FOOD_HYG", "OTHER"]
    tier_colors = {
        "IP_HIGH":  "#2b6cb0",
        "PRO_SVC":  "#229922",
        "MANUF":    "#cc7a00",
        "FOOD_HYG": "#cc4444",
        "OTHER":    "#999999",
    }
    tier_labels = {
        "IP_HIGH":  "IP/high-skill (NICE 009, 041, 042)",
        "PRO_SVC":  "Professional svc (035, 036, 045)",
        "MANUF":    "Manufactured goods (non-food, 001-028)",
        "FOOD_HYG": "Food/hygiene/restaurants (029, 030, 032, 033, 043, 044)",
        "OTHER":    "Other (031, 034, 037-040)",
    }
    pdf_long = long.to_pandas()
    eth_panels = [("White", axes[0,0]), ("API", axes[0,1]),
                  ("Hispanic", axes[1,0]), ("Black", axes[1,1])]
    for eth, ax in eth_panels:
        sub = pdf_long[(pdf_long["ethnic"] == eth) & (pdf_long["year"].between(1990, 2024))]
        pivot = sub.pivot(index="year", columns="tier", values="share_within_ethnic").fillna(0)
        pivot = pivot.reindex(columns=tiers_order)
        cum = np.zeros(len(pivot))
        for t in tiers_order:
            vals = pivot[t].to_numpy()
            ax.fill_between(pivot.index, cum, cum + vals, color=tier_colors[t],
                            alpha=0.85, label=tier_labels[t] if eth == "White" else None)
            cum = cum + vals
        ax.set_title(f"{eth}-classified debut filers")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_ylabel("Within-group share")
    axes[1,0].set_xlabel("Debut year")
    axes[1,1].set_xlabel("Debut year")
    axes[0,0].legend(loc="upper left", bbox_to_anchor=(0.0, -1.18), fontsize=9, ncol=2,
                     frameon=False)
    fig.suptitle("Within-ethnicity NICE-class composition of USPTO debut filers, 1990-2024\n"
                 "Tests Kerr (2018) / Saxenian (2006) sectoral-shift hypothesis",
                 fontsize=12, y=1.00)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(OUT / "within_ethnicity_skill_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print a punchy summary: API shift in IP_HIGH share, 1995-2024
    print()
    print("Within-ethnicity share by skill tier -- selected years/groups:")
    print()
    for eth in ["White", "API", "Hispanic", "Black"]:
        print(f"=== {eth} ===")
        for yr_band in [(1990, 1999), (2000, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]:
            lo, hi = yr_band
            sub = pdf_long[(pdf_long["ethnic"] == eth) &
                           (pdf_long["year"].between(lo, hi))]
            sums = sub.groupby("tier")["weight_in_tier"].sum()
            total = sums.sum()
            if total == 0:
                continue
            print(f"  {lo}-{hi}:  " + "  ".join(
                f"{t}={100*sums.get(t, 0)/total:5.1f}%" for t in tiers_order
            ))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
