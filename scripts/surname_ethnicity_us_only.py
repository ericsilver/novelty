"""US-only version of the surname-ethnicity analysis.

Joins to data/processed/owner_address.parquet (from the USPTO XML
re-parse) to identify each debut owner's country, then restricts to
owner_country == 'US' for the surname classification.

This deconfounds the post-2018 Chinese-applicant filing surge from
the genuine domestic Asian-American entrepreneurship trend that the
surname classifier previously conflated.

Outputs:
  paper/results/surname_ethnicity_us_panel.csv
  paper/results/surname_ethnicity_us.png
  paper/results/surname_ethnicity_us_counts.png
  paper/results/surname_ethnicity_us_compare.png    (US-only vs all-filings, share)
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
REF = REPO / "reference_data"
OUT = REPO / "paper" / "results"

ENTITY_MARKERS = {
    "LLC","L.L.C.","LL.C.","L.L.C","INC","INC.","INCORPORATED",
    "CORP","CORP.","CORPORATION","CO","CO.","COMPANY",
    "LTD","LTD.","LIMITED","LP","L.P.","LLP","L.L.P.","PLC","P.L.C.",
    "GMBH","AG","NV","N.V.","BV","B.V.","SA","S.A.","SAS",
    "OY","AB","KG","S.R.L.","SRL","S.A.S.",
    "TRUST","FOUNDATION","FUND","ASSOCIATION","ASSN","ASSN.",
    "SOCIETY","INSTITUTE","INSTITUTION",
    "GROUP","HOLDINGS","ENTERPRISES","INDUSTRIES","BRANDS",
    "INTERNATIONAL","WORLDWIDE","GLOBAL",
    "UNIVERSITY","COLLEGE","SCHOOL","ACADEMY",
    "HOSPITAL","CLINIC","MEDICAL","CHURCH","MINISTRIES","PARISH","DIOCESE",
    "BANK","INSURANCE","CAPITAL","VENTURES","PARTNERS",
    "TECHNOLOGIES","TECHNOLOGY","SOLUTIONS","SYSTEMS",
    "SERVICES","PRODUCTS","PRODUCTIONS","STUDIOS","MEDIA","ENTERTAINMENT",
    "RESTAURANTS","FOODS","AGENCY","BUREAU","DEPARTMENT","GOVERNMENT",
    "USA","U.S.A.","U.S.","AMERICA","OF","AND","AND/OR",
}
SUFFIX_TOKENS = {"JR","JR.","SR","SR.","II","III","IV","V",
                 "ESQ","ESQ.","MD","M.D.","DO","D.O.","PHD","PH.D.",
                 "DDS","D.D.S.","JD","J.D.","CPA","C.P.A."}


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def is_entity(name: str) -> bool:
    upper = name.upper()
    if any(ch in upper for ch in ("@",".COM",".NET",".ORG",".IO","/","&")):
        return True
    if any(c.isdigit() for c in upper): return True
    tokens = re.findall(r"[A-Z][A-Z.]*", upper)
    return any(tok in ENTITY_MARKERS for tok in tokens)


def parse_surname(name: str) -> str | None:
    raw = name.upper().strip()
    if not raw or is_entity(raw): return None
    tokens = re.split(r"[\s,]+", raw)
    tokens = [t.strip(".") for t in tokens if t.strip(".")]
    while tokens and tokens[-1] in SUFFIX_TOKENS: tokens.pop()
    if not (2 <= len(tokens) <= 4): return None
    if not all(re.match(r"^[A-Z'\-]+$", t) for t in tokens): return None
    if "," in raw:
        head = raw.split(",", 1)[0].strip()
        ht = [t.strip(".") for t in re.split(r"\s+", head) if t.strip(".")]
        if 1 <= len(ht) <= 2 and ht and re.match(r"^[A-Z'\-]+$", ht[0]):
            return ht[0]
    return tokens[-1]


def debut_owner_panel() -> pl.DataFrame:
    """Each owner -> earliest filing across all classes + the
    serial_number of that earliest filing (so we can look up its
    owner_country)."""
    parts = []
    for cls in class_list():
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number","owner_name","filing_date"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(pl.lit(cls).alias("class_id"))
        parts.append(tm)
        del tm
        gc.collect()
    allf = pl.concat(parts)
    return (allf.sort(["owner_name","filing_date","class_id"])
                .group_by("owner_name", maintain_order=False)
                .agg(pl.col("filing_date").first().alias("debut_date"),
                     pl.col("class_id").first().alias("class_id"),
                     pl.col("serial_number").first().alias("debut_serial"))
                .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int32).alias("year")))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] loading address parquet …", flush=True)
    addr = pl.read_parquet(PROC / "owner_address.parquet",
                            columns=["serial_number","owner_country"])
    print(f"      {addr.height:,} address rows")

    print("[2/5] building debut owner panel …", flush=True)
    debut = debut_owner_panel()
    print(f"      {debut.height:,} unique owners")

    print("[3/5] joining debut to address …", flush=True)
    debut = debut.join(addr, left_on="debut_serial", right_on="serial_number", how="left")
    print(f"      country distribution of debut filings:")
    cd = debut.group_by("owner_country").len().sort("len", descending=True).head(8)
    print(cd.to_pandas().to_string(index=False))

    # US-domiciled subset
    debut_us = debut.filter(pl.col("owner_country") == "US")
    print(f"\n      US-domiciled debut owners: {debut_us.height:,} of {debut.height:,} ({100*debut_us.height/debut.height:.1f}%)")

    print("[4/5] parsing surnames + Census ethnicity (US-only) …", flush=True)
    surnames = [parse_surname(n) if n else None for n in debut_us["owner_name"].to_list()]
    debut_us = debut_us.with_columns(pl.Series("surname", surnames))
    n_indiv = debut_us.filter(pl.col("surname").is_not_null()).height
    print(f"      US-domiciled individuals: {n_indiv:,}")

    surn = pl.read_csv(REF/"census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    joined = debut_us.filter(pl.col("surname").is_not_null()).join(
        surn.select(["name_u"] + pct_cols), left_on="surname", right_on="name_u", how="inner",
    ).with_columns([(pl.col(c).fill_null(0.0)/100.0).alias(f"w_{c[3:]}") for c in pct_cols])
    print(f"      matched Census surnames: {joined.height:,} ({100*joined.height/n_indiv:.1f}% of US individuals)")

    print("[5/5] aggregating + plotting …", flush=True)
    weight_cols = [f"w_{c[3:]}" for c in pct_cols]
    by_year = joined.group_by("year").agg([
        pl.len().alias("n_matched"),
        *[pl.col(w).sum().alias(w) for w in weight_cols],
    ]).sort("year").with_columns([
        (pl.col(w)/pl.col("n_matched")).alias(f"share_{w[2:]}") for w in weight_cols
    ])
    by_year.write_csv(OUT / "surname_ethnicity_us_panel.csv")

    # ------- Plot 1: shares over time (US-only)
    pdf = by_year.filter((pl.col("year") >= 1990) & (pl.col("year") <= 2024)).to_pandas()
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = {
        "share_white":    ("Non-Hispanic White",    "#2b6cb0"),
        "share_api":      ("Asian / Pacific Islander", "#229922"),
        "share_hispanic": ("Hispanic (any race)",  "#cc7a00"),
        "share_black":    ("Non-Hispanic Black",   "#cc4444"),
        "share_2prace":   ("Two or More Races",    "#999999"),
        "share_aian":     ("American Indian / Alaska Native", "#7a3eb2"),
    }
    for col, (lab, color) in labels.items():
        ax.plot(pdf["year"], 100*pdf[col], "-o", color=color, ms=4, lw=1.5, label=lab)
    ax.axvline(2019.6, color="#444", ls="--", lw=1)
    ax.axvline(2021.7, color="#888", ls=":", lw=1)
    ax.text(2019.7, ax.get_ylim()[1]*0.9,
            "USPTO foreign-counsel rule\n(no longer the main story\nbecause this plot is US-only)",
            fontsize=8, color="#333", style="italic", va="top")
    ax.set_xlabel("Debut filing year")
    ax.set_ylabel("Share of US-domiciled individual debut filers (%)")
    ax.set_title("US-only: ethnic composition of USPTO debut filers, 1990-2024\n(stripping foreign-domiciled records that contaminated the previous plot)")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "surname_ethnicity_us.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------- Plot 2: counts on log scale (US-only)
    fig, ax = plt.subplots(figsize=(11, 6))
    eth_cols = ["w_white", "w_api", "w_hispanic", "w_black", "w_2prace", "w_aian"]
    eth_labels = {
        "w_white":    ("Non-Hispanic White",    "#2b6cb0"),
        "w_api":      ("Asian / Pacific Islander", "#229922"),
        "w_hispanic": ("Hispanic (any race)",  "#cc7a00"),
        "w_black":    ("Non-Hispanic Black",   "#cc4444"),
        "w_2prace":   ("Two or More Races",    "#999999"),
        "w_aian":     ("American Indian / Alaska Native", "#7a3eb2"),
    }
    for col, (lab, color) in eth_labels.items():
        ax.plot(pdf["year"], pdf[col], "-o", color=color, ms=4, lw=1.5, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Debut filing year")
    ax.set_ylabel("US-domiciled debut filers (imputed count, log scale)")
    ax.set_title("US-only debut filer counts by surname-imputed ethnicity, 1990-2024")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "surname_ethnicity_us_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------- Plot 3: side-by-side ALL vs US-only (share comparison)
    tm_all = pl.read_csv(OUT / "surname_ethnicity_panel.csv").filter(
        (pl.col("year") >= 1990) & (pl.col("year") <= 2024)
    ).to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for col, (lab, color) in labels.items():
        axes[0].plot(tm_all["year"], 100*tm_all[col], "-o", color=color, ms=3, lw=1.4, label=lab)
        axes[1].plot(pdf["year"],    100*pdf[col],    "-o", color=color, ms=3, lw=1.4, label=lab)
    axes[0].set_title("All trademark debuts (includes foreign-domiciled filers)")
    axes[1].set_title("US-domiciled debuts only (foreign-applicant confound stripped)")
    for ax in axes:
        ax.set_xlabel("Debut filing year")
        ax.grid(alpha=0.3)
        ax.set_xlim(1990, 2024)
    axes[0].set_ylabel("Surname-imputed ethnic share (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("Effect of stripping foreign-domiciled trademark filings on apparent ethnic composition",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "surname_ethnicity_us_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- Summary
    print()
    print("US-only composition by decade:")
    for yr_band in [(1990,1999),(2000,2009),(2010,2019),(2020,2024)]:
        lo, hi = yr_band
        sub = pdf[(pdf["year"] >= lo) & (pdf["year"] <= hi)]
        n = sub["n_matched"].sum()
        print(f"  {lo}-{hi} (n_matched={n:>9,})  "
              + "  ".join(f"{l[:5]}={100*(sub[c]*sub['n_matched']).sum()/n:5.1f}%"
                          for c, (l, _) in labels.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
