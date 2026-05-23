"""Decompose the Black trademark-share rise and patent-share fall
by class (NICE / CPC).

For each NICE class on the trademark side: Black share of US-domiciled
individual debut filers in 1990-99 vs 2020-24; identify classes
driving the +4pp rise.

For each CPC class on the patent side: Black share of US-resident
inventor debuts in 1990-99 vs 2020-24; identify classes where Black
share fell most.

Inputs:
  data/processed/tm_class*.parquet + owner_address.parquet
  data/raw/patentsview/g_inventor_disambiguated.tsv
                       g_patent.tsv
                       g_cpc_current.tsv
                       g_location_disambiguated.tsv
  reference_data/census_surnames/Names_2010Census.csv

Outputs:
  paper/results/black_tm_decomposition.csv
  paper/results/black_patent_decomposition.csv
  paper/results/black_decomposition.png
"""

from __future__ import annotations

import gc, re, unicodedata
from pathlib import Path

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
PV = REPO / "data" / "raw" / "patentsview"
REF = REPO / "reference_data"
OUT = REPO / "paper" / "results"

ENTITY_MARKERS = {
    "LLC","L.L.C.","LL.C.","L.L.C","INC","INC.","INCORPORATED",
    "CORP","CORP.","CORPORATION","CO","CO.","COMPANY","LTD","LTD.","LIMITED",
    "LP","L.P.","LLP","L.L.P.","PLC","P.L.C.","GMBH","AG","NV","N.V.","BV","B.V.","SA","S.A.","SAS",
    "OY","AB","KG","S.R.L.","SRL","S.A.S.","TRUST","FOUNDATION","FUND",
    "ASSOCIATION","ASSN","ASSN.","SOCIETY","INSTITUTE","INSTITUTION",
    "GROUP","HOLDINGS","ENTERPRISES","INDUSTRIES","BRANDS","INTERNATIONAL","WORLDWIDE","GLOBAL",
    "UNIVERSITY","COLLEGE","SCHOOL","ACADEMY","HOSPITAL","CLINIC","MEDICAL",
    "CHURCH","MINISTRIES","PARISH","DIOCESE","BANK","INSURANCE","CAPITAL","VENTURES","PARTNERS",
    "TECHNOLOGIES","TECHNOLOGY","SOLUTIONS","SYSTEMS","SERVICES","PRODUCTS","PRODUCTIONS",
    "STUDIOS","MEDIA","ENTERTAINMENT","RESTAURANTS","FOODS",
    "AGENCY","BUREAU","DEPARTMENT","GOVERNMENT","USA","U.S.A.","U.S.","AMERICA","OF","AND","AND/OR",
}
SUFFIX_TOKENS = {"JR","JR.","SR","SR.","II","III","IV","V","ESQ","ESQ.",
                 "MD","M.D.","DO","D.O.","PHD","PH.D.","DDS","D.D.S.","JD","J.D.","CPA","C.P.A."}

INDUSTRY_NAMES = {
    "001":"Chemicals","002":"Paints","003":"Cosmetics & Cleaning",
    "004":"Oils & Fuels","005":"Pharmaceuticals","006":"Common Metals",
    "007":"Machines","008":"Hand Tools","009":"Electronics & Software",
    "010":"Surgical/Medical/Dental","011":"Lighting/Heating","012":"Vehicles",
    "013":"Firearms","014":"Jewelry","015":"Musical Instruments",
    "016":"Paper/Print","017":"Rubber/Plastic","018":"Leather",
    "019":"Building Materials","020":"Furniture","021":"Household Utensils",
    "022":"Ropes/Nets/Textiles","023":"Yarns","024":"Fabrics",
    "025":"Clothing & Footwear","026":"Lace/Embroidery","027":"Carpets",
    "028":"Toys & Sports","029":"Meat/Fish/Dairy","030":"Coffee/Tea/Sugar",
    "031":"Agricultural Products","032":"Beverages (Non-Alcoholic)",
    "033":"Alcoholic Beverages","034":"Tobacco",
    "035":"Advertising & Retail","036":"Insurance & Financial",
    "037":"Construction & Repair","038":"Telecommunications",
    "039":"Transport & Travel","040":"Materials Treatment",
    "041":"Education & Entertainment","042":"Scientific & Tech Services",
    "043":"Hotel/Restaurant/Food","044":"Medical & Veterinary",
    "045":"Legal & Personal",
}

CPC_NAMES = {
    "A": "Human Necessities",
    "B": "Performing Operations; Transporting",
    "C": "Chemistry; Metallurgy",
    "D": "Textiles; Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Eng; Lighting; Weapons",
    "G": "Physics (incl. computing)",
    "H": "Electricity",
    "Y": "General tagging of new tech",
}


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def is_entity(name: str) -> bool:
    upper = name.upper()
    if any(ch in upper for ch in ("@",".COM",".NET",".ORG",".IO","/","&")): return True
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


def normalize_surname_pv(s: str | None) -> str | None:
    if s is None or s == "": return None
    nrm = unicodedata.normalize("NFD", s)
    nrm = "".join(c for c in nrm if unicodedata.category(c) != "Mn")
    nrm = nrm.upper().strip()
    nrm = re.sub(r"[^A-Z'\- ]", "", nrm)
    parts = nrm.split()
    if not parts: return None
    return parts[-1] if len(parts[-1]) >= 2 else None


def tm_decomposition():
    """Per NICE class: Black share of US-domiciled individual-name debuts,
    1990-99 vs 2020-24."""
    print("[tm] address parquet …", flush=True)
    addr = pl.read_parquet(PROC / "owner_address.parquet",
                            columns=["serial_number","owner_country"])
    print("[tm] debut panel with class …", flush=True)
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
    allf = pl.concat(parts)
    debut = (allf.sort(["owner_name","filing_date","class_id"])
                 .group_by("owner_name", maintain_order=False)
                 .agg(pl.col("filing_date").first().alias("debut_date"),
                      pl.col("class_id").first().alias("class_id"),
                      pl.col("serial_number").first().alias("debut_serial"))
                 .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int32).alias("year"))
                 .join(addr, left_on="debut_serial", right_on="serial_number", how="left")
                 .filter(pl.col("owner_country") == "US"))

    surnames = [parse_surname(n) if n else None for n in debut["owner_name"].to_list()]
    debut = debut.with_columns(pl.Series("surname", surnames))

    surn = pl.read_csv(REF/"census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    joined = debut.filter(pl.col("surname").is_not_null()).join(
        surn.select(["name_u"] + pct_cols), left_on="surname", right_on="name_u", how="inner",
    ).with_columns([(pl.col(c).fill_null(0.0)/100.0).alias(f"w_{c[3:]}") for c in pct_cols])

    print("[tm] aggregating by (class, period) …", flush=True)
    weight_cols = [f"w_{c[3:]}" for c in pct_cols]
    pdf = joined.to_pandas()
    rows = []
    for cls, sub in pdf.groupby("class_id"):
        early = sub[(sub["year"] >= 1990) & (sub["year"] <= 1999)]
        late  = sub[(sub["year"] >= 2020) & (sub["year"] <= 2024)]
        if len(early) < 200 or len(late) < 200: continue
        rows.append({
            "class_id": cls,
            "class_name": INDUSTRY_NAMES.get(cls, cls),
            "n_early": len(early), "n_late": len(late),
            "black_share_early": early["w_black"].sum() / len(early),
            "black_share_late":  late["w_black"].sum() / len(late),
            "n_black_early": early["w_black"].sum(),
            "n_black_late":  late["w_black"].sum(),
        })
    df = pd.DataFrame(rows)
    df["delta_share"] = df["black_share_late"] - df["black_share_early"]
    df["delta_n"] = df["n_black_late"] - df["n_black_early"]
    df = df.sort_values("delta_n", ascending=False)
    df.to_csv(OUT / "black_tm_decomposition.csv", index=False)
    return df


def patent_decomposition():
    """Per CPC class (3-char): Black share of US-resident inventor debuts,
    1990-99 vs 2020-24."""
    print("[pv] loading inventors + locations + patents + CPC …", flush=True)
    loc = pl.read_csv(PV / "g_location_disambiguated.tsv", separator="\t",
                       null_values=[""]).select("location_id", "disambig_country")
    inv = pl.read_csv(
        PV / "g_inventor_disambiguated.tsv", separator="\t", null_values=[""],
    ).select("patent_id", "inventor_id", "disambig_inventor_name_last", "location_id")
    inv = inv.join(loc, on="location_id", how="left").rename({"disambig_country":"country"})
    pat = pl.read_csv(PV / "g_patent.tsv", separator="\t", null_values=[""],
                       schema_overrides={"patent_id": pl.Utf8}).select("patent_id","patent_date")
    inv = inv.join(pat, on="patent_id", how="left").with_columns(
        pl.col("patent_date").str.slice(0,4).cast(pl.Int32).alias("year")
    ).filter(pl.col("country") == "US")
    cpc = pl.read_csv(PV / "g_cpc_current.tsv", separator="\t", null_values=[""],
                       schema_overrides={"patent_id": pl.Utf8, "cpc_sequence": pl.Int64}
                       ).filter(pl.col("cpc_sequence") == 0).select("patent_id","cpc_class")
    inv = inv.join(cpc, on="patent_id", how="inner")

    print("[pv] surname + Census ethnicity …", flush=True)
    inv_debut = (inv.filter(pl.col("year").is_not_null() & pl.col("disambig_inventor_name_last").is_not_null())
                    .group_by("inventor_id").agg(
                        pl.col("year").min().alias("debut_year"),
                        pl.col("disambig_inventor_name_last").first().alias("last_name"),
                        # take the inventor's most-frequent CPC class as their primary
                        pl.col("cpc_class").mode().first().alias("primary_cpc")))
    surnames = [normalize_surname_pv(s) for s in inv_debut["last_name"].to_list()]
    inv_debut = inv_debut.with_columns(pl.Series("surname", surnames)).filter(pl.col("surname").is_not_null())

    surn = pl.read_csv(REF/"census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    joined = inv_debut.join(
        surn.select(["name_u"] + pct_cols), left_on="surname", right_on="name_u", how="inner",
    ).with_columns([(pl.col(c).fill_null(0.0)/100.0).alias(f"w_{c[3:]}") for c in pct_cols])

    print("[pv] aggregating by (CPC class, period) …", flush=True)
    pdf = joined.to_pandas()
    rows = []
    for cls, sub in pdf.groupby("primary_cpc"):
        early = sub[(sub["debut_year"] >= 1990) & (sub["debut_year"] <= 1999)]
        late  = sub[(sub["debut_year"] >= 2020) & (sub["debut_year"] <= 2024)]
        if len(early) < 200 or len(late) < 200: continue
        rows.append({
            "cpc_class": cls,
            "cpc_section": cls[0] if isinstance(cls, str) and cls else None,
            "cpc_section_name": CPC_NAMES.get(cls[0], "") if cls else "",
            "n_early": len(early), "n_late": len(late),
            "black_share_early": early["w_black"].sum() / len(early),
            "black_share_late":  late["w_black"].sum() / len(late),
            "n_black_early": early["w_black"].sum(),
            "n_black_late":  late["w_black"].sum(),
        })
    df = pd.DataFrame(rows)
    df["delta_share"] = df["black_share_late"] - df["black_share_early"]
    df["delta_n"] = df["n_black_late"] - df["n_black_early"]
    df.to_csv(OUT / "black_patent_decomposition.csv", index=False)
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TRADEMARK SIDE: Black share by NICE class, 1990-99 vs 2020-24")
    print("="*70)
    tm = tm_decomposition()
    print("\nTop 12 NICE classes by Black-debut-count growth (US-only):")
    top_n = tm.sort_values("delta_n", ascending=False).head(12)
    print(top_n[["class_id","class_name","n_early","n_late",
                 "black_share_early","black_share_late","delta_share","delta_n"]].to_string(
                  index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTop 8 NICE classes by Black-share rise:")
    top_share = tm.sort_values("delta_share", ascending=False).head(8)
    print(top_share[["class_id","class_name","black_share_early","black_share_late","delta_share"]].to_string(
                  index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("="*70)
    print("PATENT SIDE: Black share by CPC class, 1990-99 vs 2020-24 (US-resident)")
    print("="*70)
    pv = patent_decomposition()
    print("\nTop 12 CPC classes by Black-inventor-count loss (US-resident, biggest fall):")
    bot_n = pv.sort_values("delta_n", ascending=True).head(12)
    print(bot_n[["cpc_class","cpc_section_name","n_early","n_late",
                  "black_share_early","black_share_late","delta_share","delta_n"]].to_string(
                  index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBottom 8 CPC classes by Black-share fall:")
    bot_share = pv.sort_values("delta_share", ascending=True).head(8)
    print(bot_share[["cpc_class","cpc_section_name","black_share_early","black_share_late","delta_share"]].to_string(
                  index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    # Left: TM, top-10 classes by delta_n
    top_tm = tm.sort_values("delta_n", ascending=False).head(12)
    y = np.arange(len(top_tm))[::-1]
    axes[0].barh(y, top_tm["delta_n"].values, color="#2b6cb0", alpha=0.85)
    axes[0].set_yticks(y); axes[0].set_yticklabels(
        [f"{r['class_id']} {r['class_name'][:24]}" for _, r in top_tm.iterrows()], fontsize=9)
    axes[0].set_xlabel("Δ Black debut filers (count, 2020-24 vs 1990-99)")
    axes[0].set_title("Trademark side: NICE classes driving Black-share rise")
    axes[0].grid(alpha=0.3, axis="x")

    # Right: Patent, bottom-10 classes by delta_share (biggest fall)
    bot_pv = pv.sort_values("delta_share", ascending=True).head(12)
    y = np.arange(len(bot_pv))[::-1]
    axes[1].barh(y, bot_pv["delta_share"].values * 100, color="#cc4444", alpha=0.85)
    axes[1].set_yticks(y); axes[1].set_yticklabels(
        [f"{r['cpc_class']} {r['cpc_section_name'][:24]}" for _, r in bot_pv.iterrows()], fontsize=9)
    axes[1].set_xlabel("Δ Black share of US-resident inventors (pp, 2020-24 vs 1990-99)")
    axes[1].set_title("Patent side: CPC classes driving Black-share fall")
    axes[1].grid(alpha=0.3, axis="x")
    fig.suptitle("Decomposing the Black-American divergence: where trademarks rise, where patents fall",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "black_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
