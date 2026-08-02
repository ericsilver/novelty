"""For each (NICE class, surname-imputed ethnicity) cell, compute the
mean ΔKL on debut filings. Then identify "ethnic enclave" classes
(classes where one group's share of individual-name debuts is high)
and compare the ΔKL profile of enclave-group debuts to non-enclave
debuts in the same class.

Hypothesis (Kerr & Mandorff 2023; Borjas 1992 ethnic capital): ethnic
clustering in an industry produces a stable but innovation-poor
filing pattern -- harvest-tail (negative ΔKL) dominant, low
positive-ΔKL share.

Outputs:
  paper/results/ethnic_cluster_dkl_panel.csv     -- (class, ethnic, n, mean_dkl)
  paper/results/ethnic_cluster_dkl_summary.csv   -- per-class summary
  paper/results/ethnic_cluster_dkl.png
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

# Reuse the entity heuristics
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

ETH_LABELS = {
    "w_white":    ("White",    "#2b6cb0"),
    "w_black":    ("Black",    "#cc4444"),
    "w_api":      ("API",      "#229922"),
    "w_hispanic": ("Hispanic", "#cc7a00"),
    "w_2prace":   ("2+races",  "#999999"),
    "w_aian":     ("AIAN",     "#7a3eb2"),
}

# Industry names (lifted from novelty.industries, kept inline so we don't
# import that module if it's unavailable in any branch state)
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


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def is_entity(name: str) -> bool:
    upper = name.upper()
    if any(ch in upper for ch in ("@",".COM",".NET",".ORG",".IO","/","&")):
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
    if not (2 <= len(tokens) <= 4): return None
    if not all(re.match(r"^[A-Z'\-]+$", t) for t in tokens): return None
    if "," in raw:
        head = raw.split(",", 1)[0].strip()
        ht = [t.strip(".") for t in re.split(r"\s+", head) if t.strip(".")]
        if 1 <= len(ht) <= 2 and ht and re.match(r"^[A-Z'\-]+$", ht[0]):
            return ht[0]
    return tokens[-1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # Build debut panel (owner -> earliest filing + first NICE class + serial)
    print("[1/4] debut panel with surprise join …", flush=True)
    debut_parts = []
    for cls in class_list():
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number","owner_name","filing_date"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(pl.lit(cls).alias("class_id"))
        debut_parts.append(tm)
    allf = pl.concat(debut_parts)
    debut = (
        allf.sort(["owner_name","filing_date","class_id"])
            .group_by("owner_name", maintain_order=False)
            .agg(pl.col("filing_date").first().alias("debut_date"),
                 pl.col("class_id").first().alias("class_id"),
                 pl.col("serial_number").first().alias("debut_serial"))
            .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int32).alias("year"))
    )
    print(f"      {debut.height:,} unique owners")

    # Parse surnames + Census join
    print("[2/4] surname → Census probabilities …", flush=True)
    debut = debut.with_columns(pl.Series(
        "surname", [parse_surname(n) if n else None for n in debut["owner_name"].to_list()]
    ))
    surn = pl.read_csv(REF/"census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    joined = debut.filter(pl.col("surname").is_not_null()).join(
        surn.select(["name_u"] + pct_cols), left_on="surname", right_on="name_u", how="inner"
    )
    joined = joined.with_columns([(pl.col(c).fill_null(0.0)/100.0).alias(f"w_{c[3:]}") for c in pct_cols])
    print(f"      {joined.height:,} matched individuals")

    # Now join to per-class surprise to attach ΔKL
    print("[3/4] joining ΔKL from per-class surprise parquets …", flush=True)
    sur_parts = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        if not sp.exists(): continue
        s = pl.read_parquet(sp, columns=["serial_number","kl_vs_past","kl_vs_future",
                                          "n_ref_past","n_ref_future","n_terms"]).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
        ).with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl")
        ).select("serial_number","dkl")
        sur_parts.append(s)
    sur = pl.concat(sur_parts).unique("serial_number")
    print(f"      {sur.height:,} clean surprise rows")
    joined_dkl = joined.join(sur, left_on="debut_serial", right_on="serial_number", how="inner")
    print(f"      {joined_dkl.height:,} matched debut filings with ΔKL")

    # Aggregate by (class, ethnic bucket): weighted-mean ΔKL using ethnic probability
    print("[4/4] computing (class, ethnic) cells …", flush=True)
    weight_cols = [f"w_{c[3:]}" for c in pct_cols]
    rows = []
    pdf = joined_dkl.to_pandas()
    for cls, sub in pdf.groupby("class_id"):
        n_indiv = len(sub)
        if n_indiv < 100:
            continue
        for wc in weight_cols:
            w = sub[wc].to_numpy()
            wsum = w.sum()
            if wsum < 10:
                continue
            mean_dkl = (sub["dkl"].to_numpy() * w).sum() / wsum
            mean_pkl = ((sub["dkl"].to_numpy() > 0).astype(float) * w).sum() / wsum  # share positive
            rows.append({
                "class_id": cls, "ethnic": ETH_LABELS[wc][0],
                "n_eff": wsum, "n_indiv": n_indiv,
                "share_in_class": wsum / n_indiv,
                "mean_dkl": mean_dkl,
                "share_positive_dkl": mean_pkl,
            })
    panel = pl.DataFrame(rows)
    panel = panel.with_columns(
        pl.col("class_id").map_elements(lambda c: INDUSTRY_NAMES.get(c, c), return_dtype=pl.Utf8).alias("class_name")
    )
    panel.write_csv(OUT / "ethnic_cluster_dkl_panel.csv")

    # Identify "enclave" cells: ethnic share within class > 1.5x its overall share
    overall = panel.group_by("ethnic").agg(pl.col("n_eff").sum().alias("n_total"))
    grand = overall["n_total"].sum()
    overall = overall.with_columns((pl.col("n_total")/grand).alias("share_overall"))
    panel2 = panel.join(overall.select(["ethnic","share_overall"]), on="ethnic", how="left")
    panel2 = panel2.with_columns(
        (pl.col("share_in_class") / pl.col("share_overall")).alias("over_index")
    )
    enclave = panel2.filter(
        (pl.col("over_index") > 1.5) & (pl.col("share_in_class") > 0.10) & (pl.col("n_eff") > 500)
    ).sort("over_index", descending=True)
    print("\nEnclave cells (over-represented by > 1.5x, share > 10%, n > 500):")
    print(enclave.to_pandas().to_string(index=False,
                                          columns=["class_id","class_name","ethnic","n_eff",
                                                   "share_in_class","over_index",
                                                   "mean_dkl","share_positive_dkl"],
                                          float_format=lambda x: f"{x:.3f}"))

    # Plot: ethnic share within class vs mean ΔKL of that ethnic group's debuts in the class
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, eth_key in zip(axes.flat, ["White", "API", "Hispanic", "Black"]):
        sub = panel.filter(pl.col("ethnic") == eth_key).to_pandas()
        # Compute over-index for color
        sub = sub.merge(overall.to_pandas(), on="ethnic")
        sub["over_index"] = sub["share_in_class"] / sub["share_overall"]
        sc = ax.scatter(sub["share_in_class"], sub["mean_dkl"],
                        s=np.clip(sub["n_eff"]/30, 8, 240),
                        c=sub["over_index"], cmap="viridis", alpha=0.85,
                        edgecolor="black", lw=0.4)
        # Label the top-N enclave (most over-indexed) classes with their
        # industry NAME. Cap to top 6 so the plot doesn't get unreadable.
        encl = sub[sub["over_index"] > 1.3].sort_values("over_index", ascending=False).head(6)
        for _, r in encl.iterrows():
            label = INDUSTRY_NAMES.get(r["class_id"], r["class_id"])
            # Trim very long names
            if len(label) > 22:
                label = label[:22] + "…"
            ax.annotate(label, (r["share_in_class"], r["mean_dkl"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color="#222",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#888", lw=0.5, alpha=0.85))
        ax.set_xlabel(f"{eth_key}-classified share of individual-name debuts in NICE class")
        ax.set_ylabel(r"Mean $\Delta KL$ (signed; positive = innovation)")
        ax.set_title(f"{eth_key}-classified debut filers, per NICE class")
        ax.axhline(0, color="#444", lw=0.7)
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="over-index vs.\ overall share")
    fig.suptitle("Where ethnic clusters form, and what their vocabulary looks like\n"
                 "(point size ∝ n_effective; labels show top enclave-class industry names)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "ethnic_cluster_dkl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary stats
    print()
    print("Cross-class ΔKL comparison: enclave vs diverse cells, by ethnic group")
    for eth in ["White","API","Hispanic","Black"]:
        sub = panel2.filter(pl.col("ethnic") == eth).to_pandas()
        if not len(sub): continue
        encl = sub[sub["over_index"] > 1.5]
        non_encl = sub[sub["over_index"] <= 1.5]
        if not (len(encl) and len(non_encl)): continue
        # n-weighted mean
        encl_mean = (encl["mean_dkl"] * encl["n_eff"]).sum() / encl["n_eff"].sum()
        nonencl_mean = (non_encl["mean_dkl"] * non_encl["n_eff"]).sum() / non_encl["n_eff"].sum()
        print(f"  {eth:8s}  enclave mean ΔKL = {encl_mean:+.4f}  "
              f"non-enclave = {nonencl_mean:+.4f}  diff = {encl_mean-nonencl_mean:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
