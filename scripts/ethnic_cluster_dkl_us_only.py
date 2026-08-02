"""US-only re-run of the ethnic-cluster ΔKL analysis.

Same as scripts/ethnic_cluster_dkl.py but restricted to owners whose
debut filing's serial_number maps to owner_country == 'US' in
data/processed/owner_address.parquet.

Outputs:
  paper/results/ethnic_cluster_dkl_us_panel.csv
  paper/results/ethnic_cluster_dkl_us.png
"""

from __future__ import annotations

import gc, re
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
REF  = REPO / "reference_data"
OUT  = REPO / "paper" / "results"

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

ETH_LABELS = {
    "w_white":    ("White",    "#2b6cb0"),
    "w_black":    ("Black",    "#cc4444"),
    "w_api":      ("API",      "#229922"),
    "w_hispanic": ("Hispanic", "#cc7a00"),
    "w_2prace":   ("2+races",  "#999999"),
    "w_aian":     ("AIAN",     "#7a3eb2"),
}
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] address parquet …", flush=True)
    addr = pl.read_parquet(PROC / "owner_address.parquet",
                            columns=["serial_number","owner_country"])

    print("[2/5] debut panel + surprise …", flush=True)
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
                 .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int32).alias("year")))
    # Filter to US-domiciled at debut
    debut = debut.join(addr, left_on="debut_serial", right_on="serial_number", how="left")
    debut_us = debut.filter(pl.col("owner_country") == "US")
    print(f"      US-domiciled owners: {debut_us.height:,}")

    print("[3/5] surname + Census probabilities …", flush=True)
    debut_us = debut_us.with_columns(pl.Series(
        "surname", [parse_surname(n) if n else None for n in debut_us["owner_name"].to_list()]
    ))
    surn = pl.read_csv(REF/"census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    joined = debut_us.filter(pl.col("surname").is_not_null()).join(
        surn.select(["name_u"] + pct_cols),
        left_on="surname", right_on="name_u", how="inner",
    ).with_columns([(pl.col(c).fill_null(0.0)/100.0).alias(f"w_{c[3:]}") for c in pct_cols])

    print("[4/5] joining ΔKL …", flush=True)
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
    joined = joined.join(sur, left_on="debut_serial", right_on="serial_number", how="inner")
    print(f"      matched debut filings with ΔKL: {joined.height:,}")

    print("[5/5] (class, ethnic) cells …", flush=True)
    weight_cols = [f"w_{c[3:]}" for c in pct_cols]
    pdf = joined.to_pandas()
    rows = []
    for cls, sub in pdf.groupby("class_id"):
        n_indiv = len(sub)
        if n_indiv < 100: continue
        for wc in weight_cols:
            w = sub[wc].to_numpy(); wsum = w.sum()
            if wsum < 10: continue
            mean_dkl = (sub["dkl"].to_numpy()*w).sum() / wsum
            rows.append({
                "class_id": cls, "ethnic": ETH_LABELS[wc][0],
                "n_eff": wsum, "n_indiv": n_indiv,
                "share_in_class": wsum / n_indiv,
                "mean_dkl": mean_dkl,
            })
    panel = pl.DataFrame(rows).with_columns(
        pl.col("class_id").map_elements(lambda c: INDUSTRY_NAMES.get(c, c), return_dtype=pl.Utf8).alias("class_name")
    )
    panel.write_csv(OUT / "ethnic_cluster_dkl_us_panel.csv")

    # Enclave identification (same rule as before)
    overall = panel.group_by("ethnic").agg(pl.col("n_eff").sum().alias("n_total"))
    grand = overall["n_total"].sum()
    overall = overall.with_columns((pl.col("n_total")/grand).alias("share_overall"))
    panel2 = panel.join(overall.select(["ethnic","share_overall"]), on="ethnic", how="left").with_columns(
        (pl.col("share_in_class") / pl.col("share_overall")).alias("over_index")
    )
    encl = panel2.filter(
        (pl.col("over_index") > 1.3) & (pl.col("share_in_class") > 0.08) & (pl.col("n_eff") > 200)
    ).sort("over_index", descending=True)
    print("\nUS-ONLY Enclave cells (over-index > 1.3, share > 8%, n_eff > 200):")
    print(encl.to_pandas().to_string(index=False,
                                     columns=["class_id","class_name","ethnic","n_eff",
                                              "share_in_class","over_index","mean_dkl"],
                                     float_format=lambda x: f"{x:.3f}"))

    # Plot 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, eth_key in zip(axes.flat, ["White","API","Hispanic","Black"]):
        sub = panel.filter(pl.col("ethnic") == eth_key).to_pandas()
        sub = sub.merge(overall.to_pandas(), on="ethnic")
        sub["over_index"] = sub["share_in_class"] / sub["share_overall"]
        sc = ax.scatter(sub["share_in_class"], sub["mean_dkl"],
                        s=np.clip(sub["n_eff"]/15, 8, 240),
                        c=sub["over_index"], cmap="viridis", alpha=0.85,
                        edgecolor="black", lw=0.4)
        # Label top enclave classes by industry name
        encl_sub = sub[sub["over_index"] > 1.2].sort_values("over_index", ascending=False).head(6)
        for _, r in encl_sub.iterrows():
            label = INDUSTRY_NAMES.get(r["class_id"], r["class_id"])
            if len(label) > 22: label = label[:22] + "…"
            ax.annotate(label, (r["share_in_class"], r["mean_dkl"]),
                        xytext=(5,5), textcoords="offset points", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", lw=0.5, alpha=0.85))
        ax.set_xlabel(f"{eth_key} share of US-domiciled individual-name debuts in NICE class")
        ax.set_ylabel(r"Mean $\Delta KL$ (signed; positive = innovation)")
        ax.set_title(f"{eth_key}-classified US debut filers per NICE class")
        ax.axhline(0, color="#444", lw=0.7)
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="over-index")
    fig.suptitle("US-only: ethnic clusters in trademark debut filings\n"
                 "(point size ∝ n_effective; labels show top enclave industry names)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "ethnic_cluster_dkl_us.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Enclave vs non-enclave summary
    print()
    print("US-ONLY enclave-vs-non-enclave mean ΔKL, by ethnic group:")
    for eth in ["White","API","Hispanic","Black"]:
        sub = panel2.filter(pl.col("ethnic") == eth).to_pandas()
        if not len(sub): continue
        encl_g = sub[sub["over_index"] > 1.3]
        non_g = sub[sub["over_index"] <= 1.3]
        if not (len(encl_g) and len(non_g)): continue
        em = (encl_g["mean_dkl"]*encl_g["n_eff"]).sum()/encl_g["n_eff"].sum()
        nm = (non_g["mean_dkl"]*non_g["n_eff"]).sum()/non_g["n_eff"].sum()
        print(f"  {eth:8s}  enclave ΔKL = {em:+.4f}  non-enclave = {nm:+.4f}  diff = {em-nm:+.4f}  (n_encl={len(encl_g)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
