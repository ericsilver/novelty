"""For each owner, compute within-firm NICE-class diversification +
within-firm ΔKL trajectory variance. Compare ethnically-clustered
owners (whose debut class is one of their ethnic group's
over-represented "enclave" classes) to non-clustered owners.

Question: do ethnic-cluster founders stay in their cluster (low
diversification, high within-firm ΔKL variance from cycling vocabulary
within one cluster) or diversify into other classes?

Outputs:
  paper/results/within_firm_diversification.csv
  paper/results/within_firm_diversification.png
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
REF  = REPO / "reference_data"
OUT  = REPO / "paper" / "results"

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

# Enclave classes per ethnic group (from ethnic_cluster_dkl.py output)
ENCLAVE_CLASSES = {
    "API":      {"006","007","008","010","011","012","014","015","017","018",
                 "020","021","022","024","026","027","028"},
    "Hispanic": {"033","043"},
    "White":    set(),
    "Black":    set(),
}


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] building per-owner filing history + ΔKL …", flush=True)
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
        sp = PROC / f"surprise_class{cls}.parquet"
        if sp.exists():
            s = pl.read_parquet(sp, columns=["serial_number","prospective_kl","retrospective_kl",
                                              "n_ref_prospective","n_ref_retrospective","n_terms"]).filter(
                (pl.col("n_ref_prospective") >= 1000)
                & (pl.col("n_ref_retrospective") >= 1000)
                & (pl.col("n_terms") >= 3)
                & pl.col("prospective_kl").is_finite()
                & pl.col("retrospective_kl").is_finite()
            ).with_columns(
                (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl")
            ).select("serial_number","dkl")
            tm = tm.join(s, on="serial_number", how="left")
        parts.append(tm)
        del tm
        gc.collect()
    allf = pl.concat(parts)
    print(f"      {allf.height:,} filings")

    # Per-owner stats: number of distinct classes, ΔKL mean/std, debut class
    print("[2/4] aggregating per owner …", flush=True)
    debut = (
        allf.sort(["owner_name","filing_date","class_id"])
            .group_by("owner_name", maintain_order=False)
            .agg(pl.col("class_id").first().alias("debut_class"),
                 pl.col("filing_date").first().alias("debut_date"),
                 pl.col("class_id").n_unique().alias("n_classes"),
                 pl.len().alias("n_filings"),
                 pl.col("dkl").mean().alias("dkl_mean"),
                 pl.col("dkl").std().alias("dkl_std"))
            .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int32).alias("debut_year"))
    )
    print(f"      {debut.height:,} unique owners")

    # Surname + Census ethnicity
    print("[3/4] surname → Census ethnicity …", flush=True)
    debut = debut.with_columns(pl.Series(
        "surname", [parse_surname(n) if n else None for n in debut["owner_name"].to_list()]
    ))
    surn = pl.read_csv(REF/"census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    joined = debut.filter(pl.col("surname").is_not_null()).join(
        surn.select(["name_u"] + pct_cols),
        left_on="surname", right_on="name_u", how="inner"
    )
    print(f"      {joined.height:,} individual debuts matched")

    # Assign each owner to most-likely-ethnic-group (argmax of Census probs > 50%)
    pdf = joined.to_pandas()
    pct_to_eth = {"pctwhite":"White", "pctblack":"Black", "pctapi":"API",
                  "pcthispanic":"Hispanic", "pct2prace":"2+races", "pctaian":"AIAN"}
    eth_probs = pdf[list(pct_to_eth.keys())].to_numpy()
    top_idx = eth_probs.argmax(axis=1)
    top_prob = eth_probs.max(axis=1)
    eth_keys = list(pct_to_eth.keys())
    pdf["ethnic_top"] = [pct_to_eth[eth_keys[i]] if p >= 50 else "Mixed"
                         for i, p in zip(top_idx, top_prob)]

    # Tag whether debut class is in this owner's ethnic-group enclave list
    pdf["in_enclave"] = pdf.apply(
        lambda r: r["debut_class"] in ENCLAVE_CLASSES.get(r["ethnic_top"], set()),
        axis=1
    )

    # --- 4. Aggregate
    print("[4/4] aggregating diversification stats …", flush=True)
    out_rows = []
    for (eth, in_enc), sub in pdf.groupby(["ethnic_top","in_enclave"]):
        if len(sub) < 50: continue
        out_rows.append({
            "ethnic": eth, "in_enclave": in_enc, "n_owners": len(sub),
            "share_single_class": (sub["n_classes"] == 1).mean(),
            "mean_n_classes": sub["n_classes"].mean(),
            "median_n_classes": sub["n_classes"].median(),
            "mean_n_filings": sub["n_filings"].mean(),
            "mean_dkl": sub["dkl_mean"].mean(),
            "mean_dkl_std": sub["dkl_std"].dropna().mean(),
        })
    res = pl.DataFrame(out_rows).sort(["ethnic","in_enclave"])
    res.write_csv(OUT / "within_firm_diversification.csv")
    print(res.to_pandas().to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot: bar chart of share_single_class for ethnic × in_enclave
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    res_pd = res.to_pandas()
    # Subplot 1: share staying single-class
    eths = ["White","API","Hispanic","Black"]
    x = np.arange(len(eths))
    width = 0.35
    for j, in_enc in enumerate([False, True]):
        vals = []
        for eth in eths:
            r = res_pd[(res_pd["ethnic"] == eth) & (res_pd["in_enclave"] == in_enc)]
            vals.append(r["share_single_class"].iloc[0] if len(r) else 0)
        axes[0].bar(x + width*(j-0.5), vals, width,
                    label="debut in own-group enclave" if in_enc else "debut not in enclave")
    axes[0].set_xticks(x); axes[0].set_xticklabels(eths)
    axes[0].set_ylabel("Share of owners with only 1 NICE class")
    axes[0].set_title("Within-firm class concentration:\nenclave-rooted owners stay more single-class")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    # Subplot 2: mean within-firm ΔKL std
    for j, in_enc in enumerate([False, True]):
        vals = []
        for eth in eths:
            r = res_pd[(res_pd["ethnic"] == eth) & (res_pd["in_enclave"] == in_enc)]
            vals.append(r["mean_dkl_std"].iloc[0] if len(r) else 0)
        axes[1].bar(x + width*(j-0.5), vals, width,
                    label="debut in own-group enclave" if in_enc else "debut not in enclave")
    axes[1].set_xticks(x); axes[1].set_xticklabels(eths)
    axes[1].set_ylabel(r"Mean within-firm $\Delta KL$ std")
    axes[1].set_title("Within-firm vocab volatility (intra-firm $\\Delta KL$ std)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "within_firm_diversification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] {OUT/'within_firm_diversification.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
