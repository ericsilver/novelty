"""Within-firm ΔKL trajectory analysis on the US-only sample.

For each US-domiciled owner with ≥3 clean filings, compute:
  - debut ΔKL  (first filing)
  - mean ΔKL of subsequent filings
  - drift = subsequent_mean − debut

Then stratify by surname-imputed ethnicity (argmax) and by whether
the debut class is an enclave class for that ethnicity.

Question: do firms drift away from their debut ΔKL (regression to
the corpus mean)?  Do enclave-rooted firms drift differently?  If
ethnic-enclave firms drift toward zero (mean reversion) while
non-enclave firms hold, that's a sign the enclave is a 'first
filing' phenomenon that doesn't persist across the firm's
trademark portfolio.  If enclave firms hold their elevated ΔKL,
the cluster is a structural feature of how the firm operates.

Inputs:
  data/processed/tm_class*.parquet
  data/processed/surprise_class*.parquet
  data/processed/owner_address.parquet
  reference_data/census_surnames/Names_2010Census.csv

Outputs:
  paper/results/within_firm_dkl_trajectory.csv
  paper/results/within_firm_dkl_trajectory.png
"""

from __future__ import annotations

import gc, re
from pathlib import Path

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
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

# Enclave classes per ethnic group, from US-only ethnic_cluster_dkl_us run
ENCLAVE_CLASSES = {
    "API":      {"003","005","007","008","010","011","014","018","021","024","026","029","030","034","043"},
    "Hispanic": {"033","043"},
    "White":    set(),
    "Black":    set(),
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

    print("[1/4] address parquet …", flush=True)
    addr = pl.read_parquet(PROC / "owner_address.parquet",
                            columns=["serial_number","owner_country"])

    print("[2/4] per-owner filing histories + ΔKL …", flush=True)
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
    allf = pl.concat(parts)
    # Join address per filing
    allf = allf.join(addr, on="serial_number", how="left")
    # Sort filings within each owner by date
    allf = allf.sort(["owner_name","filing_date","class_id"])
    print(f"      {allf.height:,} filings")

    print("[3/4] computing per-owner debut + subsequent stats …", flush=True)
    # Use polars groupby to compute first and rest
    own = (
        allf.group_by("owner_name")
            .agg(
                pl.col("filing_date").first().alias("debut_date"),
                pl.col("class_id").first().alias("debut_class"),
                pl.col("dkl").first().alias("debut_dkl"),
                pl.col("owner_country").first().alias("debut_country"),
                pl.len().alias("n_filings"),
                pl.col("dkl").drop_nulls().alias("dkl_history"),
            )
            .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int32).alias("debut_year"))
    )
    # US-only at debut + at least 3 clean filings
    own_us = own.filter(
        (pl.col("debut_country") == "US")
        & (pl.col("n_filings") >= 3)
        & pl.col("debut_dkl").is_not_null()
    )
    print(f"      {own_us.height:,} US-domiciled owners with ≥3 filings + debut ΔKL")

    # Compute subsequent_mean = mean of dkl_history excluding the first element
    pdf = own_us.to_pandas()
    def subsequent_mean(arr):
        # arr is a list of ΔKL values for that owner; first element is debut.
        if arr is None or len(arr) < 2: return np.nan
        rest = arr[1:]
        rest_clean = [v for v in rest if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(rest_clean)) if rest_clean else np.nan
    pdf["subsequent_mean_dkl"] = pdf["dkl_history"].apply(subsequent_mean)
    pdf["drift"] = pdf["subsequent_mean_dkl"] - pdf["debut_dkl"]
    pdf = pdf.dropna(subset=["debut_dkl","subsequent_mean_dkl"])
    print(f"      {len(pdf):,} owners with valid debut + subsequent ΔKL")

    # Surname + ethnicity
    pdf["surname"] = pdf["owner_name"].apply(lambda x: parse_surname(x) if x else None)
    indiv = pdf.dropna(subset=["surname"]).copy()
    surn = pd.read_csv(REF / "census_surnames/Names_2010Census.csv", na_values=["(S)"])
    surn["name_u"] = surn["name"].str.upper()
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    surn = surn[["name_u"] + pct_cols].copy()
    surn[pct_cols] = surn[pct_cols].fillna(0.0)
    indiv = indiv.merge(surn, left_on="surname", right_on="name_u", how="inner")
    print(f"      {len(indiv):,} individual owners matched Census surname")

    # Argmax ethnic
    eth_map = {"pctwhite":"White","pctblack":"Black","pctapi":"API",
               "pcthispanic":"Hispanic","pct2prace":"2+races","pctaian":"AIAN"}
    eth_arr = indiv[list(eth_map.keys())].to_numpy()
    indiv["ethnic_top"] = [list(eth_map.values())[i] if p >= 50 else "Mixed"
                           for i, p in zip(eth_arr.argmax(axis=1), eth_arr.max(axis=1))]
    indiv["in_enclave"] = indiv.apply(
        lambda r: r["debut_class"] in ENCLAVE_CLASSES.get(r["ethnic_top"], set()),
        axis=1
    )

    print("[4/4] aggregating + plotting …", flush=True)
    # Aggregate: mean drift by (ethnic, in_enclave, debut_dkl_sign)
    indiv["debut_sign"] = np.where(indiv["debut_dkl"] > 0, "positive", "negative")
    rows = []
    for (eth, in_enc, sign), sub in indiv.groupby(["ethnic_top","in_enclave","debut_sign"]):
        if len(sub) < 100: continue
        rows.append({
            "ethnic": eth, "in_enclave": in_enc, "debut_sign": sign,
            "n_owners": len(sub),
            "mean_debut": sub["debut_dkl"].mean(),
            "mean_subsequent": sub["subsequent_mean_dkl"].mean(),
            "mean_drift": sub["drift"].mean(),
            "median_drift": sub["drift"].median(),
        })
    res = pd.DataFrame(rows).sort_values(["ethnic","in_enclave","debut_sign"])
    res.to_csv(OUT / "within_firm_dkl_trajectory.csv", index=False)
    print()
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot: drift by ethnic × in_enclave
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    eths = ["White","API","Hispanic","Black"]
    x = np.arange(len(eths))
    width = 0.35
    for j, in_enc in enumerate([False, True]):
        vals = []
        for eth in eths:
            r = res[(res["ethnic"] == eth) & (res["in_enclave"] == in_enc)]
            if len(r):
                # n-weighted mean drift across debut signs
                v = (r["mean_drift"] * r["n_owners"]).sum() / r["n_owners"].sum()
            else:
                v = np.nan
            vals.append(v)
        axes[0].bar(x + width*(j - 0.5), vals, width,
                    label="debut in enclave" if in_enc else "debut not in enclave")
    axes[0].set_xticks(x); axes[0].set_xticklabels(eths)
    axes[0].set_ylabel(r"Mean drift = subsequent $\Delta KL$ − debut $\Delta KL$")
    axes[0].set_title("Within-firm ΔKL drift after debut\n(positive = trends toward innovation; negative = mean reversion)")
    axes[0].axhline(0, color="#444", lw=0.7)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=9)

    # Right: drift vs debut ΔKL bin (regression-to-mean check, pooled)
    indiv_clip = indiv[indiv["debut_dkl"].between(-1, 1)]
    n_bins = 12
    cuts = np.quantile(indiv_clip["debut_dkl"], np.linspace(0, 1, n_bins+1))
    indiv_clip = indiv_clip.assign(bin=np.clip(np.searchsorted(cuts[1:-1], indiv_clip["debut_dkl"]), 0, n_bins-1))
    g = indiv_clip.groupby("bin").agg(
        x=("debut_dkl","mean"), y=("subsequent_mean_dkl","mean"),
        n=("debut_dkl","size")
    ).reset_index()
    axes[1].plot(g["x"], g["y"], "-o", color="#2b6cb0", ms=6, lw=1.7, label="Mean subsequent ΔKL")
    # 45-degree line for reference
    xrange = np.linspace(g["x"].min(), g["x"].max(), 50)
    axes[1].plot(xrange, xrange, "--", color="#888", lw=1, label="y = x (no drift)")
    axes[1].plot(xrange, np.zeros_like(xrange), ":", color="#444", lw=1, label="y = 0 (full mean reversion)")
    axes[1].set_xlabel("Debut filing ΔKL (binned)")
    axes[1].set_ylabel("Mean subsequent-filings ΔKL")
    axes[1].set_title("Regression to corpus mean:\nlogistic flattening between debut and rest of firm's filings")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "within_firm_dkl_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] {OUT/'within_firm_dkl_trajectory.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
