"""Long-term benefit of industry persistence: compare firms that stayed
in one NICE class (single-class concentrators) to firms that diversified
(multi-class), on survival and SEC-listing outcomes. Stratify by
surname-imputed ethnic group + enclave/non-enclave debut.

Outputs:
  paper/results/industry_persistence.csv
  paper/results/industry_persistence.png
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

    print("[1/3] per-owner filing history + outcomes …", flush=True)
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name").unique().with_columns(
        pl.lit(True).alias("in_sec")
    )
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
        op = PROC / f"outcomes_class{cls}.parquet"
        if op.exists():
            o = pl.read_parquet(op, columns=["serial_number","reached_registration",
                                              "currently_live","survived_5y"])
            tm = tm.join(o, on="serial_number", how="left")
        parts.append(tm)
        del tm
    allf = pl.concat(parts)
    print(f"      {allf.height:,} filings with outcomes")

    print("[2/3] aggregate per owner …", flush=True)
    own = (
        allf.sort(["owner_name","filing_date","class_id"])
            .group_by("owner_name", maintain_order=False)
            .agg(
                pl.col("class_id").first().alias("debut_class"),
                pl.col("filing_date").first().alias("debut_date"),
                pl.col("class_id").n_unique().alias("n_classes"),
                pl.len().alias("n_filings"),
                pl.col("survived_5y").cast(pl.Float64).mean().alias("survived_5y_rate"),
                pl.col("reached_registration").cast(pl.Float64).mean().alias("reached_rate"),
            )
            .with_columns(pl.col("debut_date").str.slice(0,4).cast(pl.Int32).alias("debut_year"))
            .join(sec, on="owner_name", how="left")
            .with_columns(pl.col("in_sec").fill_null(False))
    )
    # Restrict to firms with debut_year 1995-2018 so the 5y outcome window has data
    own = own.filter(pl.col("debut_year").is_between(1995, 2018))

    # Diversification tier
    own = own.with_columns(
        pl.when(pl.col("n_classes") == 1).then(pl.lit("single"))
          .when(pl.col("n_classes") <= 3).then(pl.lit("2-3 classes"))
          .otherwise(pl.lit("4+ classes")).alias("div_tier")
    )

    print("[3/3] surname + outcomes by diversification tier …", flush=True)
    surn = pl.read_csv(REF/"census_surnames/Names_2010Census.csv", null_values=["(S)"])
    surn = surn.with_columns(pl.col("name").str.to_uppercase().alias("name_u"))
    own = own.with_columns(pl.Series(
        "surname", [parse_surname(n) if n else None for n in own["owner_name"].to_list()]
    ))
    pct_cols = ["pctwhite","pctblack","pctapi","pctaian","pct2prace","pcthispanic"]
    indiv = own.filter(pl.col("surname").is_not_null()).join(
        surn.select(["name_u"] + pct_cols), left_on="surname", right_on="name_u", how="inner"
    )

    # Argmax ethnic
    pdf = indiv.to_pandas()
    eth_map = {"pctwhite":"White","pctblack":"Black","pctapi":"API",
               "pcthispanic":"Hispanic","pct2prace":"2+races","pctaian":"AIAN"}
    eth_arr = pdf[list(eth_map.keys())].to_numpy()
    pdf["ethnic_top"] = [list(eth_map.values())[i] if p >= 50 else "Mixed"
                          for i, p in zip(eth_arr.argmax(axis=1), eth_arr.max(axis=1))]

    # Main aggregation: outcomes by diversification tier × ethnic
    out_rows = []
    for (eth, div_tier), sub in pdf.groupby(["ethnic_top","div_tier"]):
        if len(sub) < 100: continue
        out_rows.append({
            "ethnic": eth, "div_tier": div_tier, "n_owners": len(sub),
            "mean_n_filings": sub["n_filings"].mean(),
            "p_reached":      sub["reached_rate"].mean(),
            "p_survived_5y":  sub["survived_5y_rate"].mean(),
            "p_in_sec":       sub["in_sec"].mean(),
        })
    res = pl.DataFrame(out_rows).sort(["ethnic","div_tier"])
    res.write_csv(OUT / "industry_persistence.csv")
    print(res.to_pandas().to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Also pooled, ignoring ethnicity, on the full owner population
    own_pdf = own.to_pandas()
    pooled = own_pdf.groupby("div_tier").agg(
        n_owners=("owner_name","count"),
        mean_n_filings=("n_filings","mean"),
        p_reached=("reached_rate","mean"),
        p_survived_5y=("survived_5y_rate","mean"),
        p_in_sec=("in_sec","mean"),
    ).reset_index()
    print()
    print("Pooled (all owners 1995-2018):")
    print(pooled.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plot: 2x2 of p_survived_5y and p_in_sec by div_tier x ethnic
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    eths = ["White","API","Hispanic","Black"]
    tiers = ["single","2-3 classes","4+ classes"]
    x = np.arange(len(eths))
    width = 0.27
    res_pd = res.to_pandas()
    for j, tier in enumerate(tiers):
        s5 = [res_pd[(res_pd["ethnic"]==e) & (res_pd["div_tier"]==tier)]["p_survived_5y"].iloc[0]
              if len(res_pd[(res_pd["ethnic"]==e) & (res_pd["div_tier"]==tier)]) else np.nan
              for e in eths]
        ps = [res_pd[(res_pd["ethnic"]==e) & (res_pd["div_tier"]==tier)]["p_in_sec"].iloc[0]
              if len(res_pd[(res_pd["ethnic"]==e) & (res_pd["div_tier"]==tier)]) else np.nan
              for e in eths]
        axes[0].bar(x + width*(j-1), s5, width, label=tier)
        axes[1].bar(x + width*(j-1), ps, width, label=tier)
    for ax, ttl, ylab in zip(axes,
                             ["P(survived 5y | reached registration -- proxy)", "P(owner in SEC EDGAR)"],
                             ["Survival rate", "SEC listing rate"]):
        ax.set_xticks(x); ax.set_xticklabels(eths)
        ax.set_ylabel(ylab); ax.set_title(ttl)
        ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9, title="Owner diversification")
    fig.suptitle("Outcomes by within-firm class diversification × surname-imputed ethnicity\n"
                 "(debut cohorts 1995-2018; individual-name owners only)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "industry_persistence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
