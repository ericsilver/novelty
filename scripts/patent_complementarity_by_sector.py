"""Referee check (Castaldi): does the pooled dKL-patent orthogonality mask
sector heterogeneity where trademarks and patents are complementary IPRs?

The paper reports a pooled within-firm coefficient of dKL on log(1+patents)
of -0.048 sigma (t = -7.9) under the published exponential-decay scoring and
-0.010 (t = -1.5) under topic scoring at T = 200, and reads that near-zero as
evidence the construct is not a noisy patent proxy. A pooled near-zero is
consistent with two very different worlds: (a) genuine orthogonality
everywhere, or (b) a positive slope in sectors where the two IPRs are
complements (pharma, medical devices, electronics, chemicals) offset by a
zero/negative slope where trademarks do the work alone (retail, apparel,
entertainment, personal services). This script distinguishes them.

Design. Sector is assigned at the FIRM level (dominant NICE class = the class
carrying the plurality of a firm's scored filings, pooled over all years), so
the estimating equation inside every subsample is character-for-character the
pooled specification -- firm + year FE, firm-clustered SE, firms with >= 3
years -- and nothing about the estimator changes across cells. dKL is
standardized ONCE on the pooled matched panel, so every reported coefficient
is in the same sigma units as Table~\\ref{tab:patent_within_firm} and is
directly comparable to -0.048 / -0.010.

Three cuts are reported, in increasing order of how much the a-priori theory
is doing:
  1. Per-NICE-class estimates for all 45 classes (no grouping imposed).
  2. Castaldi's own goods (NICE 1-34) vs services (NICE 35-45) split.
  3. The classic IPR complement / neutral / substitute grouping (below).
  4. A pooled interaction, dkl_z ~ logpat + logpat x complement, where the
     firm FE absorbs the level of the firm-constant group dummy and the
     interaction is identified off the slope difference.

Cut 1 is the load-bearing one: if the a-priori grouping in cut 3 turns out to
be wrong, the per-class table still answers the referee.

A secondary block splits by SEC SIC division instead of NICE class, for the
subset of owners carried in the USPTO-SEC crosswalk. That sample is far
smaller (roughly 6.5k owners have a SIC at all, before intersecting with the
patent match), so it is a robustness read, not the headline.

Inputs :  data/processed/tm_class{cls}.parquet            (serial -> owner, per class)
          data/processed/surprise_class{cls}.parquet      (token scoring)
          data/processed/topic_surprise_class{cls}_T200.parquet  (topic scoring)
          data_publish/firm_year_patents_and_dkl.csv      (n_patents per owner-year)
          data/processed/uspto_sec_crosswalk.parquet      (owner_name -> cik)
          data/processed/sec_firm_year.parquet            (cik -> sic)
Output :  paper/results/patent_complementarity_by_sector.json  (+ stderr summary)
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
PUB = REPO / "data_publish"
RES = REPO / "paper" / "results"

YEAR_LO, YEAR_HI = 1995, 2019

# Minimum cell size before a subsample estimate is reported at all. Firm-
# clustered SEs on fewer than ~50 clusters are not to be believed.
MIN_FIRMS = 50
MIN_OBS = 300

# ---------------------------------------------------------------------------
# A-priori sector grouping.
#
# COMPLEMENTS -- sectors where the patent and trademark systems are standardly
# treated as complementary IPRs: a patented technical advance is brought to
# market under a protected brand, so the two filings track the same product
# launch. This is the textbook complementarity set (pharmaceuticals, medical
# devices, chemicals, electronics/software, machinery, vehicles) and it is
# where Castaldi's conjecture predicts a POSITIVE within-firm slope.
#
# SUBSTITUTES -- sectors where trademarks are the primary appropriation
# mechanism and patenting is rare or irrelevant: retail and advertising,
# apparel and textiles, food and beverages, entertainment and education,
# hospitality, finance and insurance, personal and legal services. Predicted
# slope zero or negative (the paper's temporal-substitution channel).
#
# NEUTRAL -- everything with mixed or thin priors, kept out of the headline
# contrast rather than forced into one side of it.
#
# NICE 042 (scientific and technological services, incl. software design and
# R&D) is placed with the complements: it is the service class that most
# directly hosts patentable technical work. NICE 038 (telecommunications) is
# left neutral -- the industry patents heavily but the trademark class covers
# the service, not the apparatus, which sits in 009.
# ---------------------------------------------------------------------------
COMPLEMENT_CLASSES = {
    "001",  # chemicals for industry/science
    "002",  # paints, varnishes, anti-corrosives
    "004",  # industrial oils, lubricants, fuels
    "005",  # pharmaceuticals, veterinary, sanitary preparations
    "007",  # machines and machine tools
    "009",  # electrical/scientific apparatus, computer hardware and software
    "010",  # surgical, medical, dental apparatus
    "011",  # environmental control apparatus (heating, cooling, water)
    "012",  # vehicles
    "017",  # rubber, plastics, insulation
    "042",  # scientific and technological services, software design, R&D
}
SUBSTITUTE_CLASSES = {
    "014",  # jewelry, precious metals
    "018",  # leather goods, luggage
    "023",  # yarns and threads
    "024",  # fabrics
    "025",  # clothing, footwear, headgear
    "028",  # toys, sporting goods
    "029",  # meats and processed foods
    "030",  # staple foods
    "031",  # natural agricultural products
    "032",  # light beverages
    "033",  # wines and spirits
    "035",  # advertising and business services, retail
    "036",  # insurance and financial services
    "037",  # construction and repair
    "039",  # transportation and storage
    "041",  # education and entertainment
    "043",  # restaurants and hotels
    "045",  # personal, legal, and social services
}
# NEUTRAL is the residual: 003, 006, 008, 013, 015, 016, 019, 020, 021, 022,
# 026, 027, 034, 038, 040, 044.

# SIC division boundaries (first two digits), used for the secondary cut.
SIC_DIVISIONS = [
    (1, 9, "Agriculture, forestry, fishing"),
    (10, 14, "Mining"),
    (15, 17, "Construction"),
    (20, 39, "Manufacturing"),
    (40, 49, "Transportation and utilities"),
    (50, 51, "Wholesale trade"),
    (52, 59, "Retail trade"),
    (60, 67, "Finance, insurance, real estate"),
    (70, 89, "Services"),
    (91, 99, "Public administration"),
]
# Manufacturing is far too coarse for this question, so the patent-intensive
# two-digit majors are broken out individually.
SIC_MAJOR_SPLIT = {
    "28": "Chemicals and pharmaceuticals (SIC 28)",
    "35": "Industrial and computer machinery (SIC 35)",
    "36": "Electronics and electrical equipment (SIC 36)",
    "38": "Instruments and medical devices (SIC 38)",
    "37": "Transportation equipment (SIC 37)",
    "73": "Business services incl. software (SIC 73)",
}


def nice_group(cls: str) -> str:
    if cls in COMPLEMENT_CLASSES:
        return "complement"
    if cls in SUBSTITUTE_CLASSES:
        return "substitute"
    return "neutral"


def goods_services(cls: str) -> str:
    return "goods" if int(cls) <= 34 else "services"


def sic_group(sic: str) -> str:
    s = (sic or "").strip()
    if len(s) < 2 or not s[:2].isdigit():
        return ""
    major = s[:2]
    if major in SIC_MAJOR_SPLIT:
        return SIC_MAJOR_SPLIT[major]
    m = int(major)
    for lo, hi, label in SIC_DIVISIONS:
        if lo <= m <= hi:
            return label
    return ""


def all_classes() -> list[str]:
    return sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def owner_year_class_panel(kind: str) -> pl.DataFrame:
    """kind in {token, t200}: owner x year x NICE-class mean dkl + filing count.

    Identical per-class construction to scripts/topic_patent_expert_rescore.py
    (same filters, same year window, same join), except the NICE class is
    carried through instead of being collapsed away.
    """
    parts = []
    for cls in all_classes():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                             columns=["serial_number", "owner_name"])
        if kind == "token":
            s = pl.read_parquet(
                PROC / f"surprise_class{cls}.parquet",
                columns=["serial_number", "year", "prospective_kl",
                         "retrospective_kl", "n_ref_prospective",
                         "n_ref_retrospective", "n_terms"],
            ).filter(
                (pl.col("n_ref_prospective") >= 1000)
                & (pl.col("n_ref_retrospective") >= 1000)
                & (pl.col("n_terms") >= 3)
                & pl.col("prospective_kl").is_finite()
                & pl.col("retrospective_kl").is_finite()
                & pl.col("year").is_between(YEAR_LO, YEAR_HI)
            ).with_columns(
                (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"))
        else:
            s = pl.read_parquet(
                PROC / f"topic_surprise_class{cls}_T200.parquet"
            ).filter(pl.col("topic_dkl").is_finite()
                     & pl.col("year").is_between(YEAR_LO, YEAR_HI)
            ).rename({"topic_dkl": "dkl"})
        j = s.select("serial_number", "year", "dkl").join(
            tm, on="serial_number", how="inner").filter(
            pl.col("owner_name").is_not_null())
        parts.append(
            j.group_by(["owner_name", "year"]).agg(
                pl.len().alias("n_scored"),
                pl.col("dkl").mean().alias("mean_dkl"),
            ).with_columns(pl.lit(cls).alias("nice_class"))
        )
        del tm, s, j
        gc.collect()
    return pl.concat(parts)


def dominant_class(oyc: pl.DataFrame) -> pl.DataFrame:
    """Firm-constant sector: the NICE class holding the plurality of a firm's
    scored filings across the whole window. Ties break on the lower class
    number so the assignment is deterministic."""
    tot = oyc.group_by(["owner_name", "nice_class"]).agg(
        pl.col("n_scored").sum().alias("cls_scored"))
    return (tot.sort(["owner_name", "cls_scored", "nice_class"],
                     descending=[False, True, False])
               .group_by("owner_name", maintain_order=True)
               .first()
               .select("owner_name", "nice_class", "cls_scored"))


def collapse_owner_year(oyc: pl.DataFrame) -> pl.DataFrame:
    """Filing-weighted collapse across classes to one row per owner-year --
    the same aggregation the pooled panel uses."""
    return oyc.group_by(["owner_name", "year"]).agg(
        (pl.col("mean_dkl") * pl.col("n_scored")).sum().alias("_wd"),
        pl.col("n_scored").sum().alias("n_scored"),
    ).with_columns((pl.col("_wd") / pl.col("n_scored")).alias("mean_dkl")).drop("_wd")


def two_way_fe(df, yvar, xvars, firm_col, year_col):
    """Firm within-transform + firm-demeaned year dummies, firm-clustered SE.

    Byte-for-byte the estimator used in scripts/wsC_within_firm_patents.py and
    scripts/topic_patent_expert_rescore.py.
    """
    d = df[[firm_col, year_col, yvar] + xvars].dropna().copy()

    def demean(s):
        return s - s.groupby(d[firm_col]).transform("mean")

    yd = demean(d[yvar])
    Xparts = {x: demean(d[x]) for x in xvars}
    yr = pd.get_dummies(d[year_col].astype("category"), drop_first=True).astype(float)
    yr.index = d.index
    yr_d = yr.apply(lambda col: col - col.groupby(d[firm_col]).transform("mean"))
    X = pd.concat([pd.DataFrame(Xparts, index=d.index), yr_d], axis=1).astype(float)
    res = sm.OLS(yd.astype(float), X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d[firm_col].astype("category").cat.codes.values})
    out = {x: dict(coef=float(res.params[x]), se=float(res.bse[x]),
                   t=float(res.tvalues[x]), p=float(res.pvalues[x])) for x in xvars}
    out["n_obs"] = int(len(d))
    out["n_firms"] = int(d[firm_col].nunique())
    return out


def safe_fe(df, yvar, xvars, firm_col, year_col, label):
    """FE with the small-cell guard and a singularity catch."""
    d = df[[firm_col, year_col, yvar] + xvars].dropna()
    n_firms = d[firm_col].nunique()
    if n_firms < MIN_FIRMS or len(d) < MIN_OBS:
        return {"skipped": "cell too small", "n_obs": int(len(d)),
                "n_firms": int(n_firms)}
    try:
        return two_way_fe(df, yvar, xvars, firm_col, year_col)
    except Exception as exc:  # singular design in a thin cell
        return {"skipped": f"{type(exc).__name__}: {exc}",
                "n_obs": int(len(d)), "n_firms": int(n_firms)}


def load_patents() -> pd.DataFrame:
    """n_patents per owner-year. The published CSV carries one row per matched
    PatentsView assignee name, so a handful of owner-years appear more than
    once; averaging across matched assignees is what the pooled spec does."""
    pat = pd.read_csv(PUB / "firm_year_patents_and_dkl.csv")
    pat["uspto_owner_name"] = pat.uspto_owner_name.fillna("").astype(str).str.strip()
    pat = pat[pat.uspto_owner_name != ""]
    return pat.groupby(["uspto_owner_name", "year"], as_index=False).agg(
        n_patents=("n_patents", "mean"))


def build_estimation_frame(oyc: pl.DataFrame, pat: pd.DataFrame):
    """Owner-year frame with patents merged, dkl standardized on the POOLED
    matched panel, sector labels attached, and the >= 3-year restriction
    applied -- exactly the pooled sample, plus sector columns."""
    dom = dominant_class(oyc).to_pandas()
    oy = collapse_owner_year(oyc).to_pandas()

    m = oy.merge(pat, left_on=["owner_name", "year"],
                 right_on=["uspto_owner_name", "year"], how="inner")
    m["logpat"] = np.log1p(m.n_patents)
    m["log_nscored"] = np.log1p(m.n_scored)
    sd = float(m.mean_dkl.std())
    m["dkl_z"] = (m.mean_dkl - m.mean_dkl.mean()) / sd

    yr_counts = m.groupby("owner_name").year.transform("count")
    p3 = m[yr_counts >= 3].copy()

    p3 = p3.merge(dom[["owner_name", "nice_class"]], on="owner_name", how="left")
    p3["nice_group"] = p3.nice_class.map(nice_group)
    p3["goods_services"] = p3.nice_class.map(goods_services)
    p3["is_complement"] = (p3.nice_group == "complement").astype(float)
    p3["logpat_x_comp"] = p3.logpat * p3.is_complement
    return p3, sd, len(m)


def sector_block(p3: pd.DataFrame) -> dict:
    """Pooled replication + the three NICE cuts + the interaction."""
    out = {}

    # Pooled, to confirm this frame reproduces the published headline before
    # any splitting is trusted.
    out["pooled"] = safe_fe(p3, "dkl_z", ["logpat"], "owner_name", "year", "pooled")
    out["pooled_ctrl_nscored"] = safe_fe(
        p3, "dkl_z", ["logpat", "log_nscored"], "owner_name", "year", "pooled_ctrl")

    # Cut 1: every NICE class on its own, no grouping imposed.
    per_class = {}
    for cls, g in p3.groupby("nice_class"):
        per_class[cls] = safe_fe(g, "dkl_z", ["logpat"], "owner_name", "year", cls)
        per_class[cls]["group"] = nice_group(cls)
        per_class[cls]["goods_services"] = goods_services(cls)
    out["by_nice_class"] = per_class

    # Cut 2: Castaldi's goods (1-34) vs services (35-45).
    out["by_goods_services"] = {
        k: safe_fe(g, "dkl_z", ["logpat"], "owner_name", "year", k)
        for k, g in p3.groupby("goods_services")
    }

    # Cut 3: a-priori complement / neutral / substitute.
    out["by_ipr_group"] = {
        k: safe_fe(g, "dkl_z", ["logpat"], "owner_name", "year", k)
        for k, g in p3.groupby("nice_group")
    }

    # Cut 4: pooled interaction. Firm FE absorbs the firm-constant complement
    # dummy; logpat_x_comp is the complement-minus-noncomplement slope gap.
    out["interaction_complement"] = safe_fe(
        p3, "dkl_z", ["logpat", "logpat_x_comp"], "owner_name", "year", "interaction")

    # Descriptives, so the reader can see which cells carry the sample.
    out["cell_sizes"] = {
        "firm_years_total": int(len(p3)),
        "firms_total": int(p3.owner_name.nunique()),
        "by_nice_class": {
            cls: {"firm_years": int(len(g)), "firms": int(g.owner_name.nunique())}
            for cls, g in p3.groupby("nice_class")
        },
        "by_ipr_group": {
            k: {"firm_years": int(len(g)), "firms": int(g.owner_name.nunique())}
            for k, g in p3.groupby("nice_group")
        },
    }
    return out


def sic_block(p3: pd.DataFrame) -> dict:
    """Secondary cut: SEC SIC instead of NICE class, on crosswalked owners."""
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik").unique(subset=["owner_name"]).to_pandas()
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet").select(
        "cik", "sic").drop_nulls().unique(subset=["cik"]).to_pandas()
    link = cw.merge(sec, on="cik", how="inner")
    link["sic_group"] = link.sic.map(sic_group)
    link = link[link.sic_group != ""]

    q = p3.merge(link[["owner_name", "sic", "sic_group"]], on="owner_name", how="inner")
    out = {
        "n_firm_years": int(len(q)),
        "n_firms": int(q.owner_name.nunique()),
        "pooled": safe_fe(q, "dkl_z", ["logpat"], "owner_name", "year", "sic_pooled"),
        "by_sic_group": {
            k: safe_fe(g, "dkl_z", ["logpat"], "owner_name", "year", k)
            for k, g in q.groupby("sic_group")
        },
        "cell_sizes": {
            k: {"firm_years": int(len(g)), "firms": int(g.owner_name.nunique())}
            for k, g in q.groupby("sic_group")
        },
    }
    return out


def report(kind: str, block: dict) -> None:
    p = block["pooled"]
    if "coef" not in p.get("logpat", {}):
        print(f"[{kind}] pooled SKIPPED: {p.get('skipped')}", file=sys.stderr)
    else:
        print(f"[{kind}] pooled coef={p['logpat']['coef']:+.4f} "
              f"t={p['logpat']['t']:+.2f} n={p['n_obs']:,} firms={p['n_firms']:,}",
              file=sys.stderr, flush=True)
    for grp, r in sorted(block["by_ipr_group"].items()):
        if "logpat" in r:
            print(f"[{kind}]   {grp:11s} coef={r['logpat']['coef']:+.4f} "
                  f"t={r['logpat']['t']:+.2f} n={r['n_obs']:,} firms={r['n_firms']:,}",
                  file=sys.stderr, flush=True)
        else:
            print(f"[{kind}]   {grp:11s} skipped ({r.get('skipped')})", file=sys.stderr)
    ix = block["interaction_complement"]
    if "logpat_x_comp" in ix:
        print(f"[{kind}]   interaction logpat={ix['logpat']['coef']:+.4f} "
              f"x_complement={ix['logpat_x_comp']['coef']:+.4f} "
              f"(t={ix['logpat_x_comp']['t']:+.2f})", file=sys.stderr, flush=True)
    # The classes that most move the referee's question, ranked.
    pc = {k: v for k, v in block["by_nice_class"].items() if "logpat" in v}
    top = sorted(pc.items(), key=lambda kv: -kv[1]["logpat"]["coef"])[:5]
    bot = sorted(pc.items(), key=lambda kv: kv[1]["logpat"]["coef"])[:5]
    print(f"[{kind}]   most positive classes: "
          + ", ".join(f"{k}={v['logpat']['coef']:+.3f}(t={v['logpat']['t']:+.1f})"
                      for k, v in top), file=sys.stderr)
    print(f"[{kind}]   most negative classes: "
          + ", ".join(f"{k}={v['logpat']['coef']:+.3f}(t={v['logpat']['t']:+.1f})"
                      for k, v in bot), file=sys.stderr, flush=True)


def main() -> int:
    pat = load_patents()
    print(f"[patents] {len(pat):,} owner-years", file=sys.stderr, flush=True)

    results = {
        "spec_note": (
            "Firm + year FE, firm-clustered SE, firms with >=3 years; dkl "
            "standardized once on the pooled matched panel so coefficients are "
            "in the same sigma units as the pooled result. Sector is assigned "
            "at the firm level (dominant NICE class over the whole window), so "
            "the estimating equation is unchanged inside every subsample."),
        "min_firms": MIN_FIRMS,
        "min_obs": MIN_OBS,
        "complement_classes": sorted(COMPLEMENT_CLASSES),
        "substitute_classes": sorted(SUBSTITUTE_CLASSES),
        "scorings": {},
    }

    for kind in ("token", "t200"):
        oyc = owner_year_class_panel(kind)
        print(f"[panel:{kind}] {oyc.height:,} owner-year-class rows, "
              f"{oyc['owner_name'].n_unique():,} owners", file=sys.stderr, flush=True)
        p3, sd, n_matched = build_estimation_frame(oyc, pat)
        del oyc
        gc.collect()
        print(f"[frame:{kind}] matched owner-years={n_matched:,}  "
              f">=3yr rows={len(p3):,}  sigma_dkl={sd:.4f}", file=sys.stderr, flush=True)

        block = sector_block(p3)
        block["sigma_dkl"] = sd
        block["matched_firm_years"] = int(n_matched)
        report(kind, block)
        block["by_sic"] = sic_block(p3)
        s = block["by_sic"]["pooled"]
        if "logpat" in s:
            print(f"[{kind}]   SIC-matched pooled coef={s['logpat']['coef']:+.4f} "
                  f"t={s['logpat']['t']:+.2f} firms={s['n_firms']:,}",
                  file=sys.stderr, flush=True)
        results["scorings"][kind] = block
        del p3
        gc.collect()

    out = RES / "patent_complementarity_by_sector.json"
    out.write_text(json.dumps(results, indent=1, default=float))
    print(f"[done] -> {out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
