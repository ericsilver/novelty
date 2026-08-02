"""Does vocabulary position predict VENTURE FUNDING, and funding-conditional listing?

Motivation
----------
The paper's financial-outcome result rests on P(owner ever in the SEC-listed
universe), a 0.189% event.  At that base rate the debut panel has roughly 2,900
successes in 1.56M rows, and every subsample split is one sampling accident away
from flipping.  Venture funding is one to two orders of magnitude more common and
sits between registration and listing in the firm's life course, so it is the
natural place to look for the same signal with usable power.

Funding is observed through SEC Form D, the notice of exempt offering that
essentially every priced US venture round must file within 15 days of first sale.
Form D became mandatory in electronic form on 2009-03-16, so the structured data
sets (data/raw/formd/*.zip, 2009q1-2026q2) are a near-census of Reg D offerings
from 2009 onward and see nothing at all before that.

Constructs (as in the rest of the paper)
----------------------------------------
The parquet columns name their own reference window:
    topic_kl_vs_past   = KL(P || Q_past)     distance from the class's PRIOR 5 years
    topic_kl_vs_future = KL(P || Q_future)   distance from the class's NEXT 5 years
    topic_dkl = topic_kl_vs_past - topic_kl_vs_future,
which is the paper's signed Delta-KL (Sec. 2: "Positive Delta-KL marks a document
whose vocabulary is unusual relative to its recent past and close to its near
future").  The paper calls these two levels "prospective" and "retrospective"
surprise after Murdock/Barron; the column names say which window each is measured
against, and this script labels its outputs the same way.

ATYPICALITY  the unsigned KL level -- how far the filing's text sits from its
             class-year norm.  Reported against both windows (vs_past = the
             paper's headline axis, vs_future) and as their mean.  High = the
             filing does not sound like its class.
FORWARD LEAN signed topic_dkl.  POSITIVE = unusual against the recent past and
             close to the near future, i.e. written in language the class
             subsequently moved toward.

Both are taken from data/processed/topic_surprise_class{cls}_T200.parquet and are
ranked/standardized WITHIN the class x filing-year cell, so a quintile is always a
comparison against contemporaries in the same class.

Three questions
---------------
Q1  DESCRIPTIVE.  Do owners who are matched to a Form D operating-company issuer
    sit at different atypicality / forward lean from unmatched owners?  Reported
    twice: the raw pooled gap, and the gap net of class x filing-year fixed
    effects (the within-cell gap, which is what every regression below uses).

Q2  FUNDING.  Debut cohort (one row per owner, at the owner's first-ever USPTO
    filing), outcome = owner ever appears as a Form D operating-company issuer.
    Estimator is byte-for-byte the one in scripts/debut_edgar_substantiate.py:
    linear probability model, class x filing-year FE absorbed by within-cell
    demeaning, HC1 heteroskedasticity-robust SEs, log description length as the
    control.  Both the quintile-dummy and continuous-z forms are run, then the
    paper's |dKL| / signed-dKL decomposition, so the funding result can be read
    against the listing result term by term.

Q3  LISTING | FUNDING.  Restrict to the funded owners and ask whether atypicality
    still predicts reaching the listed universe.  Three markers of "listed", in
    increasing strictness:
      in_sec        owner_name is in data/processed/uspto_sec_crosswalk.parquet
                    (the paper's own marker, so directly comparable)
      fsds          the Form D CIK appears in data/processed/sec_firm_year.parquet
                    (SEC Financial Statement Data Sets = the XBRL filer universe);
                    this needs NO name matching beyond the Form D link
      form_8a       the CIK has an 8-A12B/8-A12G on EDGAR = Exchange Act Sec. 12
                    registration = the actual moment of exchange listing
    The prototype's S-1 / 424B / 10-K fallbacks are deliberately NOT used: S-1 and
    10-K are filed by companies that never list, so they inflate the outcome.
    The 8-A lookup is restricted to funded CIKs that already appear in the FSDS
    or crosswalk universes; a company that listed on a US exchange but files no
    XBRL financials and is absent from both is close to a null set, and that
    restriction is what makes the EDGAR pass affordable.

Caveats that are printed with the results, not buried
-----------------------------------------------------
*  Form D is electronic-only from 2009q1.  The debut cohort is therefore
   restricted to 2009-2018 filings.  A 1997 debut cannot be observed raising a
   Reg D round in this data at all, which is why the paper's full 1995-2018
   window is not used here.
*  Roughly 57% of Form D issuer-offerings are pooled funds and SPVs rather than
   operating companies; those are dropped on ISPOOLEDINVESTMENTFUNDTYPE plus the
   INVESTMENTFUNDTYPE string and the primary-issuer flag.
*  Owner-name -> issuer-name matching is normalized-exact (scripts/sec_link.py),
   and lands ~8-9%.  It favours large, formally suffixed entities.  EVERY estimate
   here is conditional on being matchable; see the printed note on direction.
*  Q3 conditions on funding, which is realized AFTER the trademark filing, so it
   is a post-treatment sample split, not a control.  Its coefficient is a
   within-survivor contrast, not a causal effect of atypicality on listing.
*  Holding-company renames break the CIK link at IPO (the prototype found
   "Coinbase, Inc." raising under one CIK and "Coinbase Global, Inc." listing
   under another).  That biases Q3 toward zero and is not fixed here.

Inputs
------
data/raw/owner_debut.parquet                          owner -> first-ever filing date
data/raw/formd/form_d_issuers.parquet                 parsed Form D (built by
                                                      scripts/funding_lag_prototype.py)
data/processed/tm_class{cls}.parquet                  serial -> owner, dates, text
data/processed/topic_surprise_class{cls}_T200.parquet topic_kl_vs_{past,future}, topic_dkl
data/processed/uspto_sec_crosswalk.parquet            owner_name -> cik (listed)
data/processed/sec_firm_year.parquet                  cik -> XBRL financial reporter

Output
------
paper/results/atypicality_and_funding.json  (+ a stderr summary table)

Usage
-----
    PYTHONPATH=src .venv-analysis/Scripts/python.exe scripts/atypicality_and_funding.py
    ... --classes 009 042 035        # sample the largest classes only
    ... --edgar-8a                   # add the EDGAR 8-A12B/8-A12G listing marker
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RAWD = REPO / "data" / "raw"
RES = REPO / "paper" / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sec_link import normalize  # noqa: E402

# Form D e-filing became mandatory 2009-03-16.  2018 is the paper's debut-panel
# upper bound (registration outcomes are resolved by then); keeping it makes the
# cohort comparable to the listing analysis and leaves >= 8 years of funding
# exposure for the last cohort year.
FILE_LO, FILE_HI = 2009, 2018

# A class x year cell smaller than this cannot support within-cell quintiles.
# 100 is the value used in scripts/debut_edgar_substantiate.py; the funded
# subsample of Q3 is far thinner, so it gets its own floor.
MIN_CELL = 100
MIN_CELL_FUNDED = 25
# Below these the estimate is not reported at all (house convention).
MIN_OBS = 300

UA = "Eric Silver epsilver@gmail.com (novelty research)"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
IPO_PRIMARY = ("8-A12B", "8-A12G")   # S-1 / 424B / 10-K fallbacks deliberately dropped
SLEEP_API = 0.12


# --------------------------------------------------------------- estimator ---
def lpm_fe(df: pl.DataFrame, yvar: str, xcols: list[str], label: str,
           cell: str = "cell", unit: str = "pp") -> dict:
    """LPM with cell fixed effects absorbed by within-cell demeaning, HC1 SEs.

    Identical in form to lpm_fe() in scripts/debut_edgar_substantiate.py, with
    the outcome column made an argument.  One row per owner, so plain
    heteroskedasticity-robust SEs are appropriate; there is no repeated-owner
    clustering to do.
    """
    d = df.select([yvar, cell] + xcols).drop_nulls()
    n_pre = d.height
    if n_pre < MIN_OBS:
        return {"label": label, "skipped": "cell too small", "n": int(n_pre)}
    y = d[yvar].cast(pl.Float64).to_numpy()
    X = d.select(xcols).cast(pl.Float64).to_numpy()
    pdf = pd.DataFrame(X, columns=xcols)
    pdf["y"] = y
    pdf["cell"] = d[cell].to_numpy()
    dem = pdf.groupby("cell")[xcols + ["y"]].transform("mean")
    Xd = (pdf[xcols] - dem[xcols]).to_numpy()
    yd = (pdf["y"] - dem["y"]).to_numpy()
    XtX = Xd.T @ Xd
    try:
        b = np.linalg.solve(XtX, Xd.T @ yd)
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        return {"label": label, "skipped": f"singular: {exc}", "n": int(n_pre)}
    e = yd - Xd @ b
    meat = (Xd * (e ** 2)[:, None]).T @ Xd
    n, k = Xd.shape
    V = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    base = float(d[yvar].cast(pl.Float64).mean())
    out = {
        "label": label, "n": int(n), "n_cells": int(pdf["cell"].nunique()),
        "base_rate": base, "coef_unit": unit,
        "coef": {c: {"b": float(bi), "se": float(si), "t": float(bi / si) if si else None}
                 for c, bi, si in zip(xcols, b, se)},
    }
    # binary outcomes read naturally in percentage points; a z-scored outcome
    # (Q1) is already in sigma units and must NOT be multiplied by 100
    mul = 100.0 if unit == "pp" else 1.0
    disp = " ".join(f"{c}={v['b']*mul:+.4f}{unit}(t={v['t']:+.1f})"
                    for c, v in out["coef"].items())
    print(f"  {label:<38s} n={n:>9,} base={base*100:6.3f}%  {disp}",
          file=sys.stderr, flush=True)
    return out


def welch_t(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sample Welch t for the raw (no-FE) gap."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"n_a": na, "n_b": nb, "gap": None, "t": None}
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    se = float(np.sqrt(va / na + vb / nb))
    return {"n_a": na, "n_b": nb, "mean_a": ma, "mean_b": mb,
            "gap": ma - mb, "se": se, "t": (ma - mb) / se if se else None}


# ------------------------------------------------------------ panel build ---
def class_list(sel: list[str] | None, scoring: str) -> list[str]:
    have = sorted(p.stem.replace("tm_class", "")
                  for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit()
                  and (PROC / f"topic_surprise_class{p.stem[-3:]}{scoring}.parquet").exists())
    return [c for c in have if c in set(sel)] if sel else have


def build_debut_panel(classes: list[str], scoring: str) -> pl.DataFrame:
    """One row per DEBUT owner: the owner's first-ever USPTO filing, scored.

    Same construction as scripts/debut_edgar_substantiate.py -- join every class's
    filings to the owner's first-ever filing date, keep the filings that ARE the
    debut, attach topic scores, then de-duplicate to one serial and one owner (a
    debut can tie across classes on the same day).
    """
    debut = pl.read_parquet(RAWD / "owner_debut.parquet")
    parts = []
    for cls in classes:
        tm = (
            pl.scan_parquet(PROC / f"tm_class{cls}.parquet")
            .select("serial_number", "owner_name", "filing_date",
                    "registration_date", "goods_services")
            .filter(pl.col("owner_name").is_not_null()
                    & (pl.col("filing_date").str.len_chars() == 8))
            .with_columns(
                pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
                (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
                .alias("registered"),
                pl.col("goods_services").fill_null("").str.len_chars().alias("gs_len"),
            )
            .filter(pl.col("fy").is_between(FILE_LO, FILE_HI))
            .join(debut.lazy(), on="owner_name", how="inner")
            .filter(pl.col("filing_date") == pl.col("debut"))
            .join(
                pl.scan_parquet(PROC / f"topic_surprise_class{cls}{scoring}.parquet")
                .select("serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl")
                .filter(pl.col("topic_kl_vs_past").is_finite()
                        & pl.col("topic_kl_vs_future").is_finite()
                        & pl.col("topic_dkl").is_finite()),
                on="serial_number", how="inner",
            )
            .with_columns(pl.lit(cls).alias("cls"),
                          pl.col("filing_date").str.to_date("%Y%m%d", strict=False)
                          .alias("filed"))
            .select("serial_number", "owner_name", "cls", "fy", "filed", "registered",
                    "gs_len", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl")
            .collect()
        )
        if tm.height:
            parts.append(tm)
        print(f"[panel] class {cls}: {tm.height:,} scored debut filings",
              file=sys.stderr, flush=True)
        del tm
        gc.collect()
    df = (pl.concat(parts)
          .unique(subset="serial_number", keep="first")
          .sort("serial_number")
          .unique(subset="owner_name", keep="first"))
    del parts
    gc.collect()
    return df


def formd_operating_issuers() -> tuple[pl.DataFrame, dict]:
    """First non-amendment Reg D offering per unambiguous operating-company name.

    Drops pooled investment funds and SPVs, non-primary issuers, and any
    normalized name that resolves to more than one CIK (a name collision cannot
    be assigned to one trademark owner).
    """
    fd = pl.read_parquet(RAWD / "formd" / "form_d_issuers.parquet")
    ops = fd.filter(~pl.col("is_pooled") & (pl.col("fund_type") == "")
                    & pl.col("is_primary"))
    amb = (ops.group_by("norm").agg(pl.col("cik").n_unique().alias("n_cik"))
           .filter(pl.col("n_cik") > 1)["norm"])
    first_round = (
        ops.filter(~pl.col("is_amend") & pl.col("round_date").is_not_null()
                   & (pl.col("norm") != ""))
        .sort("round_date")
        .group_by("norm", maintain_order=True)
        .agg(pl.col("cik").first(),
             pl.col("entity_name").first(),
             pl.col("round_date").first().alias("first_round"),
             pl.col("total_sold").first().alias("first_round_sold"),
             pl.col("industry").first().alias("fd_industry"),
             pl.len().alias("n_form_d"))
        .filter(~pl.col("norm").is_in(amb))
    )
    meta = {
        "formd_issuer_offerings_total": int(fd.height),
        "formd_pooled_share": float(fd["is_pooled"].mean()),
        "formd_operating_offerings": int(ops.height),
        "formd_operating_distinct_names": int(ops["norm"].n_unique()),
        "formd_names_dropped_as_cik_collisions": int(len(amb)),
        "formd_unambiguous_operating_issuers": int(first_round.height),
        "formd_first_quarter": "2009q1",
        "formd_last_quarter": "2026q2",
    }
    print(f"[formd] {fd.height:,} issuer-offerings; {fd['is_pooled'].mean()*100:.1f}% pooled; "
          f"{ops.height:,} operating; {first_round.height:,} unambiguous issuer names "
          f"({len(amb):,} dropped as CIK collisions)", file=sys.stderr, flush=True)
    del fd, ops
    gc.collect()
    return first_round, meta


def add_cell_stats(df: pl.DataFrame, min_cell: int) -> pl.DataFrame:
    """class x filing-year cell, within-cell quintiles and z-scores, log length."""
    d = df.with_columns(
        pl.format("{}_{}", pl.col("cls"), pl.col("fy")).alias("cell"),
        (pl.col("gs_len").cast(pl.Float64) + 1).log().alias("log_len"),
    ).filter(pl.len().over("cell") >= min_cell)
    # rank("ordinal") resolves ties by frame row order, which is not stable across
    # runs; ~27% of scored filings are tied on these scores (boilerplate text gives
    # identical topic distributions), so the tie-break has to be pinned explicitly
    # or quintile boundaries wander between runs. Sort on the score then a unique
    # key before each ranking.
    for col, tag in (("topic_kl_vs_past", "p"), ("topic_kl_vs_future", "r"), ("topic_dkl", "d")):
        d = d.sort(["cell", col, "owner_name"])
        d = d.with_columns(
            ((pl.col(col).rank("ordinal").over("cell") - 1) * 5
             // pl.len().over("cell")).cast(pl.Int8).alias(f"q_{tag}"),
            ((pl.col(col) - pl.col(col).mean().over("cell"))
             / pl.col(col).std().over("cell")).alias(f"z_{tag}"),
        )
    for tag in ("p", "r", "d"):
        for qq in range(1, 5):
            d = d.with_columns(
                (pl.col(f"q_{tag}") == qq).cast(pl.Float64).alias(f"{tag}q{qq+1}"))
    return d.with_columns(
        pl.col("z_d").abs().alias("abs_zd"),
        ((pl.col("z_p") + pl.col("z_r")) / 2).alias("z_atyp"),
    )


# ---------------------------------------------------- EDGAR 8-A (optional) ---
def edgar_8a(ciks: list[int], cache_path: Path, seed_path: Path | None) -> dict[str, str]:
    """First-ever 8-A12B/8-A12G date per CIK.  Seeded from, but never written
    back to, the prototype's cache."""
    import requests
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    seed: dict = {}
    if seed_path and seed_path.exists():
        seed = json.loads(seed_path.read_text())
    for k, rec in seed.items():
        if k in cache:
            continue
        hit = [rec[f] for f in IPO_PRIMARY if rec.get(f)]
        if hit or rec.get("_name"):     # a completed lookup, listed or not
            cache[k] = min(hit) if hit else ""
    todo = [c for c in ciks if str(c) not in cache]
    print(f"[edgar] {len(todo):,} CIKs to look up ({len(ciks)-len(todo):,} cached/seeded)",
          file=sys.stderr, flush=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    for i, cik in enumerate(todo, 1):
        best = ""
        try:
            r = s.get(SUBMISSIONS.format(cik=cik), timeout=60)
            if r.status_code == 200:
                j = r.json()
                blocks = [j["filings"]["recent"]]
                for extra in j["filings"].get("files", []):
                    time.sleep(SLEEP_API)
                    r2 = s.get("https://data.sec.gov/submissions/" + extra["name"], timeout=60)
                    if r2.status_code == 200:
                        blocks.append(r2.json())
                for b in blocks:
                    for f, dt in zip(b.get("form", []), b.get("filingDate", [])):
                        if f in IPO_PRIMARY and (not best or dt < best):
                            best = dt
        except Exception as exc:  # network hiccup: record as unknown, do not crash
            print(f"[edgar] cik {cik}: {type(exc).__name__}", file=sys.stderr)
        cache[str(cik)] = best
        if i % 200 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"[edgar] {i:,}/{len(todo):,}", file=sys.stderr, flush=True)
        time.sleep(SLEEP_API)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return cache


# ------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="NICE classes to include (default: all 45)")
    ap.add_argument("--scoring", default="_T200", choices=["", "_T200", "_T500"],
                    help="topic-model suffix: '' = T50 (the paper's headline), "
                         "_T200 (default here), _T500")
    ap.add_argument("--edgar-8a", action="store_true",
                    help="add the EDGAR 8-A12B/8-A12G exchange-listing marker for Q3")
    ap.add_argument("--max-ciks", type=int, default=6000,
                    help="cap on EDGAR submissions-API lookups")
    args = ap.parse_args()

    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    classes = class_list(args.classes, args.scoring)
    print(f"[start] {len(classes)} NICE classes, debut cohort {FILE_LO}-{FILE_HI}",
          file=sys.stderr, flush=True)

    df = build_debut_panel(classes, args.scoring)
    print(f"[panel] {df.height:,} scored debut owners", file=sys.stderr, flush=True)

    first_round, fd_meta = formd_operating_issuers()

    # ---- link owners to Form D operating-company issuers -------------------
    df = df.with_columns(
        pl.col("owner_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))
    df = df.join(first_round, on="norm", how="left").with_columns(
        pl.col("first_round").is_not_null().alias("funded"))
    # variant: the first observed round is at or after the debut filing, so the
    # mark demonstrably precedes the money rather than merely coexisting with it
    df = df.with_columns(
        (pl.col("funded") & (pl.col("first_round") >= pl.col("filed")))
        .fill_null(False).alias("funded_after"))

    n_funded = int(df["funded"].sum())
    n_after = int(df["funded_after"].sum())
    print(f"[link] {n_funded:,}/{df.height:,} debut owners matched to a Form D "
          f"operating-company issuer ({n_funded/df.height*100:.2f}%); "
          f"{n_after:,} with the first round at/after the debut filing",
          file=sys.stderr, flush=True)

    # ---- how much of the match rate is name formality? ---------------------
    # The single biggest threat to every number below is that a normalized-exact
    # match can only fire for owners whose USPTO name and SEC name agree, which
    # means formally suffixed corporate entities.  Quantify it rather than assert
    # it: split the cohort on whether the owner name carries a corporate suffix
    # and report the funding rate in each half.
    df = df.with_columns(
        pl.col("owner_name").str.to_uppercase()
        .str.contains(r"\b(?:INC|LLC|L\.L\.C|CORP|CORPORATION|INCORPORATED|LTD|LIMITED|"
                      r"COMPANY|CO|L\.P|LP|PLC|GMBH|N\.V|S\.A)\b")
        .fill_null(False).alias("formal_name"))
    fmt = (df.group_by("formal_name")
           .agg(pl.len().alias("n"), pl.col("funded").mean().alias("p_funded"))
           .sort("formal_name"))
    match_diag = {str(r["formal_name"]): {"n": int(r["n"]), "p_funded": float(r["p_funded"])}
                  for r in fmt.iter_rows(named=True)}
    print(f"[link] match rate by owner-name formality: " +
          "  ".join(f"formal={k}: {v['p_funded']*100:.2f}% of {v['n']:,}"
                    for k, v in match_diag.items()), file=sys.stderr, flush=True)

    # ---- listed-universe markers ------------------------------------------
    cw = (pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")
          .select("owner_name").unique().with_columns(pl.lit(True).alias("in_sec")))
    fsds = (pl.read_parquet(PROC / "sec_firm_year.parquet")
            .select("cik").unique().with_columns(pl.lit(True).alias("in_fsds")))
    df = (df.join(cw, on="owner_name", how="left")
            .join(fsds, on="cik", how="left")
            .with_columns(pl.col("in_sec").fill_null(False),
                          (pl.col("in_fsds").fill_null(False) & pl.col("funded"))
                          .alias("in_fsds")))

    results: dict = {
        "run_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "debut_cohort_years": [FILE_LO, FILE_HI],
        "n_classes": len(classes),
        "classes": classes,
        "scoring": f"topic_surprise_*{args.scoring}",
        "n_debut_owners_scored": int(df.height),
        "n_funded": n_funded,
        "p_funded_raw": n_funded / df.height,
        "n_funded_after_debut": n_after,
        "p_funded_after_debut": n_after / df.height,
        "match_rate_by_owner_name_formality": match_diag,
        "formd": fd_meta,
        "caveats": {
            "formd_electronic_from": "2009-03-16 (mandatory); cohort restricted to "
                                     f"{FILE_LO}-{FILE_HI} debuts for that reason",
            "pooled_funds_dropped": "ISPOOLEDINVESTMENTFUNDTYPE + INVESTMENTFUNDTYPE "
                                    "+ primary-issuer flag",
            "name_match": "normalized-exact (scripts/sec_link.py); conditional on "
                          "matchability, which favours large, formally suffixed "
                          "entities and biases the funded set toward incorporated, "
                          "professionally lawyered firms",
            "q3_post_treatment": "conditioning on funding is a post-treatment split; "
                                 "the Q3 coefficient is a within-survivor contrast",
            "holdco_rename": "holdco renames break the CIK link at IPO (e.g. Coinbase "
                             "Inc -> Coinbase Global under a different CIK); biases "
                             "Q3 toward zero, not corrected",
        },
        "Q1_descriptive": {}, "Q2_funding": {}, "Q3_listing_given_funding": {},
    }

    # ---------------------------------------------------------------- Q1 ---
    print("\n=== Q1  do funded owners sit at different vocabulary positions? ===",
          file=sys.stderr, flush=True)
    full = add_cell_stats(df, MIN_CELL)
    print(f"[Q1] {full.height:,} owners in cells of >= {MIN_CELL} "
          f"({full['cell'].n_unique():,} class-year cells); "
          f"{int(full['funded'].sum()):,} funded", file=sys.stderr, flush=True)

    q1: dict = {"n": int(full.height), "n_funded": int(full["funded"].sum()),
                "p_funded": float(full["funded"].mean()),
                "min_cell": MIN_CELL, "raw": {}, "within_class_year": {}}
    fmask = full["funded"].to_numpy()
    for col, lab in (("topic_kl_vs_past", "atypicality_vs_past"),
                     ("topic_kl_vs_future", "atypicality_vs_future"),
                     ("topic_dkl", "forward_lean")):
        a = full.filter(pl.col("funded"))[col].to_numpy()
        b = full.filter(~pl.col("funded"))[col].to_numpy()
        w = welch_t(a, b)
        q1["raw"][lab] = w
        print(f"  raw  {lab:<28s} funded={w['mean_a']:+.4f} unfunded={w['mean_b']:+.4f} "
              f"gap={w['gap']:+.4f} nats (t={w['t']:+.1f})", file=sys.stderr, flush=True)
    # within class x year: regress the cell-z of each construct on the funded dummy
    fullz = full.with_columns(pl.col("funded").cast(pl.Float64).alias("funded_d"))
    for zc, lab in (("z_p", "atypicality_vs_past"), ("z_r", "atypicality_vs_future"),
                    ("z_d", "forward_lean"), ("z_atyp", "atypicality_mean_of_levels")):
        r = lpm_fe(fullz, zc, ["funded_d"], f"Q1_within_{lab}", unit="sigma")
        if "coef" in r:
            r["gap_in_within_cell_sigma"] = r["coef"]["funded_d"]["b"]
        q1["within_class_year"][lab] = r
    results["Q1_descriptive"] = q1

    # ---------------------------------------------------------------- Q2 ---
    print("\n=== Q2  does atypicality predict RAISING venture funding? ===",
          file=sys.stderr, flush=True)
    fu = full.with_columns(pl.col("funded").cast(pl.Float64).alias("y"))
    pq = ["pq2", "pq3", "pq4", "pq5"]
    rq = ["rq2", "rq3", "rq4", "rq5"]
    dq = ["dq2", "dq3", "dq4", "dq5"]
    q2 = {"base_rate_funded": float(fu["y"].mean()), "n": int(fu.height), "specs": []}
    S = q2["specs"]
    S.append(lpm_fe(fu, "y", pq, "A1_atypicality_quintiles_FE"))
    S.append(lpm_fe(fu, "y", pq + ["log_len"], "A2_atypicality_quintiles_len"))
    S.append(lpm_fe(fu, "y", ["z_p", "log_len"], "A3_atypicality_z_len"))
    S.append(lpm_fe(fu, "y", rq + ["log_len"], "A4_atypicality_vs_future_quintiles_len"))
    S.append(lpm_fe(fu, "y", ["z_atyp", "log_len"], "A5_atypicality_meanlevel_z_len"))
    S.append(lpm_fe(fu, "y", dq + ["log_len"], "B1_forwardlean_quintiles_len"))
    S.append(lpm_fe(fu, "y", ["z_d", "log_len"], "B2_forwardlean_z_len"))
    S.append(lpm_fe(fu, "y", ["abs_zd", "z_d", "log_len"], "C1_absdKL_signeddKL_len"))
    S.append(lpm_fe(fu, "y", ["z_p", "z_d", "log_len"], "C2_atypicality_and_lean_len"))
    S.append(lpm_fe(fu, "y", ["z_p", "z_r", "z_d", "log_len"], "C3_levels_and_lean_len"))
    S.append(lpm_fe(fu, "y", dq + ["log_len", "z_p", "z_r"], "C4_leanQ_ctrl_levels"))
    # outcome variant: first round strictly at/after the debut filing
    fa = full.with_columns(pl.col("funded_after").cast(pl.Float64).alias("y"))
    q2["funded_after_debut"] = {
        "n": int(fa.height), "base_rate": float(fa["y"].mean()),
        "specs": [lpm_fe(fa, "y", pq + ["log_len"], "A2a_atypicality_quintiles_len"),
                  lpm_fe(fa, "y", ["z_p", "z_d", "log_len"], "C2a_atypicality_and_lean_len")],
    }
    # registered-only, to match the listing analysis's denominator exactly
    regd = fu.filter(pl.col("registered"))
    q2["registered_only"] = {
        "n": int(regd.height), "base_rate_funded": float(regd["y"].mean()),
        "specs": [lpm_fe(regd, "y", pq + ["log_len"], "A2r_atypicality_quintiles_len"),
                  lpm_fe(regd, "y", ["abs_zd", "z_d", "log_len"], "C1r_absdKL_signeddKL_len"),
                  lpm_fe(regd, "y", ["z_p", "z_d", "log_len"], "C2r_atypicality_and_lean_len")],
    }
    results["Q2_funding"] = q2

    # ---------------------------------------------------------------- Q3 ---
    print("\n=== Q3  conditional on funding, does atypicality predict LISTING? ===",
          file=sys.stderr, flush=True)
    fnd = add_cell_stats(df.filter(pl.col("funded")), MIN_CELL_FUNDED).with_columns(
        (pl.col("first_round_sold").cast(pl.Float64).clip(0, None) + 1).log()
        .alias("log_raised"),
        (pl.col("n_form_d").cast(pl.Float64) + 1).log().alias("log_n_rounds"),
    )
    print(f"[Q3] {fnd.height:,} funded owners in cells of >= {MIN_CELL_FUNDED} "
          f"({fnd['cell'].n_unique():,} cells)", file=sys.stderr, flush=True)

    markers = {"in_sec": "owner in USPTO-SEC crosswalk (paper's listed marker)",
               "in_fsds": "Form D CIK in SEC Financial Statement Data Sets"}

    if args.edgar_8a:
        cand = (fnd.filter(pl.col("in_sec") | pl.col("in_fsds"))["cik"]
                .drop_nulls().unique().to_list())[: args.max_ciks]
        print(f"[Q3] EDGAR 8-A pass over {len(cand):,} candidate CIKs "
              f"(funded owners already in FSDS or the crosswalk)",
              file=sys.stderr, flush=True)
        cache = edgar_8a(cand, RAWD / "formd" / "edgar_8a_funded.json",
                         RAWD / "formd" / "edgar_first_forms.json")
        hits = pl.DataFrame(
            {"cik": [int(k) for k, v in cache.items() if v],
             "form_8a_date": [v for v in cache.values() if v]},
            schema={"cik": pl.Int64, "form_8a_date": pl.Utf8})
        fnd = fnd.join(hits, on="cik", how="left").with_columns(
            pl.col("form_8a_date").is_not_null().alias("in_8a"))
        markers["in_8a"] = ("first-ever 8-A12B/8-A12G on EDGAR = Exchange Act Sec.12 "
                            "registration; screened to funded CIKs already in FSDS or "
                            "the crosswalk; S-1/424B/10-K fallbacks NOT used")

    # Baseline: the SAME listing regression on the whole 2009-2018 debut cohort,
    # unconditional on funding.  Without it there is no way to tell whether the
    # funded-conditional coefficient is large because funding selects a
    # population in which vocabulary matters more, or merely because the base
    # rate is 14x higher.  This also re-runs the paper's listing result on the
    # Form-D-observable cohort, which is a shorter window than the published
    # 1995-2018 one.
    unc = full.with_columns(pl.col("in_sec").cast(pl.Float64).alias("y"))
    print(f"  -- BASELINE unconditional listing, same cohort: "
          f"{int(unc['y'].sum()):,} of {unc.height:,} (base {unc['y'].mean()*100:.3f}%)",
          file=sys.stderr, flush=True)
    q3_base = {
        "n": int(unc.height), "n_listed": int(unc["y"].sum()),
        "base_rate": float(unc["y"].mean()),
        "specs": [
            lpm_fe(unc, "y", pq + ["log_len"], "unconditional_A2_atypicality_quintiles_len"),
            lpm_fe(unc, "y", ["z_p", "log_len"], "unconditional_A3_atypicality_z_len"),
            lpm_fe(unc, "y", ["abs_zd", "z_d", "log_len"], "unconditional_C1_absdKL_signeddKL_len"),
            lpm_fe(unc, "y", ["z_p", "z_d", "log_len"], "unconditional_C2_atypicality_and_lean_len"),
        ],
    }

    q3: dict = {"n_funded_in_cells": int(fnd.height),
                "min_cell": MIN_CELL_FUNDED, "markers": markers,
                "unconditional_baseline_same_cohort": q3_base, "by_marker": {}}
    for mk in markers:
        if mk not in fnd.columns:
            continue
        y = fnd.with_columns(pl.col(mk).cast(pl.Float64).alias("y"))
        base = float(y["y"].mean())
        print(f"  -- marker {mk}: {int(y['y'].sum()):,} listed of {y.height:,} "
              f"(base {base*100:.3f}%)", file=sys.stderr, flush=True)
        q3["by_marker"][mk] = {
            "n": int(y.height), "n_listed": int(y["y"].sum()), "base_rate": base,
            "specs": [
                lpm_fe(y, "y", pq + ["log_len"], f"{mk}_A2_atypicality_quintiles_len"),
                lpm_fe(y, "y", ["z_p", "log_len"], f"{mk}_A3_atypicality_z_len"),
                lpm_fe(y, "y", ["abs_zd", "z_d", "log_len"], f"{mk}_C1_absdKL_signeddKL_len"),
                lpm_fe(y, "y", ["z_p", "z_d", "log_len"], f"{mk}_C2_atypicality_and_lean_len"),
                # Does atypicality merely proxy for having raised a lot of money
                # from a lot of investors?  Controlling for the size of the first
                # round and the number of Form D notices answers that directly.
                lpm_fe(y, "y", ["z_p", "z_d", "log_len", "log_raised", "log_n_rounds"],
                       f"{mk}_D1_ctrl_round_size_and_count"),
            ],
            "no_fe_quintile_rates": {
                f"q{int(r['q_p'])+1}": {"n": int(r["n"]), "rate": float(r["rate"])}
                for r in y.group_by("q_p").agg(pl.len().alias("n"),
                                               pl.col("y").mean().alias("rate"))
                          .sort("q_p").iter_rows(named=True)
            },
        }
    # Robustness: restrict to owners whose first observed round is at or after
    # the debut filing, so the mark demonstrably precedes the money.  This drops
    # firms that were already financed when they first filed, for whom the
    # trademark is not a pre-funding signal at all.
    fnd_a = add_cell_stats(df.filter(pl.col("funded_after")), MIN_CELL_FUNDED)
    q3["mark_precedes_money"] = {}
    for mk in ("in_sec", "in_fsds"):
        y = fnd_a.with_columns(pl.col(mk).cast(pl.Float64).alias("y"))
        print(f"  -- [mark-first] marker {mk}: {int(y['y'].sum()):,} of {y.height:,} "
              f"(base {y['y'].mean()*100:.3f}%)", file=sys.stderr, flush=True)
        q3["mark_precedes_money"][mk] = {
            "n": int(y.height), "n_listed": int(y["y"].sum()),
            "base_rate": float(y["y"].mean()),
            "specs": [lpm_fe(y, "y", ["z_p", "log_len"], f"mf_{mk}_A3_atypicality_z_len"),
                      lpm_fe(y, "y", ["z_p", "z_d", "log_len"],
                             f"mf_{mk}_C2_atypicality_and_lean_len")],
        }

    results["Q3_listing_given_funding"] = q3

    # ------------------------------------------------------------ report ---
    results["elapsed_s"] = round(time.time() - t0, 1)
    tag = args.scoring or "_T50"
    out = RES / (f"atypicality_and_funding{tag}.json" if args.scoring != "_T200"
                 else "atypicality_and_funding.json")
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\n[done] -> {out}  ({results['elapsed_s']:.0f}s)", file=sys.stderr, flush=True)
    print("Every estimate is conditional on a normalized-exact owner-name match to a "
          "Form D issuer.\nUnmatchable owners (informal names, DBAs, individuals) are "
          "coded as unfunded, so the\nfunding rate is a LOWER bound and the estimated "
          "slopes carry whatever correlation\natypicality has with being formally "
          "named.  See results['caveats'].", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
