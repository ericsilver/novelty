"""Debut vocabulary and the ladder of SEC events: a priced round, SEC reporting, an IPO.

One row per owner: the owner's first registered filing (the debut), scored on
the production measure, with its class and filing year. Debuts 2009-2018, so
that Form D coverage (electronic and mandatory from 2009-03-16) is complete
for every owner. Rungs, each read from the SEC record and dated:

  funded      a priced Reg D offering by the owner after the debut (Form D)
  reporting   the owner appears in SEC financial-statement data sets or the
              EDGAR crosswalk (in_sec / in_fsds / crosswalk match)
  ipo         an 8-A registration of a class of securities (in_8a) or an
              IPO date

Each rung is reported unconditionally (share of debuts) and conditionally on
the rung below (share of funded owners that report; share of reporting owners
that IPO). Lead and atypicality enter as within-class-and-year quintiles; the
contrast is the top fifth minus the bottom fifth at each rung.

Curated cohorts: debuts whose description matches the internet, AI or
blockchain patterns of theme_cohorts.py, compared on every rung.

Patents: on the owner-year patent panel (data_publish/firm_year_patents_and_dkl.csv),
for owners with a Form D round, the year of the first round against the year of
the first patent grant: which comes first, and by how long.

Output: paper/results/sec_event_ladder.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
PUB = REPO / "data_publish"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
DEBUT_LO, DEBUT_HI = 2009, 2018
COHORTS = {
    "internet": (r"\binternet\b|\bonline\b|\bon-line\b|\bweb ?sites?\b|\bweb pages?\b|"
                 r"\bwebsites?\b|\bworld wide web\b|\be-?commerce\b|"
                 r"\belectronic commerce\b|\bweb portals?\b|\bweb browsers?\b"),
    "ai": (r"\bartificial intelligence\b|\bmachine learning\b|"
           r"\bdeep learning\b|\bneural networks?\b|\bnatural language "
           r"(processing|understanding)\b|\bcomputer vision\b|"
           r"\bpredictive analytics\b|\bchat ?bots?\b|"
           r"\bgenerative (ai|artificial)\b|\blarge language model"),
    "blockchain": (r"\bblockchains?\b|\bcrypto ?currenc|\bcrypto ?assets?\b|"
                   r"\bcrypto ?tokens?\b|\bbitcoin\b|\bnon-?fungible token|\bnfts?\b|"
                   r"\bdistributed ledger|\bdigital currenc|\bvirtual currenc|"
                   r"\bsmart contracts?\b"),
}


def log(m): print(m, file=sys.stderr, flush=True)


def quintile_contrast(df: pl.DataFrame, var: str, y: str, length_held: bool = False) -> dict | None:
    d = df.filter(pl.col(var).is_finite())
    if d.height < 2000:
        return None
    cells = ["cls", "fy"]
    if length_held:
        d = d.with_columns(((pl.col("n_words").rank("ordinal").over(["cls", "fy"]) - 1) * 5
                            // pl.len().over(["cls", "fy"])).cast(pl.Int8).alias("lenq"))
        cells = ["cls", "fy", "lenq"]
    s = d.sort(cells + [var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over(cells) - 1) * 5 // pl.len().over(cells)).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col(y).mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    if len(p) < 5 or min(n) < 50:
        return None
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"n": int(d.height), "base": float(d[y].mean()), "quintiles": p, "lift": p[4] - p[0], "se": se,
            "t": (p[4] - p[0]) / se if se > 0 else None}


def main() -> int:
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet").select(
        "owner_name", "first_formd_date", "n_formd", "in_sec", "in_fsds", "in_8a", "ipo_date")
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet", columns=["owner_name", "cik", "first_year"]).unique("owner_name")
    parts = []
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date", "owner_name",
                                          "goods_services"]).filter(
            pl.col("owner_name").is_not_null() & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        d = tm.join(sc, on="serial_number", how="inner").with_columns(pl.lit(c).alias("cls"))
        g = d["goods_services"].str.to_lowercase()
        d = d.with_columns(*[g.str.contains(p).alias(f"coh_{k}") for k, p in COHORTS.items()],
                           g.str.count_matches(r"[a-z]+").alias("n_words")).drop("goods_services")
        parts.append(d)
        del tm, sc, d
    d = pl.concat(parts); del parts; gc.collect()
    d = d.with_columns(pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
                       pl.col("topic_dkl").alias("L"),
                       ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    debut = d.sort(["owner_name", "fd", "serial_number"]).unique(subset="owner_name", keep="first").with_columns(
        pl.col("fd").dt.year().alias("fy")).filter(pl.col("fy").is_between(DEBUT_LO, DEBUT_HI))
    log(f"[debut] {debut.height:,} registered debut owners {DEBUT_LO}-{DEBUT_HI}")

    x = debut.join(fm, on="owner_name", how="left").join(cw, on="owner_name", how="left")
    x = x.with_columns(
        (pl.col("first_formd_date").is_not_null() & (pl.col("first_formd_date") >= pl.col("fd"))).alias("funded"),
        ((pl.col("in_sec").fill_null(0) == 1) | (pl.col("in_fsds").fill_null(0) == 1) | pl.col("cik").is_not_null()).alias("reporting"),
        ((pl.col("in_8a").fill_null(0) == 1) | pl.col("ipo_date").is_not_null()).alias("ipo"),
    ).with_columns(*[pl.col(c).cast(pl.Float64) for c in ("funded", "reporting", "ipo")])
    out = {"debut_years": [DEBUT_LO, DEBUT_HI], "n_debuts": int(x.height), "rungs": {}, "conditional": {},
           "contrasts": {}, "cohorts": {}, "patents": {}}
    for r in ("funded", "reporting", "ipo"):
        out["rungs"][r] = {"n": int(x[r].sum()), "share": float(x[r].mean())}
    out["conditional"]["reporting_given_funded"] = float(x.filter(pl.col("funded") == 1)["reporting"].mean())
    out["conditional"]["reporting_given_not_funded"] = float(x.filter(pl.col("funded") == 0)["reporting"].mean())
    out["conditional"]["ipo_given_funded"] = float(x.filter(pl.col("funded") == 1)["ipo"].mean())
    out["conditional"]["ipo_given_reporting"] = float(x.filter(pl.col("reporting") == 1)["ipo"].mean())
    log(f"  rungs: " + "  ".join(f"{k} {100*v['share']:.2f}% (n={v['n']:,})" for k, v in out["rungs"].items()))
    log(f"  conditional: {out['conditional']}")
    for r in ("funded", "reporting", "ipo"):
        out["contrasts"][r] = {"L": quintile_contrast(x, "L", r), "A": quintile_contrast(x, "A", r),
                               "L_length_held": quintile_contrast(x, "L", r, True),
                               "A_length_held": quintile_contrast(x, "A", r, True)}
        for v in ("L", "A", "L_length_held", "A_length_held"):
            c = out["contrasts"][r][v]
            if c:
                log(f"  {r:9s} by {v:14s}: Q1..Q5 {[round(100*q,3) for q in c['quintiles']]}  lift {100*c['lift']:+.3f}pp (t {c['t']:+.1f})")
    out["length_by_rung"] = {r: {"mean_words_yes": float(x.filter(pl.col(r) == 1)["n_words"].mean()),
                                 "mean_words_no": float(x.filter(pl.col(r) == 0)["n_words"].mean())} for r in ("funded", "reporting", "ipo")}
    log(f"  length by rung: {out['length_by_rung']}")
    # Conditional rungs: among funded, does vocabulary predict reporting? among reporting, IPO?
    out["contrasts"]["reporting_given_funded"] = {v: quintile_contrast(x.filter(pl.col("funded") == 1), v, "reporting") for v in ("L", "A")}
    out["contrasts"]["ipo_given_funded"] = {v: quintile_contrast(x.filter(pl.col("funded") == 1), v, "ipo") for v in ("L", "A")}
    # Cohorts.
    for k in COHORTS:
        sub = x.filter(pl.col(f"coh_{k}"))
        out["cohorts"][k] = {"n": int(sub.height), "share_of_debuts": float(x[f"coh_{k}"].mean()),
                             "mean_L": float(sub["L"].mean()), "mean_A": float(sub["A"].mean()),
                             **{r: float(sub[r].mean()) for r in ("funded", "reporting", "ipo")},
                             "by_year": {str(y): {"n": int(s.height), "funded": float(s["funded"].mean())}
                                         for y, s in sub.group_by("fy") for y in [y[0] if isinstance(y, tuple) else y]}}
        c = out["cohorts"][k]
        log(f"  cohort {k:10s} n={c['n']:7,} funded {100*c['funded']:.2f}% reporting {100*c['reporting']:.2f}% ipo {100*c['ipo']:.2f}%")
    out["cohorts"]["all"] = {"n": int(x.height), **{r: float(x[r].mean()) for r in ("funded", "reporting", "ipo")}}

    # Patents: first round vs first patent, owners on the patent panel.
    pat = pl.read_csv(PUB / "firm_year_patents_and_dkl.csv", columns=["uspto_owner_name", "year", "n_patents"]).filter(
        pl.col("n_patents") > 0).group_by("uspto_owner_name").agg(pl.col("year").min().alias("first_patent_year"))
    px = fm.filter(pl.col("first_formd_date").is_not_null()).select(
        "owner_name", pl.col("first_formd_date").dt.year().alias("first_round_year")
    ).join(pat.rename({"uspto_owner_name": "owner_name"}), on="owner_name", how="inner")
    if px.height:
        lag = (px["first_round_year"] - px["first_patent_year"]).to_numpy()
        out["patents"] = {"n_owners_with_both": int(px.height),
                          "patent_first_share": float((lag > 0).mean()), "round_first_share": float((lag < 0).mean()),
                          "same_year_share": float((lag == 0).mean()), "median_round_minus_patent_years": float(np.median(lag)),
                          "note": "Patent years are grant years on the PatentsView panel; Form D years are first priced round."}
        log(f"  patents: n={px.height:,} patent-first {100*out['patents']['patent_first_share']:.1f}% round-first {100*out['patents']['round_first_share']:.1f}% median lag {out['patents']['median_round_minus_patent_years']:+.1f}y")
    (RES / "sec_event_ladder.json").write_text(json.dumps(out, indent=1, default=float))
    log("[done] sec_event_ladder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
