"""When an application is abandoned, how often does the same firm try again?

The registration step is an inverse-U in atypicality: filings whose language sits
far from their category in either direction complete registration least often.
If abandonment were usually terminal, that shape would be a fact about which
products reach the market. If instead abandoned applications are commonly
reworked and refiled, the shape is partly an accounting artifact -- the same
commercial intent is counted as a failure and then again as a success -- and the
unconditional composite inherits it, which is what produces the M-shaped profile
the paper avoids by conditioning on grant.

An application counts as REFILED when the same normalized owner later files
another application, in the same Nice class, whose mark text matches. Two match
rules are reported:

  exact    identical normalized mark text (case, punctuation and spacing
           stripped). This is the conservative rule: the firm is trying the same
           brand again.
  fuzzy    exact, or the earlier mark text is a prefix of the later one or vice
           versa (a firm narrowing or extending a mark), requiring at least four
           characters. This catches reworking, at the cost of some false
           positives among very short marks.

The later filing must postdate the abandoned one and fall within REFILE_YEARS.
Refiling is then crossed with the vocabulary position of the ORIGINAL filing, so
the question is whether the tails of the registration inverse-U are refiled at
different rates than its middle -- and whether, once refiling is counted as
eventual success, the inverse-U flattens.

Output: paper/results/refile_after_abandon.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
FILE_LO, FILE_HI = 1995, 2015   # leave room for a later refiling to be observed
REFILE_YEARS = 6
NBINS = 20


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def norm_mark(col: str) -> pl.Expr:
    return (pl.col(col).str.to_uppercase()
            .str.replace_all(r"[^A-Z0-9 ]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars())


def norm_owner(col: str) -> pl.Expr:
    return (pl.col(col).str.to_uppercase()
            .str.replace_all(r"[^A-Z0-9 ]", "")
            .str.replace_all(
                r"\b(INC|LLC|LTD|CORP|CORPORATION|COMPANY|CO|LP|LLP|PLC|GMBH|SA|NV|AB|AG)\b", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars())


def profile(df: pl.DataFrame, score: str, outcome: str, nbins: int) -> dict | None:
    d = df.filter(pl.col(score).is_finite())
    if d.height < 5000:
        return None
    d = d.sort(score).with_columns(
        ((pl.col(score).rank("ordinal") - 1) * nbins // pl.len()).cast(pl.Int32).alias("b"))
    g = d.group_by("b").agg(pl.col(outcome).mean().alias("p"),
                            pl.len().alias("n")).sort("b")
    v = [float(r["p"]) for r in g.iter_rows(named=True)]
    ns = [int(r["n"]) for r in g.iter_rows(named=True)]
    mid = v[nbins // 2 - 1:nbins // 2 + 1]
    return {"n": int(d.height), "base": float(d[outcome].mean()),
            "bins": v, "bin_n": ns,
            "middle_minus_tails": sum(mid) / len(mid) - (v[0] + v[-1]) / 2}


def main() -> int:
    parts = []
    for c in CLASSES:
        sp, tp = PROC / f"{SRC}_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                          "registration_date", "owner_name",
                                          "mark_identification"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("mark_identification").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite()
            & pl.col("topic_kl_vs_future").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()

    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    d = d.with_columns(
        (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"),
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("A"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8).alias("registered"),
        norm_owner("owner_name").alias("own"),
        norm_mark("mark_identification").alias("mk"),
    ).filter(pl.col("fy").is_between(FILE_LO, FILE_HI)
             & (pl.col("own").str.len_chars() >= 3)
             & (pl.col("mk").str.len_chars() >= 4))
    log(f"[panel] {d.height:,} scored filings {FILE_LO}-{FILE_HI}, "
        f"{d['own'].n_unique():,} owners")

    aband = d.filter(~pl.col("registered")).select(
        "serial_number", "own", "mk", "cls", "fy", "L", "A")
    later = d.select(pl.col("own"), pl.col("mk").alias("mk2"), pl.col("cls"),
                     pl.col("fy").alias("fy2"),
                     pl.col("registered").alias("reg2"),
                     pl.col("serial_number").alias("sn2"))
    log(f"[abandoned] {aband.height:,} of {d.height:,} "
        f"({100*aband.height/d.height:.1f}%) never registered")

    # exact rule: same owner, same class, identical mark text, strictly later
    ex = aband.join(later, left_on=["own", "mk", "cls"],
                    right_on=["own", "mk2", "cls"], how="left")
    ex = ex.filter(pl.col("sn2").is_null()
                   | ((pl.col("fy2") > pl.col("fy"))
                      & (pl.col("fy2") <= pl.col("fy") + REFILE_YEARS)))
    ex = ex.group_by("serial_number").agg(
        pl.col("sn2").is_not_null().any().alias("refiled_exact"),
        (pl.col("sn2").is_not_null() & pl.col("reg2").fill_null(False))
        .any().alias("refiled_ok_exact"))

    # fuzzy rule: one mark text a prefix of the other
    fz = aband.join(later, left_on=["own", "cls"], right_on=["own", "cls"],
                    how="left")
    fz = fz.filter(
        pl.col("sn2").is_null()
        | ((pl.col("fy2") > pl.col("fy"))
           & (pl.col("fy2") <= pl.col("fy") + REFILE_YEARS)
           & (pl.col("mk2").str.starts_with(pl.col("mk"))
              | pl.col("mk").str.starts_with(pl.col("mk2")))))
    fz = fz.group_by("serial_number").agg(
        pl.col("sn2").is_not_null().any().alias("refiled_fuzzy"),
        (pl.col("sn2").is_not_null() & pl.col("reg2").fill_null(False))
        .any().alias("refiled_ok_fuzzy"))

    a = aband.join(ex, on="serial_number", how="left").join(
        fz, on="serial_number", how="left").with_columns(
        [pl.col(c).fill_null(False).cast(pl.Float64)
         for c in ("refiled_exact", "refiled_ok_exact",
                   "refiled_fuzzy", "refiled_ok_fuzzy")])
    del ex, fz
    gc.collect()

    out = {"scoring": SRC, "filing_years": [FILE_LO, FILE_HI],
           "refile_window_years": REFILE_YEARS,
           "n_scored": int(d.height), "n_abandoned": int(aband.height),
           "rates": {}, "profiles": {}, "eventual": {}}

    log("\nrefiling rates among abandoned applications")
    for c in ("refiled_exact", "refiled_ok_exact", "refiled_fuzzy", "refiled_ok_fuzzy"):
        r = float(a[c].mean())
        out["rates"][c] = r
        log(f"  {c:<20} {100*r:5.2f}%")

    # does refiling vary along the two axes?
    log("\nrefiling by vocabulary position (20 bins; middle minus tails)")
    for axis in ("A", "L"):
        for c in ("refiled_exact", "refiled_fuzzy"):
            pr = profile(a, axis, c, NBINS)
            out["profiles"][f"{axis}|{c}"] = pr
            if pr:
                log(f"  {axis} x {c:<16} base {100*pr['base']:5.2f}%  "
                    f"mid-tails {100*pr['middle_minus_tails']:+5.2f}pp  "
                    f"ends {100*pr['bins'][0]:5.2f}% / {100*pr['bins'][-1]:5.2f}%")

    # what the registration profile looks like once a successful refiling counts
    d2 = d.join(a.select("serial_number", "refiled_ok_exact", "refiled_ok_fuzzy"),
                on="serial_number", how="left").with_columns(
        pl.col("registered").cast(pl.Float64).alias("reg_strict"),
        (pl.col("registered").cast(pl.Float64)
         + pl.col("refiled_ok_exact").fill_null(0.0)).clip(0, 1).alias("reg_exact"),
        (pl.col("registered").cast(pl.Float64)
         + pl.col("refiled_ok_fuzzy").fill_null(0.0)).clip(0, 1).alias("reg_fuzzy"))
    log("\nregistration profile in atypicality, before and after crediting refilings")
    for c in ("reg_strict", "reg_exact", "reg_fuzzy"):
        pr = profile(d2, "A", c, NBINS)
        out["eventual"][c] = pr
        if pr:
            log(f"  {c:<11} base {100*pr['base']:5.2f}%  "
                f"inverse-U depth {100*pr['middle_minus_tails']:+5.2f}pp  "
                f"ends {100*pr['bins'][0]:5.2f}% / {100*pr['bins'][-1]:5.2f}%")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "refile_after_abandon.json").write_text(json.dumps(out, indent=1))
    log("\n[done] refile_after_abandon.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
