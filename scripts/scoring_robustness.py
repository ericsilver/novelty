"""Do the paper's major claims hold under every scoring variant?

The paper's estimates run on topic scoring with a flat per-filing reference
window. Three things could be carrying the results instead of the construct:
the representation (topics against terms), the weighting inside the window
(flat against exponential decay), and -- historically -- the anchoring
(per-filing against calendar-year). Anchoring is no longer varied anywhere in
the paper; every scorer compared here is anchored on the filing's own date. So
this varies the other two, in a full two-by-two:

    rolling         topics, flat window          (the paper's estimates)
    decay           topics, 2-year half-life
    termroll        terms,  flat window
    termrolldecay   terms,  2-year half-life

and re-estimates the claims the paper actually rests on:

  gate        top-minus-bottom lead quintile contrast in first-gate failure,
              cut within class x cohort -- the central result
  gate_era    the same contrast for filings 1995-1999, 2000-2004 and 2008-2014,
              split on technology classes -- the episodic finding
  atypicality the same contrast on atypicality rather than lead, which the
              paper reports as doing none of the work at the firm level
  corr        agreement between each variant's lead and the paper's, so a
              claim that survives can be told apart from a variant that is
              simply the same numbers

A claim that holds in all four is a property of the construct. A claim that
holds only under flat topics is a property of that scorer, and the paper
should say so.

Output: paper/results/scoring_robustness.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

VARIANTS = [
    ("rolling", "topics, flat window"),
    ("decay", "topics, 2-year half-life"),
    ("termroll", "terms, flat window"),
    ("termrolldecay", "terms, 2-year half-life"),
]
CLASSES = [f"{i:03d}" for i in range(1, 46)]
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
TECH = {"009", "035", "038", "042"}
ERAS = [("1995-1999", 1995, 1999), ("2000-2004", 2000, 2004),
        ("2008-2014", 2008, 2014)]


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def gate_events() -> pl.DataFrame:
    return pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())


def load(src: str, ev: pl.DataFrame) -> pl.DataFrame | None:
    parts = []
    for c in CLASSES:
        sp = PROC / f"{src}_surprise_class{c}.parquet"
        tp = PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(
            tp, columns=["serial_number", "filing_date", "registration_date"]).filter(
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(
            sp, columns=["serial_number", "topic_kl_vs_past",
                         "topic_kl_vs_future", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()
    if not parts:
        return None
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("atyp"),
    ).drop_nulls(["rd", "fd"])
    return d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"),
        pl.col("rd").dt.year().alias("ry"),
        pl.col("fd").dt.year().alias("fy"),
        pl.col("cls").is_in(TECH).alias("tech"),
    ).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"),
        pl.concat_str([pl.col("cls"), pl.col("ry").cast(pl.Utf8)],
                      separator="-").alias("cell"))


def contrast(df: pl.DataFrame, score: str, within_cell: bool = True,
             min_n: int = 3000) -> dict | None:
    if df.height < min_n:
        return None
    if within_cell:
        s = df.sort(["cell", score, "serial_number"]).with_columns(
            ((pl.col(score).rank("ordinal").over("cell") - 1) * 5
             // pl.len().over("cell")).cast(pl.Int8).alias("q"))
    else:
        s = df.sort([score, "serial_number"]).with_columns(
            ((pl.col(score).rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"),
                            pl.len().alias("n")).sort("q")
    if g.height < 5:
        return None
    v = [float(r["p"]) for r in g.iter_rows(named=True)]
    p1, p5 = v[0], v[4]
    n1, n5 = int(g["n"][0]), int(g["n"][4])
    se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
    return {"n": int(df.height), "base": float(df["failed"].mean()),
            "lift": p5 - p1, "se": se, "t": (p5 - p1) / se if se else None}


def main() -> int:
    ev = gate_events()
    out = {"variants": {v: d for v, d in VARIANTS}, "results": {}}
    ref = None

    for src, desc in VARIANTS:
        d = load(src, ev)
        if d is None:
            log(f"[{src}] no scored files -- skipped")
            continue
        gate = d.filter(pl.col("ry").is_between(REG_LO, REG_HI))
        row = {
            "description": desc,
            "n_scored": int(d.height),
            "gate": contrast(gate, "topic_dkl"),
            "gate_atypicality": contrast(gate, "atyp"),
            "eras": {},
        }
        for label, lo, hi in ERAS:
            sub = d.filter(pl.col("fy").is_between(lo, hi))
            row["eras"][label] = {
                "all": contrast(sub, "topic_dkl", within_cell=False),
                "tech": contrast(sub.filter(pl.col("tech")), "topic_dkl",
                                 within_cell=False),
                "other": contrast(sub.filter(~pl.col("tech")), "topic_dkl",
                                  within_cell=False),
            }

        if src == "rolling":
            ref = d.select("serial_number", pl.col("topic_dkl").alias("ref_dkl"),
                           pl.col("atyp").alias("ref_atyp"))
        elif ref is not None:
            j = d.select("serial_number", "topic_dkl", "atyp").join(
                ref, on="serial_number", how="inner")
            def _corr(a: str, b: str, method: str) -> float:
                return float(j.select(pl.corr(a, b, method=method)).item())
            row["corr_with_rolling"] = {
                "n": int(j.height),
                "lead": _corr("topic_dkl", "ref_dkl", "pearson"),
                "lead_spearman": _corr("topic_dkl", "ref_dkl", "spearman"),
                "atypicality": _corr("atyp", "ref_atyp", "pearson"),
            }
            del j
        out["results"][src] = row
        del d, gate
        gc.collect()

        g = row["gate"]
        log(f"\n[{src}] {desc}")
        log(f"  scored {row['n_scored']:,}")
        if g:
            log(f"  gate lead contrast    {100*g['lift']:+6.2f}pp "
                f"+/-{196*g['se']:4.2f}  (base {100*g['base']:5.2f}%, t={g['t']:+5.1f})")
        a = row["gate_atypicality"]
        if a:
            log(f"  gate atypicality      {100*a['lift']:+6.2f}pp "
                f"+/-{196*a['se']:4.2f}")
        for label in row["eras"]:
            e = row["eras"][label]
            def f(r):
                return f"{100*r['lift']:+6.2f}" if r else "   --  "
            log(f"  era {label}  all {f(e['all'])}  tech {f(e['tech'])}  "
                f"other {f(e['other'])}")
        if "corr_with_rolling" in row:
            c = row["corr_with_rolling"]
            log(f"  corr with paper's lead  r={c['lead']:+.3f}  "
                f"rho={c['lead_spearman']:+.3f}  (atypicality r={c['atypicality']:+.3f})")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "scoring_robustness.json").write_text(json.dumps(out, indent=1))
    log("\n[done] scoring_robustness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
