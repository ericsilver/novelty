"""Is the leading-mark gate penalty just the dot-com swing?

The penalty is largest for filings made into the late 1990s, reverses for those
made 2000-2004, and returns from 2008, with the whole swing inside the four
technology classes. That pattern invites a simple deflationary reading: leading
language in 1998 meant internet language, the internet firms of 1998 failed,
and the measure is an expensive way of rediscovering the dot-com bust.

The test is to remove the internet from the corpus and see what is left. A
filing is flagged WEB if its goods/services description mentions any of a fixed
list of internet-era terms, judged on the text at filing rather than on the
class it sits in, so a web business in apparel is caught and a semiconductor
maker in class 009 is not. The gate contrast is then re-estimated on the
non-web complement, overall and era by era.

What the test can and cannot settle. If the penalty survives among filings that
never mention the web, it is not a dot-com artifact. If it vanishes, the
deflationary reading is right. What it cannot do is separate "the web" from
"whatever was new at the time": internet vocabulary is much of what made a 1998
filing leading, so excluding it also removes a large part of the era's treatment
variation, and the surviving estimate is necessarily on a narrower and more
conventional population.

Output: paper/results/gate_ex_web.json
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
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
TECH = {"009", "035", "038", "042"}

# Internet-era vocabulary, matched case-insensitively on word boundaries where
# the term is a whole word. Deliberately broad: the test is stronger if the
# excluded set is generous, since anything surviving exclusion is cleaner.
WEB = [
    "internet", "web site", "website", "web-site", "web page", "webpage",
    "world wide web", "on-line", "on line", "online", "e-commerce",
    "electronic commerce", "e-mail", "email", "electronic mail", "cyber",
    "dot-com", "dot com", "www", "http", "domain name", "web based",
    "web-based", "portal", "browser", "hyperlink", "chat room", "intranet",
    "extranet", "downloadable", "streaming", "mobile app", "smartphone app",
    "cloud computing", "software as a service", "digital network",
    "computer network", "telecommunication network", "networked",
]
ERAS = [("1995-1999", 1995, 1999), ("2000-2004", 2000, 2004),
        ("2005-2007", 2005, 2007), ("2008-2014", 2008, 2014)]


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def contrast(d: pl.DataFrame, within_cell: bool = True) -> dict | None:
    if d.height < 3000:
        return None
    if within_cell:
        s = d.sort(["cell", "lead", "serial_number"]).with_columns(
            ((pl.col("lead").rank("ordinal").over("cell") - 1) * 5
             // pl.len().over("cell")).cast(pl.Int8).alias("q"))
    else:
        s = d.sort(["lead", "serial_number"]).with_columns(
            ((pl.col("lead").rank("ordinal") - 1) * 5 // pl.len())
            .cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"),
                            pl.len().alias("n")).sort("q")
    if g.height < 5:
        return None
    v = [float(r["p"]) for r in g.iter_rows(named=True)]
    p1, p5 = v[0], v[4]
    n1, n5 = int(g["n"][0]), int(g["n"][4])
    se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
    return {"n": int(d.height), "base": float(d["failed"].mean()),
            "lift": p5 - p1, "se": se, "t": (p5 - p1) / se if se else None}


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("gd")
    ).drop_nulls("gd").group_by("serial_number").agg(pl.col("gd").min())

    pat = "(?i)" + "|".join(t.replace(".", r"\.").replace(" ", r"\s+") for t in WEB)
    parts = []
    for c in CLASSES:
        sp, tp = PROC / f"{SRC}_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                          "registration_date", "goods_services"]).filter(
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
            & pl.col("goods_services").is_not_null())
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        j = tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.col("goods_services").str.contains(pat).alias("web"),
            pl.lit(c).alias("cls"))
        parts.append(j.drop("goods_services"))
        del tm, sc, j
        gc.collect()

    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        pl.col("topic_dkl").alias("lead"),
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry"))
    d = d.join(ev, on="serial_number", how="left").with_columns(
        (((pl.col("gd") - pl.col("rd")).dt.total_days() / 365.25)
         .is_between(GATE_LO, GATE_HI, closed="left")).fill_null(False)
        .cast(pl.Float64).alias("failed"),
        pl.concat_str([pl.col("cls"), pl.col("ry").cast(pl.Utf8)],
                      separator="-").alias("cell"))
    log(f"[frame] {d.height:,} scored registrations; "
        f"{100*d['web'].mean():.1f}% mention an internet-era term")

    gate = d.filter(pl.col("ry").is_between(REG_LO, REG_HI))
    out = {"scoring": SRC, "n": int(d.height),
           "web_share": float(d["web"].mean()), "terms": WEB,
           "overall": {}, "by_era": {}, "by_era_nontech": {}}

    log("\noverall, registrations 2002-2018 (within class x cohort)")
    for lab, sub in (("all", gate),
                     ("non-web", gate.filter(~pl.col("web"))),
                     ("web", gate.filter(pl.col("web")))):
        r = contrast(sub)
        out["overall"][lab] = r
        if r:
            log(f"  {lab:<9} {100*r['lift']:+6.2f}pp +/-{196*r['se']:4.2f}  "
                f"(base {100*r['base']:5.2f}%, n={r['n']:,})")

    log("\nby filing era, pooled (non-web only in the second column)")
    log(f"  {'era':<11} {'all':>18} {'non-web':>18} {'non-web, non-tech':>20}")
    for lab, lo, hi in ERAS:
        e = d.filter(pl.col("fy").is_between(lo, hi))
        ra = contrast(e, within_cell=False)
        rn = contrast(e.filter(~pl.col("web")), within_cell=False)
        rt = contrast(e.filter(~pl.col("web") & ~pl.col("cls").is_in(list(TECH))),
                      within_cell=False)
        out["by_era"][lab] = {"all": ra, "non_web": rn}
        out["by_era_nontech"][lab] = rt

        def f(r):
            return (f"{100*r['lift']:+6.2f}+/-{196*r['se']:4.2f}" if r else "        --      ")
        log(f"  {lab:<11} {f(ra):>18} {f(rn):>18} {f(rt):>20}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_ex_web.json").write_text(json.dumps(out, indent=1))
    log("\n[done] gate_ex_web.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
