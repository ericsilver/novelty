"""When a firm refiles an abandoned mark, does it rewrite the description -- and in which direction?

If leading language is penalised, a firm that abandons an application and tries
the same mark again has a reason to refile in more familiar words. Three
questions follow, each answerable from the record.

1. Among abandoned applications that the same owner later refiles for the same
   mark in the same class and carries to registration, how often is the
   goods/services text changed, and how?  Measured as the change in lead,
   atypicality and length from the abandoned filing to the registered one,
   against pairs whose text is identical (which isolates the part of any
   change that is only the reference windows moving with the date).

2. For the refiled registrations, which text predicts the gate: the original
   or the rewrite?  The registered filing is scored twice -- on its own text
   and on the abandoned predecessor's -- and the first-gate contrast computed
   on each.

3. What do refiled registrations stamp on the main estimate?  The
   within-class-and-year gate contrast with and without them.

Pairs use the exact rule of refile_after_abandon.py: identical normalized mark
text, same normalized owner, same Nice class, refiled within six years. The
earliest registered refiling is kept per abandoned application.

Output: paper/results/refile_text_change.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
FILE_LO, FILE_HI = 1995, 2015
REFILE_YEARS = 6
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def norm_mark(col: str) -> pl.Expr:
    return (pl.col(col).str.to_uppercase().str.replace_all(r"[^A-Z0-9 ]", "")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def norm_owner(col: str) -> pl.Expr:
    return (pl.col(col).str.to_uppercase().str.replace_all(r"[^A-Z0-9 ]", "")
            .str.replace_all(
                r"\b(INC|LLC|LTD|CORP|CORPORATION|COMPANY|CO|LP|LLP|PLC|GMBH|SA|NV|AB|AG)\b", "")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def norm_goods(col: str) -> pl.Expr:
    return (pl.col(col).str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(r"\s+", " ").str.strip_chars())


def contrast(df: pl.DataFrame, var: str, cells: list[str]) -> dict:
    s = df.sort(cells + [var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over(cells) - 1) * 5 // pl.len().over(cells))
        .cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"n": int(df.height), "quintiles": p, "lift": p[4] - p[0], "se": se,
            "t": (p[4] - p[0]) / se}


def mean_se(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "mean": float(x.mean()), "se": float(x.std(ddof=1) / len(x) ** 0.5),
            "median": float(np.median(x)), "share_negative": float((x < 0).mean())}


def main() -> int:
    parts = []
    for c in CLASSES:
        sp, tp = PROC / f"{SRC}_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date",
                                          "owner_name", "mark_identification"]).filter(
            pl.col("owner_name").is_not_null() & pl.col("mark_identification").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite() & pl.col("topic_kl_vs_future").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    d = d.with_columns(
        (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"),
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("A"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        norm_owner("owner_name").alias("own"),
        norm_mark("mark_identification").alias("mk"),
    ).with_columns(pl.col("rd").is_not_null().alias("registered"),
                   pl.col("rd").dt.year().alias("ry")).filter(
        (pl.col("own").str.len_chars() >= 3) & (pl.col("mk").str.len_chars() >= 4))
    log(f"[panel] {d.height:,} scored filings")

    aband = d.filter(~pl.col("registered") & pl.col("fy").is_between(FILE_LO, FILE_HI)).select(
        "serial_number", "own", "mk", "cls", "fy", "L", "A")
    later = d.filter(pl.col("registered")).select(
        "own", "mk", "cls", pl.col("fy").alias("fy2"), pl.col("serial_number").alias("sn2"),
        pl.col("L").alias("L2"), pl.col("A").alias("A2"), pl.col("ry").alias("ry2"),
        pl.col("rd").alias("rd2"))
    pairs = aband.join(later, on=["own", "mk", "cls"], how="inner").filter(
        (pl.col("fy2") > pl.col("fy")) & (pl.col("fy2") <= pl.col("fy") + REFILE_YEARS)
    ).sort(["serial_number", "fy2", "sn2"]).unique(subset="serial_number", keep="first")
    log(f"[pairs] {pairs.height:,} abandoned applications with a registered refiling "
        f"of the same mark within {REFILE_YEARS} years")

    # Fetch goods/services for both members of each pair, class by class.
    need = set(pairs["serial_number"].to_list()) | set(pairs["sn2"].to_list())
    gparts = []
    for c in CLASSES:
        tp = PROC / f"tm_class{c}.parquet"
        if not tp.exists():
            continue
        g = pl.read_parquet(tp, columns=["serial_number", "goods_services"]).filter(
            pl.col("serial_number").is_in(list(need)))
        gparts.append(g)
    goods = pl.concat(gparts).unique(subset="serial_number").with_columns(
        norm_goods("goods_services").alias("g"))
    goods = goods.with_columns(pl.col("g").str.split(" ").list.len().alias("len"),
                               pl.col("g").str.split(" ").list.unique().alias("toks"))
    del gparts
    gc.collect()
    p = pairs.join(goods.select("serial_number", pl.col("g").alias("g1"), pl.col("len").alias("len1"),
                                pl.col("toks").alias("t1")), on="serial_number", how="inner")
    p = p.join(goods.select(pl.col("serial_number").alias("sn2"), pl.col("g").alias("g2"),
                            pl.col("len").alias("len2"), pl.col("toks").alias("t2")),
               on="sn2", how="inner")
    p = p.with_columns(
        (pl.col("g1") == pl.col("g2")).alias("same_text"),
        (pl.col("t1").list.set_intersection(pl.col("t2")).list.len()
         / pl.col("t1").list.set_union(pl.col("t2")).list.len()).alias("jaccard"),
        (pl.col("L2") - pl.col("L")).alias("dL"),
        (pl.col("A2") - pl.col("A")).alias("dA"),
        (pl.col("len2").cast(pl.Float64).log() - pl.col("len1").cast(pl.Float64).log()).alias("dloglen"),
    )
    log(f"[text] {p.height:,} pairs with both texts; identical text in "
        f"{100*p['same_text'].mean():.1f}%")

    out = {"scoring": SRC, "n_pairs": int(p.height),
           "share_same_text": float(p["same_text"].mean()),
           "jaccard_changed": mean_se(p.filter(~pl.col("same_text"))["jaccard"].to_numpy()),
           "change": {}, "gate": {}, "stamp": {}}
    for lab, sub in (("changed", p.filter(~pl.col("same_text"))), ("identical", p.filter(pl.col("same_text")))):
        out["change"][lab] = {k: mean_se(sub[k].to_numpy()) for k in ("dL", "dA", "dloglen")}
        out["change"][lab]["n"] = int(sub.height)
        out["change"][lab]["original_mean_L"] = float(sub["L"].mean())
        out["change"][lab]["original_mean_A"] = float(sub["A"].mean())
        out["change"][lab]["refiled_mean_L"] = float(sub["L2"].mean())
        out["change"][lab]["refiled_mean_A"] = float(sub["A2"].mean())
        c = out["change"][lab]
        log(f"  {lab:9s} n={sub.height:,}  dL {c['dL']['mean']:+.4f} (SE {c['dL']['se']:.4f})  "
            f"dA {c['dA']['mean']:+.4f} (SE {c['dA']['se']:.4f})  dloglen {c['dloglen']['mean']:+.3f}  "
            f"share A fell {c['dA']['share_negative']:.2f}")
    # How the original's position relates to the direction of rewriting.
    # Quintile cuts are fixed on the pooled pairs so the changed and identical
    # groups are compared at the same original positions; the identical-text
    # group is the placebo for regression to the mean in the scores.
    for axis, dv in (("L", "dL"), ("A", "dA")):
        cuts = np.quantile(p[axis].to_numpy(), [0.2, 0.4, 0.6, 0.8])
        pq = p.with_columns(pl.Series("q", np.searchsorted(cuts, p[axis].to_numpy())))
        for lab, flag in (("changed", False), ("identical", True)):
            sub = pq.filter(pl.col("same_text") == flag)
            g = sub.group_by("q").agg(pl.col(dv).mean().alias("m"), pl.col(dv).std().alias("s"),
                                      pl.len().alias("n")).sort("q")
            out["change"][f"{dv}_by_original_{axis}_quintile_{lab}"] = {
                "mean": [float(v) for v in g["m"]],
                "se": [float(a / b ** 0.5) for a, b in zip(g["s"], g["n"])],
                "n": [int(v) for v in g["n"]]}
            log(f"  {dv} by original-{axis} quintile, {lab:9s}: "
                f"{[round(v,3) for v in out['change'][f'{dv}_by_original_{axis}_quintile_{lab}']['mean']]}")

    # Gate outcome of the refiled registration, scored on its own text and on the original's.
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    g = p.filter(pl.col("ry2").is_between(REG_LO, REG_HI)).join(
        ev.rename({"serial_number": "sn2"}), on="sn2", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd2")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
                   .fill_null(False).cast(pl.Float64).alias("failed"),
                   pl.col("sn2").alias("serial_number_b"))
    g = g.rename({"serial_number": "serial_a"}).rename({"serial_number_b": "serial_number"})
    for lab, sub in (("all_refiled", g), ("changed_text", g.filter(~pl.col("same_text")))):
        if sub.height < 2000:
            continue
        out["gate"][lab] = {"n": int(sub.height), "base": float(sub["failed"].mean()),
                            "on_refiled_text_L": contrast(sub, "L2", ["ry2"]),
                            "on_original_text_L": contrast(sub, "L", ["ry2"]),
                            "on_refiled_text_A": contrast(sub, "A2", ["ry2"]),
                            "on_original_text_A": contrast(sub, "A", ["ry2"])}
        r = out["gate"][lab]
        log(f"  gate {lab}: n={sub.height:,} base {100*r['base']:.1f}%  "
            f"L(refiled) {100*r['on_refiled_text_L']['lift']:+.2f} (t {r['on_refiled_text_L']['t']:+.1f})  "
            f"L(original) {100*r['on_original_text_L']['lift']:+.2f} (t {r['on_original_text_L']['t']:+.1f})  "
            f"A(refiled) {100*r['on_refiled_text_A']['lift']:+.2f}  A(original) {100*r['on_original_text_A']['lift']:+.2f}")

    # What refiled registrations stamp on the main contrast.
    full = d.filter(pl.col("registered") & pl.col("ry").is_between(REG_LO, REG_HI)).select(
        "serial_number", "cls", "ry", "L", "rd").join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
                   .fill_null(False).cast(pl.Float64).alias("failed"))
    refiled_ids = set(p["sn2"].to_list())
    full = full.with_columns(pl.col("serial_number").is_in(list(refiled_ids)).alias("is_refile"))
    out["stamp"] = {"n_gate_sample": int(full.height),
                    "n_refiled_in_sample": int(full["is_refile"].sum()),
                    "share_refiled": float(full["is_refile"].mean()),
                    "contrast_all": contrast(full, "L", ["cls", "ry"]),
                    "contrast_excluding_refiled": contrast(full.filter(~pl.col("is_refile")), "L", ["cls", "ry"]),
                    "fail_rate_refiled": float(full.filter(pl.col("is_refile"))["failed"].mean()),
                    "fail_rate_other": float(full.filter(~pl.col("is_refile"))["failed"].mean())}
    s = out["stamp"]
    log(f"  stamp: {s['n_refiled_in_sample']:,} of {s['n_gate_sample']:,} gate registrations are refilings "
        f"({100*s['share_refiled']:.2f}%); contrast all {100*s['contrast_all']['lift']:+.2f} vs "
        f"excluding {100*s['contrast_excluding_refiled']['lift']:+.2f}; fail refiled {100*s['fail_rate_refiled']:.1f}% "
        f"vs other {100*s['fail_rate_other']:.1f}%")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "refile_text_change.json").write_text(json.dumps(out, indent=1))
    log("[done] refile_text_change.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
