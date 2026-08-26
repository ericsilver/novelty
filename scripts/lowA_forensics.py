"""Forensics on the low-atypicality end: who is there, and is the text copied?

Eric's concern: the shapes at the low-atypicality end (the registration dip
in the bottom decile, the least-atypical fifth's worst five-year-proof
survival) could be created by a distinct population -- e.g. filings copying
other filings word-for-word, appropriately (USPTO ID Manual language) or
not (competitors' descriptions).

For every scored class-record 1995-2018, the normalized description text is
hashed; per class, each hash's cluster is characterized (how many filings
share the text, how many distinct owners, the earliest filing date, the
first filer). Groups: bottom within-class-year atypicality decile ("lowA")
split by registration outcome, and deciles 5-6 ("midA") as reference.

Memory-lean: each class is reduced immediately to (a) a slim stats frame
with no text and (b) up to 12 sampled description lines per group, before
the next class loads.

Outputs: paper/results/lowA_forensics.json
         paper/results/lowA_samples/{lowA_rejected,lowA_accepted,midA}.txt
"""
from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
DUMP = RES / "lowA_samples"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
FY_LO, FY_HI = 1995, 2018
GATE_LO, GATE_HI = 4.0, 8.5
SEED = 20260826
PER_CLASS_DUMP = 12
PER_GROUP_DUMP = 250

SLIM = ["serial_number", "cls", "fy", "reg", "rd", "nw", "dupN", "dup_owners",
        "prior_other_owner", "dA"]


def log(m): print(m, file=sys.stderr, flush=True)


def fmt_rows(rows):
    out = []
    for r in rows.iter_rows(named=True):
        t = " ".join((r["goods_services"] or "").split())[:400]
        out.append(f"[{r['cls']} {r['fy']} {'REG' if r['reg'] else 'ab '} "
                   f"dup{r['dupN']}/{r['dup_owners']}own"
                   f"{'*COPY' if r['prior_other_owner'] else ''} w{r['nw']}] {t}")
    return out


def main() -> int:
    DUMP.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    slims = []
    dumps = {"lowA_rejected": [], "lowA_accepted": [], "midA": []}
    for c in CLASSES:
        tp, sp = PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date",
                                          "owner_name", "goods_services"]).filter(
            pl.col("filing_date").fill_null("").str.len_chars() >= 8).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8).alias("reg"),
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
            pl.col("goods_services").fill_null("").str.to_lowercase()
              .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(r"\s+", " ")
              .str.strip_chars().alias("norm"),
            pl.col("owner_name").fill_null("").str.to_uppercase()
              .str.replace_all(r"[^A-Z0-9 ]", "").str.replace_all(r"\s+", " ")
              .str.strip_chars().alias("own"))
        tm = tm.filter(pl.col("fy").is_between(FY_LO, FY_HI)
                       & (pl.col("norm").str.len_chars() > 0)).with_columns(
            pl.col("norm").hash(seed=7).alias("h"),
            pl.col("norm").str.split(" ").list.len().alias("nw")).drop("norm")
        cl = tm.group_by("h").agg(pl.len().alias("dupN"),
                                  pl.col("own").n_unique().alias("dup_owners"),
                                  pl.col("filing_date").min().alias("first_fd"),
                                  pl.col("own").sort_by("filing_date").first().alias("first_own"))
        tm = tm.join(cl, on="h", how="left").with_columns(
            ((pl.col("dupN") > 1) & (pl.col("filing_date") > pl.col("first_fd"))
             & (pl.col("own") != pl.col("first_own"))).alias("prior_other_owner"))
        del cl
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite() & pl.col("topic_kl_vs_future").is_finite())
        j = tm.join(sc, on="serial_number", how="inner").with_columns(
            ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"),
            pl.lit(c).alias("cls"))
        del tm, sc
        j = j.with_columns(((pl.col("A").rank("ordinal").over("fy") - 1) * 10
                            // pl.len().over("fy")).cast(pl.Int8).alias("dA"))
        keep = j.filter((pl.col("dA") == 0) | pl.col("dA").is_in([4, 5]))
        del j
        gc.collect()
        slims.append(keep.select(SLIM))
        for name, cond in (("lowA_rejected", (pl.col("dA") == 0) & ~pl.col("reg")),
                           ("lowA_accepted", (pl.col("dA") == 0) & pl.col("reg")),
                           ("midA", pl.col("dA").is_in([4, 5]))):
            sub = keep.filter(cond)
            if sub.height:
                k = min(PER_CLASS_DUMP, sub.height)
                dumps[name].extend(fmt_rows(sub[rng.sample(range(sub.height), k)]))
        del keep
        gc.collect()
        log(f"[{c}] done")

    d = pl.concat(slims)
    del slims
    gc.collect()
    att = pl.read_parquet(PROC / "case_extras.parquet",
                          columns=["serial_number", "attorney_name", "intent_to_use_in"]).with_columns(
        (pl.col("attorney_name").fill_null("").str.len_chars() > 0).alias("rep"),
        (pl.col("intent_to_use_in") == "T").alias("itu")).select("serial_number", "rep", "itu")
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    d = d.join(att, on="serial_number", how="left").with_columns(
        pl.col("rep").fill_null(False), pl.col("itu").fill_null(False))
    del att
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).alias("gate_fail"))
    del ev
    gc.collect()
    log(f"[frame] {d.height:,} rows kept")

    def stats(sub, name):
        regs = sub.filter(pl.col("reg") & (pl.col("rd").dt.year() >= 2002))
        r = {"n": sub.height,
             "reg_rate": float(sub["reg"].mean()),
             "counsel_share": float(sub["rep"].mean()),
             "itu_share": float(sub["itu"].mean()),
             "median_words": int(sub["nw"].median()),
             "dup_any": float((sub["dupN"] > 1).mean()),
             "dup_prior_other_owner": float(sub["prior_other_owner"].mean()),
             "cluster_ge100": float((sub["dupN"] >= 100).mean()),
             "small_multiowner_2_9": float(((sub["dupN"].is_between(2, 9))
                                            & (sub["dup_owners"] >= 2)).mean()),
             "gate_fail_regs": float(regs["gate_fail"].mean()) if regs.height else None,
             "n_regs_gate": int(regs.height)}
        log(f"[{name}] " + json.dumps(r))
        return r

    out = {"groups": {}}
    lowA = d.filter(pl.col("dA") == 0)
    out["groups"]["lowA_rejected"] = stats(lowA.filter(~pl.col("reg")), "lowA_rejected")
    out["groups"]["lowA_accepted"] = stats(lowA.filter(pl.col("reg")), "lowA_accepted")
    out["groups"]["midA"] = stats(d.filter(pl.col("dA").is_in([4, 5])), "midA")
    lr = lowA.filter(pl.col("reg") & (pl.col("rd").dt.year() >= 2002))
    for lab, cond in (("copied", pl.col("prior_other_owner")),
                      ("manual_scale", pl.col("dupN") >= 100),
                      ("unique_text", pl.col("dupN") == 1)):
        sub = lr.filter(cond)
        out.setdefault("lowA_reg_gate_by_text", {})[lab] = {
            "n": int(sub.height),
            "gate_fail": float(sub["gate_fail"].mean()) if sub.height else None}
        log(f"[gate|{lab}] n={sub.height:,} fail={out['lowA_reg_gate_by_text'][lab]['gate_fail']}")
    # registration by text type within lowA (the margin Eric asked about)
    for lab, cond in (("copied", pl.col("prior_other_owner")),
                      ("manual_scale", pl.col("dupN") >= 100),
                      ("small_multiowner", (pl.col("dupN").is_between(2, 9)) & (pl.col("dup_owners") >= 2)),
                      ("unique_text", pl.col("dupN") == 1)):
        sub = lowA.filter(cond)
        out.setdefault("lowA_reg_by_text", {})[lab] = {
            "n": int(sub.height),
            "reg_rate": float(sub["reg"].mean()) if sub.height else None,
            "counsel_share": float(sub["rep"].mean()) if sub.height else None}
        log(f"[reg|{lab}] " + json.dumps(out["lowA_reg_by_text"][lab]))
    (RES / "lowA_forensics.json").write_text(json.dumps(out, indent=1))

    for name, lines in dumps.items():
        k = min(PER_GROUP_DUMP, len(lines))
        sample = rng.sample(lines, k)
        (DUMP / f"{name}.txt").write_text("\n".join(sorted(sample)), encoding="utf8")
        log(f"[dump] {name}: {k} texts")
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
