"""Can the pre-1990 corpus be scored and evaluated? Three feasibility checks.

1. VOLUME. Filings per decade, corpus-wide and for the six largest classes,
   and the share of filings in each decade whose 5-year and 10-year windows
   (both sides, own class) hold at least 500 filings.

2. VOCABULARY. The theme model's vocabulary was fitted on a sample across all
   years. Share of tokens in-vocabulary, by decade, on a sample of filings in
   classes 009 and 025.

3. OUTCOMES. The Section 8 use affidavit exists since the Lanham Act took
   effect (1947). Share of registrations per decade with (a) any dated event
   in case_events, (b) a C8../C71T cancellation event, and the apparent gate
   failure rate at age 4.0-8.5 by registration decade.

Output: paper/results/historical_feasibility.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
BIG = ["009", "025", "005", "016", "035", "030"]
MIN_REF = 500


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    out = {"volume": {}, "window_coverage": {}, "vocab": {}, "outcomes": {}}
    # 1. volume per decade
    parts = []
    for c in CLASSES:
        p = PROC / f"tm_class{c}.parquet"
        if not p.exists():
            continue
        d = pl.read_parquet(p, columns=["serial_number", "filing_date", "registration_date"]).filter(
            pl.col("filing_date").fill_null("").str.len_chars() >= 8).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"), pl.lit(c).alias("cls"))
        parts.append(d)
    d = pl.concat(parts).unique("serial_number").drop_nulls("fy").filter(pl.col("fy").is_between(1870, 2026))
    dec = d.with_columns((pl.col("fy") // 10 * 10).alias("dec"))
    out["volume"]["corpus_by_decade"] = {str(k): v for k, v in
        dec.group_by("dec").len().sort("dec").iter_rows()}
    for c in BIG:
        out["volume"][c] = {str(k): v for k, v in
            dec.filter(pl.col("cls") == c).group_by("dec").len().sort("dec").iter_rows()}
    log("volume: " + json.dumps(out["volume"]["corpus_by_decade"]))

    # window coverage: share of filings per decade with >=500 same-class filings in both W-year windows
    for W in (5, 10):
        cov = []
        for c in CLASSES:
            sub = d.filter(pl.col("cls") == c).sort("fy")
            ys = sub["fy"].to_numpy()
            if len(ys) < 2 * MIN_REF:
                continue
            lo_p = np.searchsorted(ys, ys - W, side="left"); hi_p = np.searchsorted(ys, ys, side="left")
            lo_f = np.searchsorted(ys, ys, side="right"); hi_f = np.searchsorted(ys, ys + W, side="right")
            ok = ((hi_p - lo_p) >= MIN_REF) & ((hi_f - lo_f) >= MIN_REF)
            cov.append(pl.DataFrame({"fy": ys, "ok": ok}))
        cv = pl.concat(cov).with_columns((pl.col("fy") // 10 * 10).alias("dec"))
        out["window_coverage"][f"W{W}"] = {str(k): round(float(v), 3) for k, v in
            cv.group_by("dec").agg(pl.col("ok").mean()).sort("dec").iter_rows()}
        log(f"W={W} coverage: " + json.dumps(out["window_coverage"][f"W{W}"]))

    # 2. vocabulary coverage by decade
    m = joblib.load(PROC / "topic_model.joblib")
    vocab = set(m["vocabulary"].keys()) if isinstance(m["vocabulary"], dict) else set(m["vocabulary"])
    rng = np.random.default_rng(7)
    for c in ("009", "025"):
        t = pl.read_parquet(PROC / f"tm_class{c}.parquet", columns=["filing_date", "goods_services"]).filter(
            pl.col("goods_services").is_not_null() & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        ).with_columns((pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False) // 10 * 10).alias("dec"))
        res = {}
        for decade in sorted(t["dec"].unique().drop_nulls().to_list()):
            sub = t.filter(pl.col("dec") == decade)
            n = min(2000, sub.height)
            if n < 50:
                continue
            texts = sub.sample(n, seed=7)["goods_services"].to_list()
            hit = tot = 0
            for txt in texts:
                toks = re.findall(r"[a-z][a-z\-]{2,}", txt.lower())
                bi = [f"{a} {b}" for a, b in zip(toks, toks[1:])]
                allt = toks + bi
                tot += len(allt); hit += sum(1 for x in allt if x in vocab)
            res[str(decade)] = round(hit / max(tot, 1), 3)
        out["vocab"][c] = res
        log(f"vocab {c}: " + json.dumps(res))

    # 3. outcomes by registration decade
    ev = pl.scan_parquet(PROC / "case_events.parquet").select("serial_number", "code", "date").collect()
    anyev = ev.group_by("serial_number").len().rename({"len": "n_ev"})
    c8 = ev.filter(pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)).with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    regs = pl.concat(parts).unique("serial_number").filter(
        pl.col("registration_date").fill_null("").str.len_chars() >= 8).with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")).drop_nulls("rd").with_columns(
        (pl.col("rd").dt.year() // 10 * 10).alias("rdec"))
    j = regs.join(anyev, on="serial_number", how="left").join(c8, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age"))
    tab = j.group_by("rdec").agg(
        pl.len().alias("n_regs"),
        (pl.col("n_ev").fill_null(0) > 0).mean().alias("share_any_event"),
        pl.col("cd").is_not_null().mean().alias("share_c8_event"),
        ((pl.col("age") >= 4.0) & (pl.col("age") < 8.5)).fill_null(False).mean().alias("gate_fail_rate")).sort("rdec")
    out["outcomes"] = {str(r["rdec"]): {"n_regs": r["n_regs"], "any_event": round(r["share_any_event"], 3),
                                        "c8_event": round(r["share_c8_event"], 3),
                                        "gate_fail_4_85": round(r["gate_fail_rate"], 3)}
                       for r in tab.iter_rows(named=True) if r["rdec"] and r["rdec"] >= 1880}
    log("outcomes: " + json.dumps(out["outcomes"]))
    (RES / "historical_feasibility.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
