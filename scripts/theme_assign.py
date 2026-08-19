"""Assign every filing its dominant theme, then aggregate by class and month.

The theme explorer lists what each theme is made of; this produces what each
theme is *used for*. For a fitted model it computes, per theme: how many filings
in each Nice class it dominates, a monthly count series split by class, and the
lifecycle outcomes of the filings it dominates -- registration, first-gate
failure, and whether the owner ever reached SEC reporting.

Dominant theme rather than weighted mass, because the page has to be readable:
"this theme is the main thing 4,812 filings are about" is a sentence a reader
can check, where a mass share is not. Filings are counted once each.

Classes are capped at CAP filings (most fall under it) so the pass is bounded;
the cap is recorded per class so the page can say which series are samples.

Usage:  python scripts/theme_assign.py [T] [CAP]
Output: paper/results/theme_assign_T{T}.json
"""
from __future__ import annotations

import gc, json, sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
T = int(sys.argv[1]) if len(sys.argv) > 1 else 500
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
GATE_LO, GATE_HI = 4.0, 8.5


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    mp = PROC / ("topic_model.joblib" if T == 50 else f"topic_model_T{T}.joblib")
    m = joblib.load(mp); lda, vocab = m["lda"], m["vocabulary"]
    vec = CountVectorizer(vocabulary=vocab, lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("gd")
    ).drop_nulls("gd").group_by("serial_number").agg(pl.col("gd").min())
    cw = set(pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")["owner_name"].to_list())
    log(f"[setup] {ev.height:,} gate events, {len(cw):,} SEC-matched owners")

    months, outcomes, capped = {}, {}, {}
    for c in CLASSES:
        f = PROC / f"tm_class{c}.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f, columns=["serial_number", "filing_date",
                                        "registration_date", "owner_name",
                                        "goods_services"]).filter(
            pl.col("goods_services").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        n_all = d.height
        if n_all > CAP:
            d = d.sample(CAP, seed=42)
        capped[c] = {"total": n_all, "used": d.height}
        texts = d["goods_services"].to_list()
        dom = np.empty(len(texts), dtype=np.int32)
        for s in range(0, len(texts), 50_000):
            e = min(s + 50_000, len(texts))
            dom[s:e] = lda.transform(vec.transform(texts[s:e])).argmax(axis=1)
        del texts; gc.collect()

        d = d.with_columns(
            pl.Series("theme", dom),
            pl.col("filing_date").str.slice(0, 6).alias("ym"),
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
            .cast(pl.Int64).alias("reg"),
            pl.col("owner_name").is_in(list(cw)).cast(pl.Int64).alias("sec"),
        ).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"))
        d = d.join(ev, on="serial_number", how="left").with_columns(
            (((pl.col("gd") - pl.col("rd")).dt.total_days() / 365.25)
             .is_between(GATE_LO, GATE_HI, closed="left")).fill_null(False)
            .cast(pl.Int64).alias("failed"))

        g = d.group_by(["theme", "ym"]).len()
        for r in g.iter_rows():
            months.setdefault(str(r[0]), {}).setdefault(c, {})[r[1]] = int(r[2])
        # SEC reporting is a property of the OWNER, so counting it over every
        # filing gives a firm with 300 marks 300 votes. The debut split counts
        # each owner once, on its first filing, which is the denominator the
        # paper's firm-level results use.
        d = d.with_columns(
            (pl.col("filing_date") == pl.col("filing_date").min().over("owner_name"))
            .alias("is_debut"))
        for lab, sub in (("", d), ("debut_", d.filter(pl.col("is_debut")))):
            o = sub.group_by("theme").agg(
                pl.len().alias("n"), pl.col("reg").sum().alias("reg"),
                pl.col("sec").sum().alias("sec"),
                pl.col("failed").sum().alias("failed"))
            for r in o.iter_rows(named=True):
                t = outcomes.setdefault(str(r["theme"]),
                                        {"n": 0, "reg": 0, "sec": 0, "failed": 0,
                                         "debut_n": 0, "debut_reg": 0,
                                         "debut_sec": 0, "debut_failed": 0,
                                         "by_class": {}})
                t[lab + "n"] += r["n"]; t[lab + "reg"] += r["reg"]
                t[lab + "sec"] += r["sec"]; t[lab + "failed"] += r["failed"]
                if not lab:
                    t["by_class"][c] = r["n"]
            del o
        log(f"  [{c}] {d.height:,} of {n_all:,} filings assigned")
        del d, dom, g, o; gc.collect()

    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"theme_assign_T{T}.json").write_text(json.dumps(
        {"T": T, "cap": CAP, "classes": capped,
         "outcomes": outcomes, "months": months}))
    log(f"[done] theme_assign_T{T}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
