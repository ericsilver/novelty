"""Mixing the reference windows, under the production scoring.

"Leading" is relative to a reference, and the two references need not be the
same width. Scoring against a short past and a long future asks whether the
filing anticipates where the industry is heading (foresight); a long past and a
short future asks whether it has broken with what the industry had established
(past-rupture). The earlier version of this decomposition was run on the
retired term-level, calendar-year scoring, and its numbers do not transfer.
This is the same design on the production representation (T=50 themes) and
per-filing windows, for every class.

Per class, every filing's theme distribution is computed once and scored
against per-filing windows of W in {3, 5, 7} years on each side. Five lead
variants follow: sym3, sym5, sym7, asym37 = past(W3) - future(W7) (foresight),
asym73 = past(W7) - future(W3) (past-rupture). All five are evaluated on the
identical sample: filings whose 7-year windows lie inside the corpus and whose
3-year windows each hold at least MIN_REF filings.

Outcomes: event-dated first-gate failure (registrations 2002-2018, failure at
age 4.0-8.5), quintiles within class x registration-year cells; registration
completion (filings 1995-2018), quintiles within class x filing-year cells,
reported as the top-minus-bottom contrast and the inverse-U depth
q3 - (q1+q5)/2.

Usage:  python scripts/window_mix_rolling.py [CLASS ...]     default: all 45
Output: paper/results/window_mix_rolling.json (+ per-class partials in
        paper/results/window_mix_parts/ so a run can resume)
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
PARTS = RES / "window_mix_parts"

CLASSES = sys.argv[1:] or [f"{i:03d}" for i in range(1, 46)]
T = 50
WYEARS = (3, 5, 7)
MIN_REF = 500
YEAR_LO, YEAR_HI = 1995, 2019
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
EPS = 1e-12
VARIANTS = ("sym3", "sym5", "sym7", "asym37", "asym73")
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def window_means(theta, lo, hi, T):
    n = theta.shape[0]
    out = np.empty((n, T), dtype=np.float64)
    cnt = (hi - lo).astype(np.int64)
    run = np.zeros(T, dtype=np.float64)
    cl = ch = 0
    for i in range(n):
        while ch < hi[i]:
            run += theta[ch]; ch += 1
        while cl < lo[i]:
            run -= theta[cl]; cl += 1
        out[i] = run / cnt[i] if cnt[i] > 0 else np.nan
    return out, cnt


def kl_rows(P, Q):
    Q = np.clip(Q, EPS, None)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return np.einsum("ij,ij->i", P, np.log(P) - np.log(Q))


def quintile_stats(df: pl.DataFrame, var: str, y: str, cells: list[str]) -> dict:
    s = df.sort(cells + [var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over(cells) - 1) * 5
         // pl.len().over(cells)).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col(y).mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]
    n = [int(v) for v in g["n"]]
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"n": int(df.height), "quintiles": p, "lift": p[4] - p[0], "se": se,
            "depth": p[2] - 0.5 * (p[0] + p[4])}


def run_class(cls: str, lda, vec, ev: pl.DataFrame) -> pl.DataFrame | None:
    f = PROC / f"tm_class{cls}.parquet"
    if not f.exists():
        return None
    df = pl.read_parquet(f, columns=["serial_number", "filing_date", "goods_services",
                                     "registration_date"]).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd")
    ).drop_nulls("fd").sort("fd")
    n = df.height
    theta = np.empty((n, T), dtype=np.float64)
    texts = df["goods_services"].to_list()
    for s in range(0, n, 200_000):
        theta[s:s + 200_000] = lda.transform(vec.transform(texts[s:s + 200_000]))
    del texts
    np.clip(theta, EPS, None, out=theta)
    theta /= theta.sum(axis=1, keepdims=True)

    days = df["fd"].to_numpy().astype("datetime64[D]").astype(np.int64)
    years = df["fd"].dt.year().to_numpy()
    ok = (years >= YEAR_LO) & (years <= YEAR_HI)
    past, fut = {}, {}
    for wy in WYEARS:
        W = int(round(wy * 365.25))
        lo_p = np.searchsorted(days, days - W, side="left")
        hi_p = np.searchsorted(days, days, side="left")
        lo_f = np.searchsorted(days, days, side="right")
        hi_f = np.searchsorted(days, days + W, side="right")
        q, n_p = window_means(theta, lo_p, hi_p, T)
        past[wy] = kl_rows(theta, q); del q; gc.collect()
        q, n_f = window_means(theta, lo_f, hi_f, T)
        fut[wy] = kl_rows(theta, q); del q; gc.collect()
        ok &= (n_p >= MIN_REF) & (n_f >= MIN_REF)
        ok &= (days - W >= days[0]) & (days + W <= days[-1])
    del theta
    gc.collect()

    cols = {"sym3": past[3] - fut[3], "sym5": past[5] - fut[5], "sym7": past[7] - fut[7],
            "asym37": past[3] - fut[7], "asym73": past[7] - fut[3]}
    out = df.select("serial_number", "registration_date").with_columns(
        pl.Series("fy", years), pl.Series("ok", ok),
        *[pl.Series(k, v) for k, v in cols.items()]).filter(pl.col("ok")).drop("ok")
    out = out.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).with_columns(
        pl.col("rd").is_not_null().cast(pl.Float64).alias("registered"),
        pl.col("rd").dt.year().alias("ry"), pl.lit(cls).alias("cls"))
    out = out.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed")
    ).select("serial_number", "cls", "fy", "ry", "registered", "failed", *VARIANTS)
    log(f"  [{cls}] {n:,} filings, {out.height:,} scored on the common sample")
    return out


def main() -> int:
    m = joblib.load(PROC / "topic_model.joblib")
    lda = m["lda"]
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    PARTS.mkdir(parents=True, exist_ok=True)

    parts = []
    for cls in CLASSES:
        p = PARTS / f"{cls}.parquet"
        if p.exists():
            parts.append(pl.read_parquet(p))
            log(f"  [{cls}] cached")
            continue
        d = run_class(cls, lda, vec, ev)
        if d is None:
            continue
        d.write_parquet(p)
        parts.append(d)
        gc.collect()
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()

    gate = d.filter(pl.col("ry").is_between(REG_LO, REG_HI))
    reg = d.filter(pl.col("fy").is_between(1995, 2018))
    out = {"T": T, "windows_years": WYEARS, "min_ref": MIN_REF,
           "n_gate": int(gate.height), "n_reg": int(reg.height),
           "gate_pooled": {}, "reg_pooled": {}, "per_class": {}}
    for v in VARIANTS:
        out["gate_pooled"][v] = quintile_stats(gate, v, "failed", ["cls", "ry"])
        out["reg_pooled"][v] = quintile_stats(reg, v, "registered", ["cls", "fy"])
        g, r = out["gate_pooled"][v], out["reg_pooled"][v]
        log(f"  {v:7s} gate lift {100*g['lift']:+.2f}pp (SE {100*g['se']:.2f})   "
            f"reg lift {100*r['lift']:+.2f}pp depth {100*r['depth']:+.2f}pp")
    for cls in sorted(d["cls"].unique().to_list()):
        gc_, rc = gate.filter(pl.col("cls") == cls), reg.filter(pl.col("cls") == cls)
        row = {"n_gate": int(gc_.height), "n_reg": int(rc.height)}
        for v in ("sym5", "asym37", "asym73"):
            if gc_.height >= 3000:
                row[f"gate_{v}"] = quintile_stats(gc_, v, "failed", ["ry"])
            if rc.height >= 3000:
                row[f"reg_{v}"] = quintile_stats(rc, v, "registered", ["fy"])
        out["per_class"][cls] = row
    (RES / "window_mix_rolling.json").write_text(json.dumps(out, indent=1))
    log("[done] window_mix_rolling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
