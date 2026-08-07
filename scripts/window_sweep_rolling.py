"""Is five years the window that maximises signal, under per-filing references?

The paper fixes W = 1,826 days (five years) on each side. That choice predates
the move to per-filing reference windows: the earlier W-sweep was run on
class-year buckets, so it does not answer the question for the measure now in
use. This re-scores one class at several window widths under the per-filing
construction and reports, for each, the first-gate quintile contrast.

Signal is judged by the gate contrast rather than by anything internal to the
measure. A wider window pools more filings, which lowers variance but blurs the
distinction between recent and distant language; a narrower one does the
reverse. There is no reason the optimum should be five years.

Usage:  python scripts/window_sweep_rolling.py [CLASS] [W_YEARS ...]
        default: 009  2 3 5 7 10
Output: paper/results/window_sweep_rolling.json
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

CLS = sys.argv[1] if len(sys.argv) > 1 else "009"
WYEARS = [float(a) for a in sys.argv[2:]] or [2, 3, 5, 7, 10]
T = 50
MIN_REF = 500
YEAR_LO, YEAR_HI = 1995, 2019
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
EPS = 1e-12


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


def main() -> int:
    df = pl.read_parquet(PROC / f"tm_class{CLS}.parquet",
                         columns=["serial_number", "filing_date", "goods_services",
                                  "registration_date"]).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd")
    ).drop_nulls("fd").sort("fd")
    n = df.height
    log(f"[load] {n:,} filings in class {CLS}")

    m = joblib.load(PROC / "topic_model.joblib")
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b", ngram_range=(1, 2))
    lda = m["lda"]
    theta = np.empty((n, T), dtype=np.float64)
    texts = df["goods_services"].to_list()
    for s in range(0, n, 200_000):
        e = min(s + 200_000, n)
        theta[s:e] = lda.transform(vec.transform(texts[s:e]))
    del texts
    np.clip(theta, EPS, None, out=theta)
    theta /= theta.sum(axis=1, keepdims=True)
    log("[theta] done")

    days = df["fd"].to_numpy().astype("datetime64[D]").astype(np.int64)
    years = df["fd"].dt.year().to_numpy()

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())

    base = df.select("serial_number", "registration_date").with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry"))
    base = base.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed1")
    ).filter(pl.col("ry").is_between(REG_LO, REG_HI)).select(
        "serial_number", "ry", "failed1")

    out = {"class": CLS, "T": T, "gate_window": [GATE_LO, GATE_HI], "by_W": {}}
    for wy in WYEARS:
        W = int(round(wy * 365.25))
        lo_p = np.searchsorted(days, days - W, side="left")
        hi_p = np.searchsorted(days, days, side="left")
        lo_f = np.searchsorted(days, days, side="right")
        hi_f = np.searchsorted(days, days + W, side="right")
        q_p, n_p = window_means(theta, lo_p, hi_p, T)
        k_past = kl_rows(theta, q_p); del q_p; gc.collect()
        q_f, n_f = window_means(theta, lo_f, hi_f, T)
        k_fut = kl_rows(theta, q_f); del q_f; gc.collect()
        ok = ((n_p >= MIN_REF) & (n_f >= MIN_REF)
              & (days - W >= days[0]) & (days + W <= days[-1])
              & (years >= YEAR_LO) & (years <= YEAR_HI))
        dkl = np.where(ok, k_past - k_fut, np.nan)

        sc = df.select("serial_number").with_columns(pl.Series("dkl", dkl))
        j = base.join(sc, on="serial_number", how="inner").filter(
            pl.col("dkl").is_finite())
        j = j.with_columns(
            pl.format("{}", pl.col("ry")).alias("cell")).sort(["cell", "dkl", "serial_number"])
        j = j.with_columns(
            ((pl.col("dkl").rank("ordinal").over("cell") - 1) * 5
             // pl.len().over("cell")).cast(pl.Int8).alias("q"))
        g = j.group_by("q").agg(pl.col("failed1").mean().alias("p"),
                                pl.len().alias("nn")).sort("q")
        v = [float(r["p"]) for r in g.iter_rows(named=True)]
        p1, p5 = v[0], v[4]
        n1, n5 = int(g["nn"][0]), int(g["nn"][4])
        se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
        out["by_W"][f"{wy:g}"] = {
            "W_days": W, "n_scored": int(ok.sum()), "n_gate": int(j.height),
            "sd_dkl": float(np.nanstd(dkl)), "quintiles": v,
            "lift": p5 - p1, "se": se, "t": (p5 - p1) / se}
        log(f"  W={wy:>4g}y ({W:5d}d): scored {int(ok.sum()):>9,}  gate n={j.height:>8,}  "
            f"sd(L)={np.nanstd(dkl):.4f}  lift={100*(p5-p1):+5.2f}pp  t={(p5-p1)/se:+5.1f}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "window_sweep_rolling.json").write_text(json.dumps(out, indent=1))
    log("[done] window_sweep_rolling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
