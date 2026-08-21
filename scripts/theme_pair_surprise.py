"""Is there a surprise measure for the *combination* of themes, and does it predict anything?

The construct scores which themes a filing draws on. It is nearly blind to how
it combines them (theme_combination_test.py). But a combination can be scored
on its own: "Uber for vacation rentals" is two familiar themes in a pairing
that was rare when it was filed and common afterwards. This builds that
measure and asks how it fares against the same outcome as the main text.

For each filing take its two heaviest themes (a, b). In the five calendar
years before the filing year and the five after, compute within the class

    lift_window = P(a and b both present) / (P(a) P(b))

where "present" means a theme carries at least 0.10 of a filing's mass. Then

    pair surprise, past   S- = -log lift_past      (rare pairing before)
    pair surprise, future S+ = -log lift_future    (rare pairing after)
    pair lead             L_pair = S- - S+         (pairing the class moved toward)
    pair atypicality      A_pair = (S- + S+) / 2

These mirror lead and atypicality, but for the arrangement rather than the
parts. Windows are annual buckets (the cost of a pairwise count is a 50x50
table per class-year, which is cheap; per-filing windows would need one per
filing and are not needed to answer whether the signal exists).

Outcome: event-dated first-gate failure, registrations 2002-2018, one class.
Reported: the quintile contrast for L_pair and A_pair, raw and after
residualising on the filing's own lead and atypicality, plus correlations.

Usage:  python scripts/theme_pair_surprise.py [CLASS]    default 009
Output: paper/results/theme_pair_surprise_{CLASS}.json
"""
from __future__ import annotations

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
T, THRESH, W = 50, 0.10, 5
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"


def log(m): print(m, file=sys.stderr, flush=True)


def contrast(df, var):
    s = df.sort(["ry", var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over("ry") - 1) * 5 // pl.len().over("ry"))
        .cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"quintiles": p, "lift": p[4] - p[0], "se": se, "t": (p[4] - p[0]) / se}


def main() -> int:
    tm = pl.read_parquet(PROC / f"tm_class{CLS}.parquet",
                         columns=["serial_number", "filing_date", "registration_date",
                                  "goods_services"]).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("fy"))
    sc = pl.read_parquet(PROC / f"rolling_surprise_class{CLS}.parquet",
                         columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future",
                                  "topic_dkl"]).filter(pl.col("topic_dkl").is_finite())
    d = tm.join(sc, on="serial_number", how="inner").unique("serial_number").sort("fy")
    n = d.height
    log(f"[load] {n:,} scored filings in class {CLS}")

    m = joblib.load(PROC / "topic_model.joblib")
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    texts = d["goods_services"].to_list()
    theta = np.empty((n, T), dtype=np.float32)
    for s in range(0, n, 200_000):
        theta[s:s + 200_000] = m["lda"].transform(vec.transform(texts[s:s + 200_000]))
    del texts
    present = theta >= THRESH
    order = np.argsort(-theta, axis=1)
    a, b = order[:, 0], order[:, 1]
    log("[theta] done")

    # Per-year marginal and pair presence counts, then window sums.
    years = d["fy"].to_numpy()
    ys = np.arange(years.min(), years.max() + 1)
    Pf = present.astype(np.float32)
    marg = {}; co = {}; cnt = {}
    for y in ys:
        mk = years == y
        cnt[y] = int(mk.sum())
        marg[y] = Pf[mk].sum(axis=0)
        co[y] = Pf[mk].T @ Pf[mk]
    def window(y, lo, hi):
        yy = [v for v in range(y + lo, y + hi + 1) if v in cnt]
        N = sum(cnt[v] for v in yy)
        if N < 2000:
            return None
        M = sum(marg[v] for v in yy) / N
        C = sum(co[v] for v in yy) / N
        return M, C
    eps = 1e-6
    Sm = np.full(n, np.nan); Sp = np.full(n, np.nan)
    for y in ys:
        mk = np.where(years == y)[0]
        wp, wf = window(y, -W, -1), window(y, 1, W)
        if wp is None or wf is None:
            continue
        for S, (M, C) in ((Sm, wp), (Sp, wf)):
            pa, pb = M[a[mk]], M[b[mk]]
            pab = C[a[mk], b[mk]]
            S[mk] = -np.log((pab + eps) / (pa * pb + eps))
    d = d.with_columns(pl.Series("S_minus", Sm), pl.Series("S_plus", Sp)).filter(
        pl.col("S_minus").is_finite() & pl.col("S_plus").is_finite()
    ).with_columns(
        (pl.col("S_minus") - pl.col("S_plus")).alias("L_pair"),
        ((pl.col("S_minus") + pl.col("S_plus")) / 2).alias("A_pair"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"),
        pl.col("topic_dkl").alias("L"),
    )
    log(f"[pairs] {d.height:,} filings with both windows")

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    g = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry")).filter(
        pl.col("ry").is_between(REG_LO, REG_HI)
    ).join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
                   .fill_null(False).cast(pl.Float64).alias("failed"))
    log(f"[gate] {g.height:,} registrations {REG_LO}-{REG_HI}")

    # Residualise the pair measures on the filing's own lead and atypicality
    # (plus cohort means), so the contrast is the increment over the parts.
    X = np.column_stack([np.ones(g.height), g["L"].to_numpy(), g["A"].to_numpy()])
    def resid(v):
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ beta
    g = g.with_columns(pl.Series("L_pair_res", resid(g["L_pair"].to_numpy())),
                       pl.Series("A_pair_res", resid(g["A_pair"].to_numpy())))

    out = {"class": CLS, "T": T, "thresh": THRESH, "W_years": W, "n_scored": int(d.height),
           "n_gate": int(g.height), "base": float(g["failed"].mean()),
           "sd": {k: float(g[k].std()) for k in ("L_pair", "A_pair", "L", "A")},
           "corr": {f"{x}~{y}": float(g.select(pl.corr(x, y)).item())
                    for x, y in (("L_pair", "L"), ("A_pair", "A"), ("L_pair", "A_pair"),
                                 ("A_pair", "L"), ("L_pair", "A"))},
           "gate": {k: contrast(g, k) for k in ("L", "A", "L_pair", "A_pair",
                                                 "L_pair_res", "A_pair_res")}}
    for k, v in out["gate"].items():
        log(f"  {k:11s} Q5-Q1 {100*v['lift']:+.2f}pp (SE {100*v['se']:.2f}, t {v['t']:+.1f})")
    log(f"  corr: {out['corr']}")
    RES.mkdir(exist_ok=True, parents=True)
    (RES / f"theme_pair_surprise_{CLS}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
