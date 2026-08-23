"""Novel combinations, at the word level and the theme level, and what the gate does to each.

Section 1 states that the construct is nearly blind to combination. Two
combination measures make that concrete and testable on the software class.

1. WORD RECOMBINATION. For each filing, the share of its adjacent word pairs
   (bigrams) that are new to the class while both component words are
   established: the bigram appears fewer than BIGRAM_MIN times in the class's
   filings of the five prior calendar years, and each unigram at least
   UNIGRAM_MIN times. "Uber for vacation rentals" at the word level: familiar
   parts, new adjacency. Reported: gate failure by quintile of the
   recombination share, raw and net of the filing's own lead and atypicality
   and description length.

2. THEME-PORTFOLIO SURPRISE. The top-two pair lead of theme_pair_surprise.py,
   extended to every pair of present themes (mass >= 0.10), weighted by the
   product of the two themes' masses. Same outcome, same controls.

Output: paper/results/combination_measures_{CLS}.json
"""
from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLS = sys.argv[1] if len(sys.argv) > 1 else "009"
BIGRAM_MIN, UNIGRAM_MIN = 5, 50
THRESH = 0.10
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
W = 5


def log(m): print(m, file=sys.stderr, flush=True)


def tokens(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z][a-z\-]{2,}", text.lower())


def contrast(df, var):
    d = df.filter(pl.col(var).is_finite())
    s = d.sort(["ry", var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over("ry") - 1) * 5 // pl.len().over("ry")).cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"n": int(d.height), "quintiles": p, "lift": p[4] - p[0], "se": se, "t": (p[4] - p[0]) / se}


def main() -> int:
    tm = pl.read_parquet(PROC / f"tm_class{CLS}.parquet",
                         columns=["serial_number", "filing_date", "registration_date", "goods_services"]).filter(
        pl.col("goods_services").is_not_null() & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
                   pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")).sort("fy")
    sc = pl.read_parquet(PROC / f"rolling_surprise_class{CLS}.parquet",
                         columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"]).filter(
        pl.col("topic_dkl").is_finite())
    d = tm.join(sc, on="serial_number", how="inner").with_columns(
        pl.col("topic_dkl").alias("L"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    log(f"[load] {d.height:,} scored filings in class {CLS}")

    # ---- word recombination, per calendar year against the prior five ----
    years = sorted(d["fy"].unique().to_list())
    uni_year: dict[int, defaultdict] = {}
    bi_year: dict[int, defaultdict] = {}
    texts_by_year = {y: d.filter(pl.col("fy") == y)["goods_services"].to_list() for y in years}
    for y in years:
        u, b = defaultdict(int), defaultdict(int)
        for t in texts_by_year[y]:
            tk = tokens(t)
            for w_ in set(tk):
                u[w_] += 1
            for i in range(len(tk) - 1):
                b[(tk[i], tk[i + 1])] += 1
        uni_year[y], bi_year[y] = u, b
    log("[vocab] per-year counts done")

    rec_share = {}
    for y in years:
        if y - W < years[0] or y < 1995 or y > 2019:
            continue
        u, b = defaultdict(int), defaultdict(int)
        for yy in range(y - W, y):
            for k, v in uni_year.get(yy, {}).items():
                u[k] += v
            for k, v in bi_year.get(yy, {}).items():
                b[k] += v
        sub = d.filter(pl.col("fy") == y)
        vals = []
        for t in sub["goods_services"].to_list():
            tk = tokens(t)
            pairs = [(tk[i], tk[i + 1]) for i in range(len(tk) - 1)]
            elig = [p for p in pairs if u.get(p[0], 0) >= UNIGRAM_MIN and u.get(p[1], 0) >= UNIGRAM_MIN]
            if not elig:
                vals.append(None); continue
            vals.append(sum(1 for p in elig if b.get(p, 0) < BIGRAM_MIN) / len(elig))
        rec_share[y] = (sub["serial_number"].to_list(), vals)
        log(f"  [{y}] recombination shares computed ({sub.height:,})")
    sers, vals = [], []
    for y, (s_, v_) in rec_share.items():
        sers += s_; vals += v_
    rec = pl.DataFrame({"serial_number": sers, "recomb": vals})
    del uni_year, bi_year, texts_by_year
    gc.collect()

    # ---- theme portfolio pair surprise ----
    m = joblib.load(PROC / "topic_model.joblib")
    vec = CountVectorizer(vocabulary=m["vocabulary"], lowercase=True,
                          token_pattern=r"(?u)\b[a-z][a-z\-]{2,}\b", ngram_range=(1, 2))
    dd = d.sort("fy")
    texts = dd["goods_services"].to_list()
    n = len(texts)
    theta = np.empty((n, 50), dtype=np.float32)
    for s_ in range(0, n, 200_000):
        theta[s_:s_ + 200_000] = m["lda"].transform(vec.transform(texts[s_:s_ + 200_000]))
    del texts
    present = theta >= THRESH
    yrs = dd["fy"].to_numpy()
    marg, co, cnt = {}, {}, {}
    for y in np.unique(yrs):
        mk = yrs == y
        P = present[mk].astype(np.float32)
        cnt[y] = int(mk.sum()); marg[y] = P.sum(0); co[y] = P.T @ P
    eps = 1e-6
    Sm = np.full(n, np.nan); Sp = np.full(n, np.nan)
    for y in np.unique(yrs):
        past = [v for v in range(y - W, y) if v in cnt]
        fut = [v for v in range(y + 1, y + W + 1) if v in cnt]
        if sum(cnt[v] for v in past) < 2000 or sum(cnt[v] for v in fut) < 2000:
            continue
        def win(vs):
            N = sum(cnt[v] for v in vs)
            return sum(marg[v] for v in vs) / N, sum(co[v] for v in vs) / N
        Mp, Cp = win(past); Mf, Cf = win(fut)
        idx = np.where(yrs == y)[0]
        for i in idx:
            th = np.where(present[i])[0]
            if len(th) < 2:
                continue
            wsum = ssum_p = ssum_f = 0.0
            for a_ in range(len(th)):
                for b_ in range(a_ + 1, len(th)):
                    ta, tb = th[a_], th[b_]
                    w_ = float(theta[i, ta] * theta[i, tb])
                    sp_ = -np.log((Cp[ta, tb] + eps) / (Mp[ta] * Mp[tb] + eps))
                    sf_ = -np.log((Cf[ta, tb] + eps) / (Mf[ta] * Mf[tb] + eps))
                    wsum += w_; ssum_p += w_ * sp_; ssum_f += w_ * sf_
            if wsum > 0:
                Sm[i], Sp[i] = ssum_p / wsum, ssum_f / wsum
    port = dd.select("serial_number").with_columns(pl.Series("Sp_past", Sm), pl.Series("Sp_fut", Sp)).with_columns(
        (pl.col("Sp_past") - pl.col("Sp_fut")).alias("port_lead"),
        ((pl.col("Sp_past") + pl.col("Sp_fut")) / 2).alias("port_atyp"))
    log("[portfolio] pair surprise done")

    # ---- gate outcomes ----
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    g = d.drop("goods_services").join(rec, on="serial_number", how="left").join(port, on="serial_number", how="left"
        ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry")).filter(
        pl.col("ry").is_between(REG_LO, REG_HI)).join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64).alias("failed"))
    g = g.with_columns(pl.col("recomb").cast(pl.Float64))
    out = {"class": CLS, "n_gate": int(g.height), "base": float(g["failed"].mean()),
           "recomb_mean": float(g["recomb"].mean()), "recomb_sd": float(g["recomb"].std()),
           "gate": {}}
    for v in ("recomb", "port_lead", "port_atyp"):
        out["gate"][v] = contrast(g, v)
        c = out["gate"][v]
        log(f"  {v:9s} Q5-Q1 {100*c['lift']:+.2f}pp (t {c['t']:+.1f})  n={c['n']:,}")
    # net of L and A (and each other)
    gg = g.filter(pl.col("recomb").is_finite() & pl.col("port_lead").is_finite()
                  & pl.col("L").is_finite() & pl.col("A").is_finite())
    X = np.column_stack([np.ones(gg.height), gg["L"].to_numpy(), gg["A"].to_numpy()])
    for v in ("recomb", "port_lead", "port_atyp"):
        beta, *_ = np.linalg.lstsq(X, gg[v].to_numpy(), rcond=None)
        gg = gg.with_columns(pl.Series(f"{v}_res", gg[v].to_numpy() - X @ beta))
        out["gate"][f"{v}_net_LA"] = contrast(gg, f"{v}_res")
        c = out["gate"][f"{v}_net_LA"]
        log(f"  {v:9s} net L,A Q5-Q1 {100*c['lift']:+.2f}pp (t {c['t']:+.1f})")
    out["corr"] = {f"{a}~{b}": float(gg.select(pl.corr(a, b)).item())
                   for a, b in (("recomb", "L"), ("recomb", "A"), ("port_lead", "L"), ("port_atyp", "A"), ("recomb", "port_lead"))}
    log(f"  corr: {out['corr']}")
    (RES / f"combination_measures_{CLS}.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
