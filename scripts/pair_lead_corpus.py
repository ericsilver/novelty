"""Corpus-wide pair lead: does the one-class (009) result generalize?

Takes over the construction of theme_pair_surprise.py exactly -- top-2
themes per filing at the T=50 production model, presence at theta >= 0.10,
pair lift against the class's five calendar years before and after,
windows honored only at >= 2,000 filings, L_pair = S- - S+ -- and runs it
over every Nice class.

Stage A (generality): per class, the top-vs-bottom L_pair quintile
contrast in failure at the five-year proof (event-dated C8../C71T at
registration age 4.0-8.5, cohorts 2002-2018), quintiles within
registration year; raw and residualized on the filing's own lead and
atypicality (per class, as the original). Pooled: quintiles within
class x registration year, contrast with owner-clustered SEs (normalized
owner), raw and residualized.

Stage B (the 035 flip): for 009 and 035, the contrast by filing-year era,
within dominant-theme x registration-year cells, and on two random
halves.

Stage C (anchoring): for 009, pair references re-anchored on each
filing's own date via monthly buckets (window = the ~60 months on each
side, floors unchanged), compared with the calendar-bucket construction
on the same filings.

Outputs: paper/results/pair_lead_corpus.json
         paper/results/pair_lead_corpus.tex
Log:     paper/v3/_eval/logs/pair_lead_corpus.log (stderr)
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
CLASSES = [f"{i:03d}" for i in range(1, 46)]
T, THRESH, W = 50, 0.10, 5
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
FLOOR = 2000
PER_CLASS_MIN_GATE = 3000
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
ERAS = [("1995-1999", 1995, 1999), ("2000-2004", 2000, 2004),
        ("2005-2007", 2005, 2007), ("2008-2014", 2008, 2014),
        ("2015-2018", 2015, 2018)]
SEED = 20260831


def log(m): print(m, file=sys.stderr, flush=True)


def contrast_within(df, var, cells):
    """Q5-Q1 contrast of `failed` on quintiles of var cut within `cells`."""
    s = df.sort(cells + [var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over(cells) - 1) * 5 // pl.len().over(cells))
        .cast(pl.Int8).alias("q"))
    g = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
    p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
    if len(p) < 5 or min(n) < 25:
        return None
    se = ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
    return {"quintiles": p, "n": int(sum(n)), "lift": p[4] - p[0], "se": se,
            "t": (p[4] - p[0]) / se if se > 0 else None}


def clustered_contrast(df, var, cells):
    """Pooled Q5-Q1 on quintiles within cells, failed demeaned within cells,
    owner-clustered SE (each owner one cluster; demeaning estimation error
    ignored, matching the paper's raw-contrast convention)."""
    s = df.sort(cells + [var, "serial_number"]).with_columns(
        ((pl.col(var).rank("ordinal").over(cells) - 1) * 5 // pl.len().over(cells))
        .cast(pl.Int8).alias("q"),
        (pl.col("failed") - pl.col("failed").mean().over(cells)).alias("ydm"))
    top = s.filter(pl.col("q") == 4)
    bot = s.filter(pl.col("q") == 0)
    n5, n1 = top.height, bot.height
    lift = float(top["ydm"].mean() - bot["ydm"].mean())
    psi = pl.concat([top.select("own", (pl.col("ydm") / n5).alias("psi")),
                     bot.select("own", (-pl.col("ydm") / n1).alias("psi"))])
    gs = psi.group_by("own").agg(pl.col("psi").sum())
    se = float((gs["psi"].to_numpy() ** 2).sum() ** 0.5)
    return {"lift": lift, "se": se, "t": lift / se if se > 0 else None,
            "n": int(s.height), "n_owners": int(s["own"].n_unique())}


def pair_scores_for_class(cls, model, vec):
    """Return the scored frame with L_pair/A_pair (calendar buckets), the
    per-filing theta top-2, and for 009 also monthly-bucket per-filing
    anchored scores. None if the class cannot be built."""
    tp = PROC / f"tm_class{cls}.parquet"
    sp = PROC / f"rolling_surprise_class{cls}.parquet"
    if not (tp.exists() and sp.exists()):
        return None
    tm = pl.read_parquet(tp, columns=["serial_number", "filing_date", "registration_date",
                                      "owner_name", "goods_services"]).filter(
        pl.col("goods_services").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
    ).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
        pl.col("owner_name").fill_null("").str.to_uppercase()
          .str.replace_all(r"[^A-Z0-9 ]", "").str.replace_all(r"\s+", " ")
          .str.strip_chars().alias("own"))
    sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                      "topic_kl_vs_future", "topic_dkl"]).filter(
        pl.col("topic_dkl").is_finite())
    d = tm.join(sc, on="serial_number", how="inner").unique("serial_number").sort("fy")
    del tm, sc
    gc.collect()
    n = d.height
    if n < 20_000:
        return None
    texts = d["goods_services"].to_list()
    theta = np.empty((n, T), dtype=np.float32)
    for s in range(0, n, 200_000):
        theta[s:s + 200_000] = model["lda"].transform(vec.transform(texts[s:s + 200_000]))
    del texts
    d = d.drop("goods_services")
    present = theta >= THRESH
    order = np.argsort(-theta, axis=1)
    a, b = order[:, 0].astype(np.int16), order[:, 1].astype(np.int16)
    del theta, order
    gc.collect()

    years = d["fy"].to_numpy()
    ys = np.arange(years.min(), years.max() + 1)
    Pf = present.astype(np.float32)
    marg, co, cnt = {}, {}, {}
    for y in ys:
        mk = years == y
        cnt[y] = int(mk.sum())
        marg[y] = Pf[mk].sum(axis=0)
        co[y] = Pf[mk].T @ Pf[mk]

    def window(y, lo, hi):
        yy = [v for v in range(y + lo, y + hi + 1) if v in cnt]
        N = sum(cnt[v] for v in yy)
        if N < FLOOR:
            return None
        return (sum(marg[v] for v in yy) / N, sum(co[v] for v in yy) / N)

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

    monthly = None
    if cls == "009":
        # monthly buckets for Stage C, computed before Pf is discarded
        fd = d["fd"].to_numpy()
        month_idx = (years - years.min()) * 12 + np.array(
            [dt.month - 1 for dt in d["fd"].to_list()], dtype=np.int32)
        nm = month_idx.max() + 1
        mcnt = np.zeros(nm, dtype=np.int64)
        mmarg = np.zeros((nm, T), dtype=np.float64)
        mco = np.zeros((nm, T, T), dtype=np.float64)
        for mi in range(nm):
            mk = month_idx == mi
            k = int(mk.sum())
            if k == 0:
                continue
            mcnt[mi] = k
            mmarg[mi] = Pf[mk].sum(axis=0)
            mco[mi] = Pf[mk].T @ Pf[mk]
        ccnt = np.concatenate([[0], np.cumsum(mcnt)])
        cmarg = np.concatenate([np.zeros((1, T)), np.cumsum(mmarg, axis=0)])
        cco = np.concatenate([np.zeros((1, T, T)), np.cumsum(mco, axis=0)])
        WM = 60  # ~1826 days in months
        Sm2 = np.full(n, np.nan); Sp2 = np.full(n, np.nan)
        for i in range(n):
            mi = month_idx[i]
            lo_p, hi_p = max(0, mi - WM), mi          # [mi-60, mi)
            lo_f, hi_f = mi + 1, min(nm, mi + 1 + WM)  # (mi, mi+60]
            Np = ccnt[hi_p] - ccnt[lo_p]
            Nf = ccnt[hi_f] - ccnt[lo_f]
            if Np < FLOOR or Nf < FLOOR:
                continue
            for S, lo, hi, N in ((Sm2, lo_p, hi_p, Np), (Sp2, lo_f, hi_f, Nf)):
                M = (cmarg[hi] - cmarg[lo]) / N
                C = (cco[hi] - cco[lo]) / N
                pa, pb = M[a[i]], M[b[i]]
                pab = C[a[i], b[i]]
                S[i] = -np.log((pab + eps) / (pa * pb + eps))
        monthly = (Sm2, Sp2)
        del mmarg, mco, cmarg, cco
    del Pf
    gc.collect()

    d = d.with_columns(pl.Series("S_minus", Sm), pl.Series("S_plus", Sp),
                       pl.Series("theme_a", a))
    if monthly is not None:
        d = d.with_columns(pl.Series("S_minus_pf", monthly[0]),
                           pl.Series("S_plus_pf", monthly[1]))
    d = d.filter(pl.col("S_minus").is_finite() & pl.col("S_plus").is_finite()).with_columns(
        (pl.col("S_minus") - pl.col("S_plus")).alias("L_pair"),
        ((pl.col("S_minus") + pl.col("S_plus")) / 2).alias("A_pair"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"),
        pl.col("topic_dkl").alias("L"))
    return d


def gate_frame(d, ev):
    g = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd").with_columns(pl.col("rd").dt.year().alias("ry")).filter(
        pl.col("ry").is_between(REG_LO, REG_HI)
    ).join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
                   .fill_null(False).cast(pl.Float64).alias("failed"))
    return g


def residualize(g):
    X = np.column_stack([np.ones(g.height), g["L"].to_numpy(), g["A"].to_numpy()])
    def r(v):
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        return v - X @ beta
    return g.with_columns(pl.Series("L_pair_res", r(g["L_pair"].to_numpy())),
                          pl.Series("A_pair_res", r(g["A_pair"].to_numpy())))


def main() -> int:
    model = joblib.load(PROC / "topic_model.joblib")
    vec = CountVectorizer(vocabulary=model["vocabulary"], lowercase=True,
                          token_pattern=TOKEN, ngram_range=(1, 2))
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())

    out = {"construction": "theme_pair_surprise.py, corpus-wide",
           "per_class": {}, "diagnostics": {}}
    slims = []
    stagec = {}
    for cls in CLASSES:
        try:
            d = pair_scores_for_class(cls, model, vec)
        except Exception as e:
            log(f"[{cls}] ERROR {type(e).__name__}: {e}")
            out["per_class"][cls] = {"error": str(e)}
            continue
        if d is None:
            log(f"[{cls}] skipped (missing files or too small)")
            out["per_class"][cls] = {"skipped": True}
            continue
        g = gate_frame(d, ev)
        if g.height < 500:
            log(f"[{cls}] {d.height:,} scored, {g.height:,} gate rows -- below floor")
            out["per_class"][cls] = {"n_scored": int(d.height), "n_gate": int(g.height),
                                     "skipped": True}
            del d, g
            gc.collect()
            continue
        g = residualize(g)
        raw = contrast_within(g, "L_pair", ["ry"])
        net = contrast_within(g, "L_pair_res", ["ry"])
        out["per_class"][cls] = {
            "n_scored": int(d.height), "n_gate": int(g.height),
            "base": float(g["failed"].mean()),
            "raw": raw, "net": net,
            "corr_Lpair_L": float(g.select(pl.corr("L_pair", "L")).item()),
        }
        log(f"[{cls}] gate n={g.height:,} raw {100*raw['lift']:+.2f} (t {raw['t']:+.1f})  "
            f"net {100*net['lift']:+.2f} (t {net['t']:+.1f})" if raw and net
            else f"[{cls}] gate n={g.height:,} contrast unavailable")
        keep_cols = ["serial_number", "own", "ry", "fy", "failed", "L", "A",
                     "L_pair", "A_pair", "L_pair_res", "A_pair_res", "theme_a"]
        slims.append(g.select([c for c in keep_cols if c in g.columns])
                     .with_columns(pl.lit(cls).alias("cls")))
        if cls == "009":
            gg = g.join(d.select("serial_number", "S_minus_pf", "S_plus_pf"),
                        on="serial_number", how="left").filter(
                pl.col("S_minus_pf").is_finite() & pl.col("S_plus_pf").is_finite()
            ).with_columns((pl.col("S_minus_pf") - pl.col("S_plus_pf")).alias("L_pair_pf"))
            both = gg.height
            c_cal = contrast_within(gg, "L_pair", ["ry"])
            c_pf = contrast_within(gg, "L_pair_pf", ["ry"])
            r = float(gg.select(pl.corr("L_pair", "L_pair_pf")).item())
            stagec = {"n_common": int(both), "corr_cal_pf": r,
                      "calendar": c_cal, "per_filing_monthly": c_pf}
            log(f"[stageC 009] n={both:,} corr {r:.3f}  cal {100*c_cal['lift']:+.2f} "
                f"pf {100*c_pf['lift']:+.2f}")
            del gg
        del d, g
        gc.collect()

    pooled = pl.concat(slims, how="diagonal")
    del slims
    gc.collect()
    log(f"[pooled] {pooled.height:,} gate rows, {pooled['cls'].n_unique()} classes")
    out["pooled"] = {
        "n": int(pooled.height),
        "raw": clustered_contrast(pooled, "L_pair", ["cls", "ry"]),
        "net": clustered_contrast(pooled, "L_pair_res", ["cls", "ry"]),
        "A_pair_net": clustered_contrast(pooled, "A_pair_res", ["cls", "ry"]),
    }
    for k in ("raw", "net", "A_pair_net"):
        v = out["pooled"][k]
        log(f"[pooled {k}] {100*v['lift']:+.2f}pp (clustered SE {100*v['se']:.2f}, "
            f"t {v['t']:+.1f}, owners {v['n_owners']:,})")

    # Stage B: 009 and 035 diagnostics
    for cls in ("009", "035"):
        sub = pooled.filter(pl.col("cls") == cls)
        if sub.height == 0:
            continue
        diag = {"eras": {}, "halves": {}}
        for lab, lo, hi in ERAS:
            e = sub.filter(pl.col("fy").is_between(lo, hi))
            if e.height >= 2000:
                diag["eras"][lab] = {"raw": contrast_within(e, "L_pair", ["ry"]),
                                     "net": contrast_within(e, "L_pair_res", ["ry"]),
                                     "n": int(e.height)}
        wt = contrast_within(sub, "L_pair", ["theme_a", "ry"])
        wtn = contrast_within(sub, "L_pair_res", ["theme_a", "ry"])
        diag["within_theme"] = {"raw": wt, "net": wtn}
        rng = np.random.default_rng(SEED)
        half = rng.random(sub.height) < 0.5
        for name, mask in (("half1", half), ("half2", ~half)):
            h = sub.filter(pl.Series(mask))
            diag["halves"][name] = contrast_within(h, "L_pair_res", ["ry"])
        out["diagnostics"][cls] = diag
        e_str = {k: (round(100 * v["net"]["lift"], 2) if v.get("net") else None)
                 for k, v in diag["eras"].items()}
        log(f"[diag {cls}] net by era {e_str}  within-theme net "
            f"{100*wtn['lift']:+.2f}" if wtn else f"[diag {cls}] eras {e_str}")

    out["stage_c_009"] = stagec if stagec else {"not_run": "009 missing"}
    (RES / "pair_lead_corpus.json").write_text(json.dumps(out, indent=1))

    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Class & $n$ & Raw Q5$-$Q1 (pp) & Net of $L$, $A$ (pp) & SE \\",
            r"\midrule"]
    for cls in CLASSES:
        pc = out["per_class"].get(cls, {})
        if not pc.get("raw") or pc.get("n_gate", 0) < PER_CLASS_MIN_GATE:
            continue
        rows.append(f"{cls} & {pc['n_gate']:,} & ${100*pc['raw']['lift']:+.2f}$ & "
                    f"${100*pc['net']['lift']:+.2f}$ & {100*pc['net']['se']:.2f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (RES / "pair_lead_corpus.tex").write_text("\n".join(rows) + "\n")
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
