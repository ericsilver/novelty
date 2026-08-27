"""Decompose the W in the signed-lead curve into its pieces, as stacked bars.

Outcome: P(owner ever in SEC EDGAR | registered debut 1995-2018), the
paper's W-shaped signed-lead profile, in 20 pooled ventiles of L.

Per ventile b, the deviation of the bin rate from the base rate is split
into four additive pieces by sequential counterfactual prediction:

  1. atypicality composition: mean over the bin of E[y | A] (50 A-quantile
     bins) minus the base -- the symmetric U inherited from A.
  2. boilerplate notch: mean over the bin of E[y | A, dup] - E[y | A],
     where dup marks exact-duplicate text (dupN > 1) -- the center notch.
  3. lead tilt: beta * (mean_b(L) - mean(L)), beta from OLS of the stage-2
     residual on L -- the antisymmetric part.
  4. residual: total minus the three, including any true curvature in L.

Figure: diverging stacked bars (pieces) with the actual curve overlaid;
second panel: the curve on all filings vs unique-text filings only.

Also computed: the five-year-proof atypicality contrast (Q5-Q1 within
class x registration year) on all registrations 2002-2018, excluding
duplicate-text registrations, and excluding only Manual-scale text
(dupN >= 100); and the lead contrast under the same exclusions.

Outputs: paper/results/fig_L_decomposition.png
         paper/results/L_decomposition.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
FLAGS = RES / "dup_flags"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
FY_LO, FY_HI = 1995, 2018
NB = 20
NA = 50
EDGE = "2026-04-02"


def log(m): print(m, file=sys.stderr, flush=True)


def all_classes():
    return sorted(p.stem.replace("tm_class", "") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class", "").isdigit())


def build_sec_frame():
    parts = []
    for cls in all_classes():
        p = PROC / f"tm_class{cls}.parquet"
        parts.append(pl.read_parquet(p, columns=["owner_name", "filing_date"]).filter(
            pl.col("owner_name").is_not_null() & (pl.col("filing_date").str.len_chars() == 8)
        ).group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date")))
    debut = pl.concat(parts).group_by("owner_name").agg(pl.col("debut_date").min().alias("debut_date"))
    del parts; gc.collect()
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name").unique().with_columns(
        pl.lit(True).alias("in_sec"))
    parts = []
    for cls in all_classes():
        tp = PROC / f"rolling_surprise_class{cls}.parquet"
        fp = FLAGS / f"{cls}.parquet"
        if not (tp.exists() and fp.exists()):
            continue
        topic = pl.read_parquet(tp).filter(pl.col("topic_dkl").is_finite()
                                           & pl.col("year").is_between(FY_LO, FY_HI))
        tok = pl.read_parquet(PROC / f"surprise_class{cls}.parquet",
                              columns=["serial_number", "n_terms"]).filter(pl.col("n_terms") >= 3)
        t = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                            columns=["serial_number", "owner_name", "filing_date", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8
        ).join(debut, on="owner_name", how="left").filter(pl.col("filing_date") == pl.col("debut_date"))
        fl = pl.read_parquet(fp)
        j = topic.join(tok, on="serial_number", how="inner").join(t, on="serial_number", how="inner").join(
            sec, on="owner_name", how="left").join(fl, on="serial_number", how="left").with_columns(
            pl.col("in_sec").fill_null(False).cast(pl.Float64),
            pl.col("dupN").fill_null(1)).select(
            "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl", "in_sec", "dupN")
        if j.height:
            parts.append(j)
        del topic, tok, t, fl, j
        gc.collect()
    d = pl.concat(parts).with_columns(
        pl.col("topic_dkl").alias("L"),
        ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"),
        (pl.col("dupN") > 1).alias("dup"))
    del parts; gc.collect()
    return d


def decompose(d, out):
    y = d["in_sec"].to_numpy()
    L = d["L"].to_numpy()
    A = d["A"].to_numpy()
    dup = d["dup"].to_numpy()
    base = y.mean()
    # L ventiles, pooled cuts (as in the paper's figure)
    lcuts = np.quantile(L, np.linspace(0, 1, NB + 1))
    lbin = np.clip(np.searchsorted(lcuts[1:-1], L), 0, NB - 1)
    # stage 1: E[y | A] over NA quantile bins of A
    acuts = np.quantile(A, np.linspace(0, 1, NA + 1))
    abin = np.clip(np.searchsorted(acuts[1:-1], A), 0, NA - 1)
    yA = np.zeros(NA)
    for k in range(NA):
        yA[k] = y[abin == k].mean()
    pred1 = yA[abin]
    # stage 2: E[y | A, dup]
    pred2 = pred1.copy()
    for k in range(NA):
        for dv in (False, True):
            m = (abin == k) & (dup == dv)
            if m.sum() >= 200:
                pred2[m] = y[m].mean()
    # stage 3: linear tilt of the residual on L
    r2 = y - pred2
    beta = float(np.cov(r2, L)[0, 1] / np.var(L))
    log(f"[beta] residual lead slope {100*beta:+.3f} pp per nat")
    rows = []
    for b in range(NB):
        m = lbin == b
        tot = y[m].mean() - base
        c1 = pred1[m].mean() - base
        c2 = (pred2[m] - pred1[m]).mean()
        c3 = beta * (L[m].mean() - L.mean())
        rows.append({"bin": b, "midL": float(L[m].mean()), "n": int(m.sum()),
                     "total": float(tot), "atypicality": float(c1),
                     "boilerplate": float(c2), "lead_tilt": float(c3),
                     "residual": float(tot - c1 - c2 - c3),
                     "se_total": float((y[m].mean() * (1 - y[m].mean()) / m.sum()) ** 0.5)})
    out["base"] = float(base)
    out["beta_pp_per_nat"] = 100 * beta
    out["bins"] = rows
    # duplicate-excluded curve on the same cuts
    curves = {}
    for lab, mask in (("all", np.ones_like(dup, bool)), ("unique_text", ~dup)):
        p, mid = [], []
        for b in range(NB):
            m = (lbin == b) & mask
            p.append(float(y[m].mean()))
            mid.append(float(L[m].mean()))
        curves[lab] = {"mid": mid, "p": p}
    out["curves"] = curves
    return rows, curves, base


def gate_contrasts(out):
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    parts = []
    for c in CLASSES:
        tp, sp, fp = (PROC / f"tm_class{c}.parquet", PROC / f"rolling_surprise_class{c}.parquet",
                      FLAGS / f"{c}.parquet")
        if not (tp.exists() and sp.exists() and fp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "registration_date"]).filter(
            pl.col("registration_date").fill_null("").str.len_chars() >= 8).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")).drop_nulls("rd")
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future",
                                          "topic_dkl"]).filter(pl.col("topic_dkl").is_finite())
        fl = pl.read_parquet(fp)
        parts.append(tm.join(sc, on="serial_number", how="inner").join(fl, on="serial_number", how="left")
                     .with_columns(pl.lit(c).alias("cls"), pl.col("dupN").fill_null(1)))
        del tm, sc, fl
        gc.collect()
    d = pl.concat(parts).unique("serial_number"); del parts; gc.collect()
    d = d.with_columns(pl.col("rd").dt.year().alias("ry"), pl.col("topic_dkl").alias("L"),
                       ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))
    d = d.filter(pl.col("ry").is_between(2002, 2018)).join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")).with_columns(
        ((pl.col("age") >= 4.0) & (pl.col("age") < 8.5)).fill_null(False).cast(pl.Float64).alias("fail"))
    out["gate"] = {}
    for lab, sub in (("all", d), ("unique_text", d.filter(pl.col("dupN") == 1)),
                     ("excl_manual_scale", d.filter(pl.col("dupN") < 100))):
        s = sub
        for var in ("A", "L"):
            s = s.with_columns(((pl.col(var).rank("ordinal").over(["cls", "ry"]) - 1) * 5
                                // pl.len().over(["cls", "ry"])).cast(pl.Int8).alias(f"q{var}"))
        rec = {"n": int(s.height)}
        for var in ("A", "L"):
            g = s.group_by(f"q{var}").agg(pl.col("fail").mean().alias("p"), pl.len().alias("n")).sort(f"q{var}")
            p = [float(v) for v in g["p"]]; n = [int(v) for v in g["n"]]
            rec[f"{var}_q5_q1_pp"] = 100 * (p[4] - p[0])
            rec[f"{var}_se_pp"] = 100 * ((p[0] * (1 - p[0]) / n[0]) + (p[4] * (1 - p[4]) / n[4])) ** 0.5
        out["gate"][lab] = rec
        log(f"[gate|{lab}] " + json.dumps(rec))
    return out


def main() -> int:
    d = build_sec_frame()
    log(f"[sec frame] {d.height:,} registered debuts, dup share {float(d['dup'].mean()):.3f}")
    out = {}
    rows, curves, base = decompose(d, out)
    del d
    gc.collect()
    gate_contrasts(out)
    (RES / "L_decomposition.json").write_text(json.dumps(out, indent=1))

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    xs = np.arange(NB)
    comps = [("atypicality", "#2b6cb0", "atypicality composition"),
             ("boilerplate", "#e67e22", "duplicate-text notch"),
             ("lead_tilt", "#c0392b", "lead tilt"),
             ("residual", "#bbbbbb", "residual")]
    pos = np.zeros(NB); neg = np.zeros(NB)
    for key, color, label in comps:
        v = np.array([100 * r[key] for r in rows])
        bottom = np.where(v >= 0, pos, neg)
        axA.bar(xs, v, bottom=bottom, width=0.8, color=color, label=label,
                edgecolor="white", linewidth=0.3)
        pos = np.where(v >= 0, pos + v, pos)
        neg = np.where(v < 0, neg + v, neg)
    tot = [100 * r["total"] for r in rows]
    se = [196 * r["se_total"] for r in rows]
    axA.errorbar(xs, tot, yerr=se, fmt="ko", ms=4, lw=1, capsize=2,
                 label="actual bin rate $-$ base", zorder=5)
    axA.axhline(0, color="#333", lw=1)
    axA.set_xticks(xs[::2])
    axA.set_xticklabels([f"{rows[i]['midL']:+.2f}" for i in range(0, NB, 2)], fontsize=7.5)
    axA.set_xlabel("signed lead $L$ (ventile means, nats)")
    axA.set_ylabel("deviation from the 0.65% base rate (pp)")
    axA.legend(fontsize=8.5, frameon=False, loc="upper center")
    axA.grid(alpha=0.25, axis="y")

    for lab, color, name in (("all", "#2b6cb0", "all filings"),
                             ("unique_text", "#c0392b", "unique-text filings only")):
        cv = out["curves"][lab]
        axB.plot(cv["mid"], [100 * v for v in cv["p"]], "o-", color=color, lw=1.8,
                 ms=3.5, label=name)
    axB.axhline(100 * base, ls="--", color="#555", lw=1)
    axB.set_xlabel("signed lead $L$ (nats)")
    axB.set_ylabel("% of owners ever in SEC EDGAR")
    axB.legend(fontsize=8.5, frameon=False, loc="upper center")
    axB.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RES / "fig_L_decomposition.png", dpi=150, bbox_inches="tight")
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
