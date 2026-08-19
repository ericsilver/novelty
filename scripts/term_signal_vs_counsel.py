"""Is the term-scored firm signal just "represented by counsel"?

Appendix "The measurement hazard" says term-scored lead predicts firm outcomes
and that the effect is lexical specificity rather than position: serious firms
describe specific products in specific, therefore rare, words, and rare-term
mass inflates a term-level divergence mechanically. That phrasing is opaque, and
it invites a much simpler reading -- that professionally drafted applications
both use richer language and belong to firms likelier to reach public markets,
so the whole thing is a proxy for having hired a trademark attorney.

This tests the simpler reading directly. First the raw association between
having counsel of record and the owner ever reaching SEC reporting. Then a
regression of the outcome on term-scored lead with counsel entered, and with
description length entered, and with both, each inside class x filing-year. If
the term-scored signal is counsel wearing a disguise, it dies when counsel is
controlled. If it dies only when length is controlled, the specificity reading
is right. If it survives both, neither explanation is sufficient and the
appendix needs rewriting.

The same regressions are run on topic-scored lead for comparison, since the
appendix's claim is about the contrast between the two representations.

Output: paper/results/term_signal_vs_counsel.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

CLASSES = [f"{i:03d}" for i in range(1, 46)]
FILE_LO, FILE_HI = 1995, 2018


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def ols_hc1(X: np.ndarray, y: np.ndarray):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    r = y - X @ b
    n, k = X.shape
    V = XtXi @ ((X * (r ** 2)[:, None]).T @ X) @ XtXi * (n / max(n - k, 1))
    ss = float(np.sum((y - y.mean()) ** 2))
    return b, np.sqrt(np.diag(V)), (1.0 - float(r @ r) / ss if ss > 0 else float("nan"))


def main() -> int:
    cw = set(pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")["owner_name"].to_list())
    att = pl.read_parquet(PROC / "case_extras.parquet",
                          columns=["serial_number", "attorney_name"]).with_columns(
        (pl.col("attorney_name").is_not_null()
         & (pl.col("attorney_name").str.strip_chars().str.len_chars() > 1))
        .cast(pl.Float64).alias("counsel")).select("serial_number", "counsel")
    log(f"[setup] counsel flag on {att.height:,} cases, "
        f"{100*att['counsel'].mean():.1f}% represented")

    parts = []
    for c in CLASSES:
        tp = PROC / f"tm_class{c}.parquet"
        rp = PROC / f"rolling_surprise_class{c}.parquet"
        sp = PROC / f"termroll_surprise_class{c}.parquet"
        if not (tp.exists() and rp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                          "registration_date", "owner_name",
                                          "goods_services"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("goods_services").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        rr = pl.read_parquet(rp, columns=["serial_number", "topic_dkl"]).rename(
            {"topic_dkl": "L_topic"}).filter(pl.col("L_topic").is_finite())
        ss = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).rename(
            {"topic_dkl": "L_term"}).filter(pl.col("L_term").is_finite())
        j = tm.join(rr, on="serial_number", how="inner").join(
            ss, on="serial_number", how="inner")
        parts.append(j.select(
            "serial_number", "owner_name", "filing_date", "registration_date",
            "L_topic", "L_term",
            pl.col("goods_services").str.len_chars().alias("glen"),
            pl.lit(c).alias("cls")))
        del tm, rr, ss, j
        gc.collect()
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()

    d = d.join(att, on="serial_number", how="left").with_columns(
        pl.col("counsel").fill_null(0.0),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
    ).filter(pl.col("fy").is_between(FILE_LO, FILE_HI))
    del att
    gc.collect()

    deb = d.with_columns(
        pl.col("filing_date").min().over("owner_name").alias("d0")
    ).filter(pl.col("filing_date") == pl.col("d0")).unique(subset="owner_name")
    deb = deb.filter(pl.col("registration_date").fill_null("").str.len_chars() >= 8)
    deb = deb.with_columns(
        pl.col("owner_name").is_in(list(cw)).cast(pl.Float64).alias("y"),
        pl.concat_str([pl.col("cls"), pl.col("fy").cast(pl.Utf8)],
                      separator="-").alias("cell"),
        pl.col("glen").log().alias("loglen"))
    del d
    gc.collect()

    # ---- the raw association the question asks for -----------------------
    g = deb.group_by("counsel").agg(pl.col("y").mean().alias("p"),
                                    pl.len().alias("n")).sort("counsel")
    raw = {}
    log(f"\n[raw] owner ever SEC-reporting, by whether the debut filing had counsel")
    for r in g.iter_rows(named=True):
        lab = "counsel" if r["counsel"] == 1.0 else "self-filed"
        raw[lab] = {"n": int(r["n"]), "rate": float(r["p"])}
        log(f"  {lab:<11} {100*r['p']:.3f}%   n = {r['n']:,}")
    if len(raw) == 2:
        raw["ratio"] = raw["counsel"]["rate"] / max(raw["self-filed"]["rate"], 1e-12)
        log(f"  represented owners reach SEC reporting "
            f"{raw['ratio']:.1f}x as often")

    z = lambda s: (s - s.mean()) / s.std()
    deb = deb.with_columns([z(pl.col(c)).alias(c) for c in
                            ("L_topic", "L_term", "loglen")])
    deb = deb.with_columns([(pl.col(c) - pl.col(c).mean().over("cell")).alias(c)
                            for c in ("L_topic", "L_term", "loglen", "counsel", "y")])
    y = deb["y"].to_numpy()
    C = {k: deb[k].to_numpy() for k in ("L_topic", "L_term", "loglen", "counsel")}
    out = {"n": int(deb.height), "raw_counsel": raw, "models": {}}

    log(f"\n[models] n = {deb.height:,}, pp per sd (counsel is a 0/1 dummy), "
        f"class x filing-year absorbed")
    specs = [
        ("term lead alone", ["L_term"]),
        ("term lead + counsel", ["L_term", "counsel"]),
        ("term lead + length", ["L_term", "loglen"]),
        ("term lead + counsel + length", ["L_term", "counsel", "loglen"]),
        ("topic lead alone", ["L_topic"]),
        ("topic lead + counsel + length", ["L_topic", "counsel", "loglen"]),
        ("both leads + counsel + length", ["L_term", "L_topic", "counsel", "loglen"]),
    ]
    for name, ks in specs:
        X = np.column_stack([C[k] for k in ks])
        b, se, r2 = ols_hc1(X, y)
        m = {k: {"b_pp": float(100 * b[i]), "se_pp": float(100 * se[i]),
                 "t": float(b[i] / se[i])} for i, k in enumerate(ks)}
        out["models"][name] = {"coefs": m, "r2": r2}
        cells = " ".join(f"{k}={100*b[i]:+.4f}(t{b[i]/se[i]:+.1f})"
                         for i, k in enumerate(ks))
        log(f"  {name:<30} {cells}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "term_signal_vs_counsel.json").write_text(json.dumps(out, indent=1))
    log("\n[done] term_signal_vs_counsel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
