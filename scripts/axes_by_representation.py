"""Atypicality and lead, under term and theme scoring, against both outcomes.

Two sentences in the abstract need numbers that do not yet exist. The first
asserts that atypicality is uncorrelated with patenting within the firm; what
has been measured is LEAD against patenting, which is a different quantity. The
second wants a clean three-way statement about what term scoring, theme scoring
and their disagreement each do -- again on atypicality, where the existing
evidence is on lead.

So this computes the full grid rather than the corner already known: both axes,
under both representations, against both firm-level outcomes.

  SEC reporting   registered debut owners, one row each, class x filing-year
                  absorbed, description length and counsel of record held.
  patenting       firm-year panel matched to PatentsView, firm fixed effects,
                  errors clustered on firm, on log(1 + patents).

Everything is standardized, so coefficients are comparable across axes and
representations, which is the only way "stronger" or "uncorrelated" can be
given a meaning.

Output: paper/results/axes_by_representation.json
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
PUB = REPO / "data_publish"
RES = REPO / "paper" / "results"

CLASSES = [f"{i:03d}" for i in range(1, 46)]
FILE_LO, FILE_HI = 1995, 2018
SRCS = {"theme": "rolling", "term": "termroll"}


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def ols_hc1(X, y):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    r = y - X @ b
    n, k = X.shape
    V = XtXi @ ((X * (r ** 2)[:, None]).T @ X) @ XtXi * (n / max(n - k, 1))
    return b, np.sqrt(np.diag(V))


def cluster_fe(d: pl.DataFrame, yc: str, xc: str, gc_: str):
    """Within-group demeaned OLS with errors clustered on the group."""
    d = d.with_columns([(pl.col(c) - pl.col(c).mean().over(gc_)).alias(c)
                        for c in (yc, xc)])
    x = d[xc].to_numpy()
    y = d[yc].to_numpy()
    sxx = float(x @ x)
    if sxx <= 0:
        return None
    b = float(x @ y) / sxx
    r = y - b * x
    ids = d[gc_].to_numpy()
    o = np.argsort(ids, kind="stable")
    xs, rs, idss = x[o], r[o], ids[o]
    bd = np.flatnonzero(np.r_[True, idss[1:] != idss[:-1], True])
    meat = sum((float(xs[a:z] @ rs[a:z])) ** 2 for a, z in zip(bd[:-1], bd[1:]))
    ng = len(bd) - 1
    se = float(np.sqrt(meat) / sxx) * np.sqrt(ng / max(ng - 1, 1))
    return {"coef": b, "se": se, "t": b / se if se else None,
            "n": int(d.height), "n_firms": int(ng)}


def load(src: str, cols_extra: list[str]) -> pl.DataFrame:
    parts = []
    for c in CLASSES:
        sp = PROC / f"{src}_surprise_class{c}.parquet"
        tp = PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number"] + cols_extra)
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite()
            & pl.col("topic_kl_vs_future").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.lit(c).alias("cls")))
        del tm, sc
        gc.collect()
    d = pl.concat(parts)
    del parts
    gc.collect()
    return d.with_columns(
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("A"),
        (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"))


def main() -> int:
    out = {"sec": {}, "patents": {}}

    # ---------- patenting, both axes, both representations ----------------
    panel = pl.read_csv(PUB / "firm_year_patents_and_dkl.csv").select(
        pl.col("uspto_owner_name").alias("owner_name"),
        pl.col("normalized_name").alias("firm"), "year", "n_patents")
    log(f"[patents] panel {panel.height:,} firm-years")
    for rep, src in SRCS.items():
        d = load(src, ["filing_date", "owner_name"])
        d = d.with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
        ).filter(pl.col("owner_name").is_not_null())
        fy = d.group_by(["owner_name", "year"]).agg(
            pl.col("A").mean().alias("A"), pl.col("L").mean().alias("L"),
            pl.len().alias("k"))
        del d
        gc.collect()
        j = panel.join(fy, on=["owner_name", "year"], how="inner").group_by(
            ["firm", "year"]).agg(
            ((pl.col("A") * pl.col("k")).sum() / pl.col("k").sum()).alias("A"),
            ((pl.col("L") * pl.col("k")).sum() / pl.col("k").sum()).alias("L"),
            pl.col("n_patents").max().alias("n_patents"))
        j = j.with_columns(
            ((pl.col("n_patents").cast(pl.Float64) + 1.0).log()).alias("x"),
            ((pl.col("A") - pl.col("A").mean()) / pl.col("A").std()).alias("Az"),
            ((pl.col("L") - pl.col("L").mean()) / pl.col("L").std()).alias("Lz"))
        j = j.with_columns(pl.col("x").std().over("firm").alias("sdx"),
                           pl.len().over("firm").alias("kk")
                           ).filter((pl.col("kk") >= 2) & (pl.col("sdx") > 0))
        for axis, col in (("atypicality", "Az"), ("lead", "Lz")):
            r = cluster_fe(j, col, "x", "firm")
            out["patents"][f"{rep}_{axis}"] = r
            if r:
                log(f"  {rep:>5} {axis:<12} beta {r['coef']:+.4f} sd "
                    f"(t {r['t']:+5.1f}, {r['n_firms']:,} firms)")
        del fy, j
        gc.collect()

    # ---------- SEC reporting, both axes, both representations ------------
    cw = set(pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")["owner_name"].to_list())
    att = pl.read_parquet(PROC / "case_extras.parquet",
                          columns=["serial_number", "attorney_name"]).with_columns(
        (pl.col("attorney_name").is_not_null()
         & (pl.col("attorney_name").str.strip_chars().str.len_chars() > 1))
        .cast(pl.Float64).alias("counsel")).select("serial_number", "counsel")

    frames = {}
    for rep, src in SRCS.items():
        d = load(src, ["filing_date", "registration_date", "owner_name",
                       "goods_services"])
        d = d.filter(pl.col("owner_name").is_not_null()
                     & pl.col("goods_services").is_not_null()
                     & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        d = d.with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy")
        ).filter(pl.col("fy").is_between(FILE_LO, FILE_HI))
        d = d.with_columns(pl.col("filing_date").min().over("owner_name").alias("d0")
                           ).filter(pl.col("filing_date") == pl.col("d0")
                                    ).unique(subset="owner_name")
        d = d.filter(pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        frames[rep] = d.select(
            "serial_number", "owner_name", "cls", "fy",
            pl.col("A").alias(f"A_{rep}"), pl.col("L").alias(f"L_{rep}"),
            pl.col("goods_services").str.len_chars().log().alias("loglen"))
        del d
        gc.collect()

    m = frames["theme"].join(
        frames["term"].select("owner_name", "A_term", "L_term"),
        on="owner_name", how="inner").join(att, on="serial_number", how="left")
    del frames, att
    gc.collect()
    m = m.with_columns(
        pl.col("counsel").fill_null(0.0),
        pl.col("owner_name").is_in(list(cw)).cast(pl.Float64).alias("y"),
        pl.concat_str([pl.col("cls"), pl.col("fy").cast(pl.Utf8)],
                      separator="-").alias("cell"))
    cols = ["A_theme", "L_theme", "A_term", "L_term", "loglen", "counsel"]
    m = m.with_columns([((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
                        for c in cols[:5]])
    m = m.with_columns([(pl.col(c) - pl.col(c).mean().over("cell")).alias(c)
                        for c in cols + ["y"]])
    log(f"\n[sec] {m.height:,} owners scored under both representations")
    y = m["y"].to_numpy()
    C = {c: m[c].to_numpy() for c in cols}
    specs = [("theme A alone", ["A_theme"]), ("term A alone", ["A_term"]),
             ("both A", ["A_theme", "A_term"]),
             ("both A + counsel + length", ["A_theme", "A_term", "counsel", "loglen"]),
             ("both L + counsel + length", ["L_theme", "L_term", "counsel", "loglen"]),
             ("all four + counsel + length", cols)]
    for name, ks in specs:
        b, se = ols_hc1(np.column_stack([C[k] for k in ks]), y)
        out["sec"][name] = {k: {"b_pp": float(100 * b[i]), "t": float(b[i] / se[i])}
                            for i, k in enumerate(ks)}
        log("  " + name.ljust(28) + " ".join(
            f"{k}={100*b[i]:+.4f}(t{b[i]/se[i]:+.1f})" for i, k in enumerate(ks)))

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "axes_by_representation.json").write_text(json.dumps(out, indent=1))
    log("\n[done] axes_by_representation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
