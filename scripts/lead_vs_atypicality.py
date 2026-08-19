"""Does lead carry signal beyond atypicality, or only borrow it?

The worry is a ceiling. Lead is L = K- minus K+ and atypicality is
A = (K- plus K+)/2, and both KL levels are non-negative, so

    |L| = |K- - K+| <= K- + K+ = 2A

is not an empirical regularity but an identity of the construction: a filing
cannot be strongly leading or strongly lagging unless it is also atypical. If
financial success tracks atypicality alone, the U-shape in lead follows for
free, because the tails of L are populated by high-A filings and the middle by
low-A ones.

Four things, in increasing order of how much they settle it.

  1. How binding the ceiling is -- the share of the |L| <= 2A bound actually
     used, and the largest |L| observed in each decile of A.
  2. A horse race: the outcome on standardized A, signed L and |L| together,
     with class x filing-year absorbed and description length held. Standardized
     so the coefficients are comparable, which is what "stronger" has to mean.
  3. Incremental R-squared from adding each regressor to the others, which is
     the part of the variance only that regressor can explain.
  4. The decisive one: the signed-L contrast computed WITHIN narrow bands of A.
     If lead only proxies atypicality, it has nothing left to do once A is held
     nearly fixed, and the within-band contrasts collapse to zero.

Sample and outcome follow Section "Being early costs at the public-reporting
margin": registered debut owners, one row per owner, outcome is whether the
owner ever appears in SEC financial-reporting records.

Output: paper/results/lead_vs_atypicality.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
FILE_LO, FILE_HI = 1995, 2018
NBAND = 10


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def demean(df: pl.DataFrame, cols: list[str], by: str) -> pl.DataFrame:
    return df.with_columns([(pl.col(c) - pl.col(c).mean().over(by)).alias(c)
                            for c in cols])


def ols_hc1(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with heteroskedasticity-robust (HC1) standard errors and R^2."""
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    b = XtXi @ (X.T @ y)
    r = y - X @ b
    n, k = X.shape
    meat = (X * (r ** 2)[:, None]).T @ X
    V = XtXi @ meat @ XtXi * (n / max(n - k, 1))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(r @ r) / ss if ss > 0 else float("nan")
    return b, np.sqrt(np.diag(V)), r2


def main() -> int:
    cw = set(pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")["owner_name"].to_list())
    parts = []
    for c in CLASSES:
        sp, tp = PROC / f"{SRC}_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                          "registration_date", "owner_name",
                                          "goods_services"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("goods_services").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite()
            & pl.col("topic_kl_vs_future").is_finite())
        j = tm.join(sc, on="serial_number", how="inner")
        parts.append(j.select(
            "serial_number", "owner_name", "filing_date", "registration_date",
            "topic_kl_vs_past", "topic_kl_vs_future",
            pl.col("goods_services").str.len_chars().alias("glen"),
            pl.lit(c).alias("cls")))
        del tm, sc, j
        gc.collect()
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()

    d = d.with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("A"),
        (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"),
    ).filter(pl.col("fy").is_between(FILE_LO, FILE_HI))

    # ---- 1. how binding is |L| <= 2A? ------------------------------------
    A = d["A"].to_numpy()
    L = d["L"].to_numpy()
    ratio = np.abs(L) / (2.0 * A)
    dec = np.minimum((np.argsort(np.argsort(A)) * 10 // len(A)), 9)
    ceiling = {"max_ratio": float(ratio.max()),
               "mean_ratio": float(ratio.mean()),
               "p99_ratio": float(np.percentile(ratio, 99)),
               "by_A_decile": []}
    log("[ceiling] |L| <= 2A holds by construction; how much of it is used")
    log(f"  {'A decile':>9} {'mean A':>8} {'max |L|':>9} {'2A at dec. mean':>16} {'max ratio':>10}")
    for k in range(10):
        m = dec == k
        row = {"decile": k + 1, "mean_A": float(A[m].mean()),
               "max_absL": float(np.abs(L[m]).max()),
               "max_ratio": float(ratio[m].max()),
               "sd_L": float(L[m].std())}
        ceiling["by_A_decile"].append(row)
        log(f"  {k+1:>9} {row['mean_A']:>8.3f} {row['max_absL']:>9.3f} "
            f"{2*row['mean_A']:>16.3f} {row['max_ratio']:>10.3f}")
    log(f"  overall: mean |L|/2A = {ratio.mean():.4f}, 99th pct = "
        f"{np.percentile(ratio,99):.3f}, max = {ratio.max():.3f}")

    # ---- debut owners, one row each --------------------------------------
    deb = d.with_columns(
        pl.col("filing_date").min().over("owner_name").alias("d0")
    ).filter(pl.col("filing_date") == pl.col("d0")).unique(subset="owner_name")
    deb = deb.filter(pl.col("registration_date").fill_null("").str.len_chars() >= 8)
    deb = deb.with_columns(
        pl.col("owner_name").is_in(list(cw)).cast(pl.Float64).alias("y"),
        pl.concat_str([pl.col("cls"), pl.col("fy").cast(pl.Utf8)],
                      separator="-").alias("cell"),
        pl.col("glen").log().alias("loglen"),
        pl.col("L").abs().alias("absL"))
    log(f"\n[sample] {deb.height:,} registered debut owners, "
        f"base rate {100*deb['y'].mean():.3f}%")

    z = lambda s: (s - s.mean()) / s.std()
    deb = deb.with_columns([z(pl.col(c)).alias(c) for c in ("A", "L", "absL", "loglen")])
    deb = demean(deb, ["A", "L", "absL", "loglen", "y"], "cell")

    y = deb["y"].to_numpy()
    cols = {c: deb[c].to_numpy() for c in ("A", "L", "absL", "loglen")}
    out = {"n": int(deb.height), "base": float(deb["y"].mean() + 0),
           "ceiling": ceiling, "models": {}, "increments": {}, "within_A": []}

    specs = [("A only", ["A"]), ("L only", ["L"]), ("|L| only", ["absL"]),
             ("A + L", ["A", "L"]), ("A + |L|", ["A", "absL"]),
             ("A + L + |L|", ["A", "L", "absL"]),
             ("A + L + |L| + length", ["A", "L", "absL", "loglen"])]
    log("\n[horse race] outcome in pp per standard deviation, "
        "class x filing-year absorbed")
    log(f"  {'model':<24} " + "".join(f"{c:>16}" for c in ("A", "L", "|L|", "loglen")) + "     R2x1e4")
    for name, ks in specs:
        X = np.column_stack([cols[k] for k in ks])
        b, se, r2 = ols_hc1(X, y)
        m = {k: {"b_pp": float(100 * b[i]), "se_pp": float(100 * se[i]),
                 "t": float(b[i] / se[i])} for i, k in enumerate(ks)}
        out["models"][name] = {"coefs": m, "r2": r2}
        cells = "".join(
            f"{100*m[k]['b_pp']/100:>+9.4f}({m[k]['t']:>+5.1f})" if k in m else f"{'':>16}"
            for k in ("A", "L", "absL", "loglen"))
        log(f"  {name:<24} {cells}  {1e4*r2:>9.2f}")

    full = ["A", "L", "absL", "loglen"]
    Xf = np.column_stack([cols[k] for k in full])
    _, _, r2f = ols_hc1(Xf, y)
    log("\n[increment] R2 lost when each regressor is dropped from the full model")
    for k in full:
        ks = [c for c in full if c != k]
        _, _, r2 = ols_hc1(np.column_stack([cols[c] for c in ks]), y)
        out["increments"][k] = {"delta_r2": r2f - r2, "delta_r2_x1e4": 1e4 * (r2f - r2)}
        log(f"  {k:<8} drop -> R2 falls by {1e4*(r2f-r2):>7.3f} x 1e-4")

    # ---- 4. signed L within narrow bands of A -----------------------------
    log("\n[within A] signed-lead top-minus-bottom quintile contrast, "
        "inside each decile of atypicality")
    dd = deb.with_columns(
        ((pl.col("A").rank("ordinal") - 1) * NBAND // pl.len()).cast(pl.Int8).alias("ab"))
    log(f"  {'A decile':>9} {'n':>9} {'base':>8} {'Q5-Q1 on L':>14}")
    for k in range(NBAND):
        s = dd.filter(pl.col("ab") == k)
        if s.height < 5000:
            continue
        s = s.sort(["L", "owner_name"]).with_columns(
            ((pl.col("L").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
        g = s.group_by("q").agg(pl.col("y").mean().alias("p"),
                                pl.len().alias("n")).sort("q")
        v = [float(r["p"]) for r in g.iter_rows(named=True)]
        if len(v) < 5:
            continue
        p1, p5 = v[0], v[4]
        n1, n5 = int(g["n"][0]), int(g["n"][4])
        se = ((abs(p1) * (1 - abs(p1)) / n1) + (abs(p5) * (1 - abs(p5)) / n5)) ** 0.5
        row = {"decile": k + 1, "n": int(s.height), "lift_pp": 100 * (p5 - p1),
               "se_pp": 100 * se}
        out["within_A"].append(row)
        log(f"  {k+1:>9} {s.height:>9,} {'':>8} {row['lift_pp']:>+9.4f}pp")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "lead_vs_atypicality.json").write_text(json.dumps(out, indent=1))
    log("\n[done] lead_vs_atypicality.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
