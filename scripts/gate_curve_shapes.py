"""Is the relationship linear? Fit competing shapes at all three gates.

The paper summarises each outcome with a quintile contrast or a per-sigma
slope, both of which are linear readings. Several of the underlying profiles
are visibly not linear -- registration completion falls at both ends of
atypicality, and SEC reporting is U-shaped in lead -- so a linear summary can
understate the relationship, or point the wrong way, without looking wrong.

Three gates, each a different question about the same filing:

  registration   did the application become a registration at all
  survival       did the registration survive its first use-proof declaration
  commercial     did the owner ever appear in SEC records

and, for the commercial gate, three definitions of the outcome, because
"appears in SEC records" mixes companies that went public with entities that
merely registered securities:

  listed          the owner's CIK carries a ticker in SEC's company-ticker file
  reporting       files financial statements, no ticker
  registered_only in the crosswalk with neither

Four shapes are fitted to each profile by weighted least squares on
equal-count bins, weighting each bin by its size:

  linear        a + b x
  quadratic     a + b x + c x^2
  exponential   a + b exp(c x)              accelerating or saturating, monotone
  symmetric     a + b exp(c |x - x0|)       a V or U with a free vertex

The last is the one worth having: a U-shaped profile is not a failure of the
linear fit to be steep enough, it is a claim that both tails differ from the
middle in the same direction, and only a form with a free vertex can say where
the middle is. Selection is by AIC on the binned means, and the linear fit is
reported alongside whatever wins so the cost of the linear summary is visible.

Output: paper/results/gate_curve_shapes.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RAW = REPO / "data" / "raw" / "sec"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
CLASSES = [f"{i:03d}" for i in range(1, 46)]
FILE_LO, FILE_HI = 1995, 2018
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
NBINS = 40


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
def f_lin(x, a, b):
    return a + b * x


def f_quad(x, a, b, c):
    return a + b * x + c * x * x


def f_exp(x, a, b, c):
    return a + b * np.exp(np.clip(c * x, -50, 50))


def f_sym(x, a, b, c, x0):
    return a + b * np.exp(np.clip(c * np.abs(x - x0), -50, 50))


SHAPES = [
    ("linear", f_lin, lambda x, y: [y.mean(), 0.0]),
    ("quadratic", f_quad, lambda x, y: [y.mean(), 0.0, 0.0]),
    ("exponential", f_exp, lambda x, y: [y.min(), float(y.max() - y.min()) or 1e-6, 0.5]),
    ("symmetric", f_sym, lambda x, y: [y.min(), float(y.max() - y.min()) or 1e-6, 0.5, 0.0]),
]


def fit_shapes(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Weighted fits of every shape; AIC computed on the weighted residuals."""
    out = {}
    sigma = 1.0 / np.sqrt(np.maximum(w, 1.0))
    n = len(x)
    ybar = np.average(y, weights=w)
    sst = float(np.sum(w * (y - ybar) ** 2))
    for name, fn, p0 in SHAPES:
        try:
            popt, _ = curve_fit(fn, x, y, p0=p0(x, y), sigma=sigma,
                                absolute_sigma=False, maxfev=40000)
            pred = fn(x, *popt)
            if not np.all(np.isfinite(pred)):
                raise RuntimeError("non-finite prediction")
            sse = float(np.sum(w * (y - pred) ** 2))
            k = len(popt)
            # AIC on the same weighted loss the R^2 uses; previously this was
            # the unweighted residual sum, so the selected shape and the
            # reported fit answered to different criteria.
            aic = n * np.log(max(sse / w.sum(), 1e-300)) + 2 * k
            out[name] = {"params": [float(v) for v in popt],
                         "r2": 1.0 - sse / sst if sst > 0 else None,
                         "aic": aic, "k": k}
        except Exception as e:  # a shape that will not converge is a result
            out[name] = {"error": str(e)[:80]}
    ok = {k: v for k, v in out.items() if "aic" in v and v.get("r2") is not None}
    if ok:
        best = min(ok, key=lambda k: ok[k]["aic"])
        out["_best"] = best
        out["_aic_gain_over_linear"] = (
            ok["linear"]["aic"] - ok[best]["aic"] if "linear" in ok else None)
        # Where a quadratic wins, the vertex is the substantive quantity: it says
        # where along the axis the outcome turns, in standard deviations.
        q = out.get("quadratic", {})
        if "params" in q and q["params"][2] != 0:
            a, b, c = q["params"]
            out["_quadratic_vertex_z"] = -b / (2 * c)
            out["_quadratic_opens"] = "up" if c > 0 else "down"
        sy = out.get("symmetric", {})
        if "params" in sy:
            a, b, c, x0 = sy["params"]
            out["_symmetric_vertex_z"] = x0
            # |c| this small linearizes exp(c|x-x0|) to 1 + c|x-x0|: the form
            # has degenerated to a V, and a and b have blown up to compensate.
            out["_symmetric_is_V"] = bool(abs(c) < 1e-2)
    return out


def profile(df: pl.DataFrame, score: str, outcome: str,
            nbins: int = NBINS) -> dict | None:
    """Equal-count bins on `score`; mean outcome and mean score in each."""
    d = df.filter(pl.col(score).is_finite())
    if d.height < 5000:
        return None
    d = d.sort(score).with_columns(
        ((pl.col(score).rank("ordinal") - 1) * nbins // pl.len())
        .cast(pl.Int32).alias("bin"))
    g = d.group_by("bin").agg(
        pl.col(score).mean().alias("x"),
        pl.col(outcome).mean().alias("y"),
        pl.len().alias("n")).sort("bin")
    x = g["x"].to_numpy().astype(np.float64)
    y = g["y"].to_numpy().astype(np.float64)
    w = g["n"].to_numpy().astype(np.float64)
    # Standardize x on the FILING-level mean and sd, not on the 40 bin centres.
    # Binning discards within-bin spread, so the sd of the centres is narrower
    # than the population's and every fitted vertex quoted in "z" would be on a
    # scale that does not match the per-sigma slopes reported elsewhere.
    mu = float(d[score].mean())
    sd = float(d[score].std())
    xz = (x - mu) / (sd if sd > 0 else 1.0)
    return {"n": int(d.height), "base": float(d[outcome].mean()),
            "x_mean": mu, "x_sd": sd,
            "bins": {"x_z": xz.tolist(), "y": y.tolist(), "n": w.astype(int).tolist()},
            "fits": fit_shapes(xz, y, w)}


def main() -> int:
    # ---- filing-level panel -------------------------------------------------
    parts = []
    for c in CLASSES:
        sp, tp = PROC / f"{SRC}_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                          "registration_date", "owner_name"])
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                          "topic_kl_vs_future"]).filter(
            pl.col("topic_kl_vs_past").is_finite()
            & pl.col("topic_kl_vs_future").is_finite())
        parts.append(tm.join(sc, on="serial_number", how="inner"))
        del tm, sc
        gc.collect()
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    d = d.with_columns(
        (pl.col("topic_kl_vs_past") - pl.col("topic_kl_vs_future")).alias("L"),
        (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("A"),
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        .cast(pl.Float64).alias("registered"),
    ).filter(pl.col("fy").is_between(FILE_LO, FILE_HI))
    log(f"[panel] {d.height:,} scored filings {FILE_LO}-{FILE_HI}")

    out = {"scoring": SRC, "nbins": NBINS, "gates": {}}

    # ---- gate 1: registration ----------------------------------------------
    log("\n=== registration gate ===")
    out["gates"]["registration"] = {}
    for axis in ("L", "A"):
        p = profile(d, axis, "registered")
        out["gates"]["registration"][axis] = p
        if p and "_best" in p["fits"]:
            fb = p["fits"]
            v = fb.get("_quadratic_vertex_z")
            log(f"  {axis}: base {100*p['base']:.2f}%  best={fb['_best']}  "
                f"dAIC vs linear {fb['_aic_gain_over_linear']:.1f}  "
                f"R2 lin {fb['linear']['r2']:.3f} -> {fb[fb['_best']]['r2']:.3f}"
                + (f"  vertex {v:+.2f}z ({fb['_quadratic_opens']})" if v is not None else ""))

    # ---- gate 2: survival ---------------------------------------------------
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("gd")
    ).drop_nulls("gd").group_by("serial_number").agg(pl.col("gd").min())
    r = d.filter(pl.col("registered") == 1.0).with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd")
    r = r.with_columns(pl.col("rd").dt.year().alias("ry")).filter(
        pl.col("ry").is_between(REG_LO, REG_HI))
    r = r.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("gd") - pl.col("rd")).dt.total_days() / 365.25).alias("age"))
    r = r.with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"))
    log(f"\n=== survival gate === ({r.height:,} registrations)")
    out["gates"]["survival"] = {}
    for axis in ("L", "A"):
        p = profile(r, axis, "failed")
        out["gates"]["survival"][axis] = p
        if p and "_best" in p["fits"]:
            fb = p["fits"]
            v = fb.get("_quadratic_vertex_z")
            log(f"  {axis}: base {100*p['base']:.2f}%  best={fb['_best']}  "
                f"dAIC vs linear {fb['_aic_gain_over_linear']:.1f}  "
                f"R2 lin {fb['linear']['r2']:.3f} -> {fb[fb['_best']]['r2']:.3f}"
                + (f"  vertex {v:+.2f}z ({fb['_quadratic_opens']})" if v is not None else ""))

    # ---- gate 3: commercial, split by what "in SEC records" means -----------
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik")
    tick = json.load((RAW / "company_tickers.json").open())
    listed = {int(v["cik_str"]) for v in tick.values()}
    fin = set(pl.read_parquet(PROC / "sec_firm_year.parquet")["cik"]
              .cast(pl.Int64).unique().to_list())
    cw = cw.with_columns(
        pl.when(pl.col("cik").cast(pl.Int64).is_in(list(listed))).then(pl.lit("listed"))
        .when(pl.col("cik").cast(pl.Int64).is_in(list(fin))).then(pl.lit("reporting"))
        .otherwise(pl.lit("registered_only")).alias("tier"))
    log("\n=== commercial gate === matched-owner tiers")
    for row in cw.group_by("tier").len().sort("len", descending=True).iter_rows():
        log(f"  {row[0]:<16} {row[1]:,} owner strings")

    # debut = owner's first filing; one row per owner, registered debuts only
    deb = d.filter(pl.col("owner_name").is_not_null()).with_columns(
        pl.col("filing_date").min().over("owner_name").alias("debut_date")
    ).filter(pl.col("filing_date") == pl.col("debut_date")).unique(subset="owner_name")
    deb = deb.filter(pl.col("registered") == 1.0)
    deb = deb.join(cw, on="owner_name", how="left").with_columns(
        pl.col("tier").fill_null("none"))
    log(f"  {deb.height:,} registered debut owners")
    out["gates"]["commercial"] = {}
    for label, expr in (
            ("any_sec", pl.col("tier") != "none"),
            ("listed", pl.col("tier") == "listed"),
            ("reporting_not_listed", pl.col("tier") == "reporting")):
        sub = deb.with_columns(expr.cast(pl.Float64).alias("hit"))
        out["gates"]["commercial"][label] = {}
        for axis in ("L", "A"):
            p = profile(sub, axis, "hit")
            out["gates"]["commercial"][label][axis] = p
            if p and "_best" in p["fits"]:
                fb = p["fits"]
                v = fb.get("_quadratic_vertex_z")
                log(f"  {label:<20} {axis}: base {100*p['base']:.3f}%  "
                    f"best={fb['_best']:<11} dAIC {fb['_aic_gain_over_linear']:7.1f}  "
                    f"R2 {fb['linear']['r2']:.3f} -> {fb[fb['_best']]['r2']:.3f}"
                    + (f"  vertex {v:+.2f}z ({fb['_quadratic_opens']})" if v is not None else ""))

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gate_curve_shapes.json").write_text(json.dumps(out, indent=1))
    log("\n[done] gate_curve_shapes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
