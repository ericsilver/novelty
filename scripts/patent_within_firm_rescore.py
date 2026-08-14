"""Re-estimate the within-firm patenting comparison under per-filing anchoring.

The claim is that lead is empirically distinct from patent-track invention:
within a firm, years of heavier patenting are not years of more unusual
trademark vocabulary. The table backing it was computed under an
exponential-decay reference on TERMS anchored on calendar years, which is not
the construction used anywhere else in the paper, and the topic-scored figures
quoted alongside it came from the same class-year build.

This re-estimates the whole comparison on per-filing reference windows, under
each available scoring variant, so the claim can be stated on the paper's own
construction rather than on a legacy one.

The firm-year patent counts are taken from the published panel
(data_publish/firm_year_patents_and_dkl.csv), which carries both the USPTO
owner string it was built on and the normalized clustering key. Scores are
re-joined on the raw owner string and year, because that is what the scored
filings carry; the firm identifier is then the normalized key, matching the
construction the original estimate used. Patent counts are untouched, so the
only thing varying across rows of the output is the scoring.

Specification: mean lead, standardized, on log(1 + patents), with firm fixed
effects absorbed by within-firm demeaning and standard errors clustered on
firm. Only firms with variation in both variables contribute, which is why the
estimation sample is smaller than the panel.

The timing decomposition is run the same way, one offset at a time rather than
jointly: log(1+patents) is strongly autocorrelated within firm, so entering the
offsets together splits one coefficient between them arithmetically. Offset
t+1 puts the patent AFTER the trademark year, which is the ordering that would
hold if the mark were filed first.

Output: paper/results/patent_within_firm_rescore.json
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

VARIANTS = [
    ("rolling", "topics, flat per-filing window"),
    ("decay", "topics, 2-year half-life"),
    ("termroll", "terms, flat per-filing window"),
    ("termrolldecay", "terms, 2-year half-life"),
]
CLASSES = [f"{i:03d}" for i in range(1, 46)]


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def firm_year_lead(src: str) -> pl.DataFrame | None:
    """Mean lead per (owner string, filing year) under one scoring variant."""
    parts = []
    for c in CLASSES:
        sp = PROC / f"{src}_surprise_class{c}.parquet"
        tp = PROC / f"tm_class{c}.parquet"
        if not (sp.exists() and tp.exists()):
            continue
        tm = pl.read_parquet(
            tp, columns=["serial_number", "filing_date", "owner_name"]).filter(
            pl.col("owner_name").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        parts.append(
            tm.join(sc, on="serial_number", how="inner").with_columns(
                pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year")
            ).select("owner_name", "year", "topic_dkl"))
        del tm, sc
        gc.collect()
    if not parts:
        return None
    d = pl.concat(parts)
    del parts
    gc.collect()
    return d.group_by(["owner_name", "year"]).agg(
        pl.col("topic_dkl").mean().alias("mean_dkl"),
        pl.len().alias("n_scored"))


def within_firm_fe(df: pl.DataFrame) -> dict | None:
    """Standardized mean lead on log(1+patents), firm FE, clustered on firm."""
    d = df.with_columns(
        (pl.col("mean_dkl") - pl.col("mean_dkl").mean()).truediv(
            pl.col("mean_dkl").std()).alias("y"),
        (pl.col("n_patents").cast(pl.Float64) + 1.0).log().alias("x"),
    )
    # keep firms with within-firm variation in the regressor
    d = d.with_columns(
        pl.col("x").std().over("firm").alias("sdx"),
        pl.len().over("firm").alias("k"))
    d = d.filter((pl.col("k") >= 2) & (pl.col("sdx") > 0))
    if d.height < 1000:
        return None
    d = d.with_columns(
        (pl.col("y") - pl.col("y").mean().over("firm")).alias("yd"),
        (pl.col("x") - pl.col("x").mean().over("firm")).alias("xd"))
    x = d["xd"].to_numpy()
    y = d["yd"].to_numpy()
    sxx = float(x @ x)
    if sxx <= 0:
        return None
    b = float(x @ y) / sxx
    resid = y - b * x
    # cluster-robust variance on the firm
    ids = d["firm"].to_numpy()
    order = np.argsort(ids, kind="stable")
    xs, rs, idss = x[order], resid[order], ids[order]
    bounds = np.flatnonzero(np.r_[True, idss[1:] != idss[:-1], True])
    meat = 0.0
    for a, z in zip(bounds[:-1], bounds[1:]):
        s = float(xs[a:z] @ rs[a:z])
        meat += s * s
    se = float(np.sqrt(meat) / sxx)
    n_firms = len(bounds) - 1
    # small-sample correction, as in standard cluster practice
    dof = n_firms / max(n_firms - 1, 1)
    se *= np.sqrt(dof)
    return {"coef": b, "se": se, "t": b / se if se else None,
            "n_obs": int(d.height), "n_firms": int(n_firms)}


def main() -> int:
    panel = pl.read_csv(PUB / "firm_year_patents_and_dkl.csv").select(
        pl.col("uspto_owner_name").alias("owner_name"),
        pl.col("normalized_name").alias("firm"), "year", "n_patents")
    log(f"[panel] {panel.height:,} firm-years, "
        f"{panel['owner_name'].n_unique():,} owner strings, "
        f"{panel['firm'].n_unique():,} normalized firms")

    out = {"panel_rows": int(panel.height), "results": {}}
    for src, desc in VARIANTS:
        fy = firm_year_lead(src)
        if fy is None:
            log(f"[{src}] no scored files -- skipped")
            continue
        j = panel.join(fy, on=["owner_name", "year"], how="inner")
        # collapse owner-string variants onto the normalized firm key, weighting
        # each string by how many of its filings were scored
        j = j.group_by(["firm", "year"]).agg(
            ((pl.col("mean_dkl") * pl.col("n_scored")).sum()
             / pl.col("n_scored").sum()).alias("mean_dkl"),
            pl.col("n_patents").max().alias("n_patents"))
        r = within_firm_fe(j)
        lags = {}
        for off in (-2, -1, 0, 1):
            # shift the patent count by `off` years onto the same firm-year
            shifted = j.select(
                "firm", (pl.col("year") + off).alias("year"),
                pl.col("n_patents").alias("np_off"))
            k = j.select("firm", "year", "mean_dkl").join(
                shifted, on=["firm", "year"], how="inner").rename(
                {"np_off": "n_patents"})
            lags[f"t{off:+d}" if off else "t"] = within_firm_fe(k)
            del shifted, k
        out["results"][src] = {"description": desc,
                               "matched_firm_years": int(j.height),
                               "fe": r, "timing": lags}
        if r:
            log(f"[{src:14}] {desc:32}  matched {j.height:>7,}  "
                f"beta {r['coef']:+.4f} sigma  (SE {r['se']:.4f}, "
                f"t {r['t']:+.1f}, {r['n_firms']:,} firms)")
            tl = " ".join(
                f"{k}={v['coef']:+.3f}(t{v['t']:+.1f})"
                for k, v in out["results"][src]["timing"].items() if v)
            log(f"                 timing: {tl}")
        else:
            log(f"[{src:14}] matched {j.height:,} -- too few for FE")
        del fy, j
        gc.collect()

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "patent_within_firm_rescore.json").write_text(json.dumps(out, indent=1))
    log("\n[done] patent_within_firm_rescore.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
