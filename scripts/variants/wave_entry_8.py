"""Waves: does joining cost, does the cost scale with the wave, and does joining early beat joining late?

Episodes are class-level surges as in theme_surge_class.py: a theme doubling,
over its own three prior years, to at least 1% of its class's filings, first
such year only. The entry window here is six years, [y, y+6), so that early
and late joiners exist. Registrations in the gate sample (2002-2018) whose
dominant theme and class match an episode and whose filing year is inside
the window are joiners; their entry year is e = filing year - surge year.

Per episode with at least MIN_N joiners:
    size        surge-year share of the class, and the growth multiple
    aftermath   share(y+5) / share(y) on the thinned universe -- did the
                theme keep growing after the surge, or recede
    excess      joiners' gate failure minus the class-and-cohort expectation
    timing      joiners' demeaned failure by entry year e

Pooled: excess failure regressed on log growth and log surge share; episodes
split by aftermath (kept growing vs receded); demeaned failure by entry year
pooled across episodes (each episode demeaned by its own mean, so the timing
comparison is within wave).

Output: paper/results/wave_timing.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "v3" / "_eval"
BASERES = REPO / "paper" / "results"
CACHE = BASERES / "theme_novelty_cache"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
LEVEL = 0.01
DOUBLE = 2.0
PRIOR_YEARS = 3
ENTRY_YEARS = 8
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5
MIN_N = 300


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    words = json.loads((PROC / "topic_lda_meta_T500.json").read_text())["top_words"]
    names = json.loads((BASERES / "per_industry_names.json").read_text())
    all_ = pl.concat([pl.read_parquet(CACHE / f"{c}.parquet") for c in CLASSES if (CACHE / f"{c}.parquet").exists()])
    thin = all_.filter(pl.col("thin") & pl.col("fy").is_between(1986, 2024))
    tot = thin.group_by(["cls", "fy"]).len().rename({"len": "N"})
    sh = thin.group_by(["theme", "cls", "fy"]).len().join(tot, on=["cls", "fy"]).with_columns(
        (pl.col("len") / pl.col("N")).alias("share")).sort(["theme", "cls", "fy"])
    grid = sh.select("theme", "cls").unique().join(pl.DataFrame({"fy": list(range(1986, 2025))}), how="cross")
    shg = grid.join(sh, on=["theme", "cls", "fy"], how="left").with_columns(
        pl.col("share").fill_null(0.0)).sort(["theme", "cls", "fy"])
    shg = shg.with_columns(pl.col("share").rolling_mean(window_size=PRIOR_YEARS).shift(1).over(["theme", "cls"]).alias("prior"))
    surge = shg.filter((pl.col("share") >= LEVEL) & (pl.col("prior") > 0)
                       & (pl.col("share") >= DOUBLE * pl.col("prior"))).group_by(["theme", "cls"]).agg(
        pl.col("fy").min().alias("surge_y"))
    surge = surge.join(shg, left_on=["theme", "cls", "surge_y"], right_on=["theme", "cls", "fy"], how="left").select(
        "theme", "cls", "surge_y", pl.col("share").alias("s0"), pl.col("prior").alias("sprior"))
    after = shg.select("theme", "cls", pl.col("fy").alias("y5"), pl.col("share").alias("s5"))
    surge = surge.join(after, left_on=["theme", "cls", (pl.col("surge_y") + 5)],
                       right_on=["theme", "cls", "y5"], how="left").with_columns(
        (pl.col("s5") / pl.col("s0")).alias("aftermath"))
    log(f"[episodes] {surge.height} class-level surges at {100*LEVEL:.0f}%")

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    g = all_.filter(pl.col("ry").is_between(REG_LO, REG_HI)).join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64).alias("failed"))
    g = g.with_columns((pl.col("failed") - pl.col("failed").mean().over(["cls", "ry"])).alias("f_dm"))

    j = g.join(surge, on=["theme", "cls"], how="inner").with_columns(
        (pl.col("fy") - pl.col("surge_y")).alias("e")).filter(pl.col("e").is_between(0, ENTRY_YEARS - 1))
    log(f"[joiners] {j.height:,} gate registrations inside an entry window")

    episodes = []
    for r in surge.iter_rows(named=True):
        sub = j.filter((pl.col("theme") == r["theme"]) & (pl.col("cls") == r["cls"]))
        if sub.height < MIN_N:
            continue
        episodes.append({"theme": int(r["theme"]), "cls": r["cls"], "cls_name": names.get(r["cls"], ""),
                         "surge_year": int(r["surge_y"]), "growth": float(r["s0"] / r["sprior"]),
                         "share": float(r["s0"]),
                         "aftermath": float(r["aftermath"]) if r["aftermath"] is not None else None,
                         "n": int(sub.height), "excess": float(sub["f_dm"].mean()),
                         "se": float(sub["f_dm"].std() / sub.height ** 0.5),
                         "words": words[str(int(r["theme"]))][:5]})
    log(f"[tabled] {len(episodes)} episodes with >= {MIN_N} joiners")
    out = {"level": LEVEL, "entry_years": ENTRY_YEARS, "min_n": MIN_N,
           "n_episodes": len(episodes), "episodes": sorted(episodes, key=lambda e: -e["n"])[:40]}

    ep = pl.DataFrame(episodes)
    if ep.height >= 10:
        lg = np.log(ep["growth"].to_numpy()); ls = np.log(ep["share"].to_numpy()); ex = ep["excess"].to_numpy()
        w = ep["n"].to_numpy().astype(float)
        out["dose"] = {"corr_loggrowth_excess": float(np.corrcoef(lg, ex)[0, 1]),
                       "corr_logshare_excess": float(np.corrcoef(ls, ex)[0, 1]),
                       "wcorr_logshare_excess": float(np.cov(ls, ex, aweights=w)[0, 1]
                                                      / (np.sqrt(np.cov(ls, ls, aweights=w)[0, 1])
                                                         * np.sqrt(np.cov(ex, ex, aweights=w)[0, 1]))),
                       "mean_excess": float(ex.mean()), "share_excess_positive": float((ex > 0).mean())}
        # terciles of size (surge share)
        q = np.quantile(ls, [1 / 3, 2 / 3])
        for lab, mask in (("small", ls <= q[0]), ("mid", (ls > q[0]) & (ls <= q[1])), ("large", ls > q[1])):
            out["dose"][f"excess_{lab}"] = {"k": int(mask.sum()), "mean": float(ex[mask].mean()),
                                            "mean_share": float(ep["share"].to_numpy()[mask].mean())}
        log(f"[dose] corr(log growth, excess) {out['dose']['corr_loggrowth_excess']:+.2f}  "
            f"corr(log share, excess) {out['dose']['corr_logshare_excess']:+.2f}  "
            f"small/mid/large excess {100*out['dose']['excess_small']['mean']:+.1f}/"
            f"{100*out['dose']['excess_mid']['mean']:+.1f}/{100*out['dose']['excess_large']['mean']:+.1f}pp")
        am = ep.filter(pl.col("aftermath").is_not_null())
        grow = am.filter(pl.col("aftermath") >= 1.0); rec = am.filter(pl.col("aftermath") < 1.0)
        out["aftermath"] = {"n_grew": int(grow.height), "n_receded": int(rec.height),
                            "excess_grew": float(grow["excess"].mean()) if grow.height else None,
                            "excess_receded": float(rec.height and rec["excess"].mean()),
                            "corr_aftermath_excess": float(np.corrcoef(np.log(am["aftermath"].to_numpy()),
                                                                        am["excess"].to_numpy())[0, 1])}
        log(f"[aftermath] grew {grow.height} excess {100*out['aftermath']['excess_grew']:+.1f}pp  "
            f"receded {rec.height} excess {100*out['aftermath']['excess_receded']:+.1f}pp  "
            f"corr(log aftermath, excess) {out['aftermath']['corr_aftermath_excess']:+.2f}")

    # Timing within wave: demean failure within episode, pool by entry year.
    keys = {(e["theme"], e["cls"]) for e in episodes}
    jt = j.filter(pl.struct(["theme", "cls"]).map_elements(lambda s: (s["theme"], s["cls"]) in keys,
                                                           return_dtype=pl.Boolean))
    jt = jt.with_columns((pl.col("f_dm") - pl.col("f_dm").mean().over(["theme", "cls"])).alias("f_ep"))
    tim = jt.group_by("e").agg(pl.len().alias("n"), pl.col("f_ep").mean().alias("m"),
                               (pl.col("f_ep").std() / pl.len().sqrt()).alias("se"),
                               pl.col("f_dm").mean().alias("m_vs_class")).sort("e")
    out["timing_by_entry_year"] = {str(r["e"]): {"n": r["n"], "within_wave": r["m"], "se": r["se"],
                                                 "vs_class": r["m_vs_class"]} for r in tim.iter_rows(named=True)}
    for r in tim.iter_rows(named=True):
        log(f"  e={r['e']}: n={r['n']:6,}  within-wave {100*r['m']:+.2f}pp (SE {100*r['se']:.2f})  vs class {100*r['m_vs_class']:+.2f}pp")
    (RES / "wave_entry_8.json").write_text(json.dumps(out, indent=1, default=float))
    log("[done] wave_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
