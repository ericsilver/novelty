"""Effective novelty: themes whose share doubled to more than 1% (and 2%), and what the gate did to them.

Truly new themes are rare in a 500-theme partition of a 140-year record (779
gate registrations). The observable event is a surge: an established theme's
corpus share doubling to a level that matters. A theme has a surge year y when

    share(y) >= LEVEL          (1% main, 2% strict)
    share(y) >= 2 x mean share over the three years before y

with share measured on the thinned all-filing universe (every fourth serial)
so classes and years are comparable, and only a theme's first surge counted.
Registrations whose dominant theme is inside [y, y+SURGE_YEARS) at filing are
"in surge".

Reported, for surging against non-surging registrations, within class and
registration year: gate failure, raw and net of the filing's own lead and
atypicality.

The surge table also carries what Eric's competition-against-novelty question
needs: per surge episode, the theme's growth multiple into the surge, the base
failure of its in-surge registrations, and the within-theme lead contrast among
them. If crowding is what kills, base failure should rise with the growth
multiple and the within-theme lead contrast should be small; if new things are
hard, the within-theme contrast should persist regardless of growth.

Input : paper/results/theme_novelty_cache/*.parquet   (from theme_novelty_origin.py)
Output: paper/results/theme_surge.json
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
CACHE = RES / "theme_novelty_cache"
CLASSES = [f"{i:03d}" for i in range(1, 46)]
LEVELS = {"1pct": 0.01, "2pct": 0.02}
DOUBLE = 2.0
PRIOR_YEARS = 3
SURGE_YEARS = 3
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    words = json.loads((PROC / "topic_lda_meta_T500.json").read_text())["top_words"]
    all_ = pl.concat([pl.read_parquet(CACHE / f"{c}.parquet") for c in CLASSES if (CACHE / f"{c}.parquet").exists()])
    thin = all_.filter(pl.col("thin") & pl.col("fy").is_between(1986, 2024))
    tot = thin.group_by("fy").len().rename({"len": "N"})
    sh = thin.group_by(["theme", "fy"]).len().join(tot, on="fy").with_columns(
        (pl.col("len") / pl.col("N")).alias("share")).sort(["theme", "fy"])
    sh = sh.with_columns(pl.col("share").rolling_mean(window_size=PRIOR_YEARS).shift(1).over("theme").alias("prior"))
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    sc = pl.concat([pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                                    columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"])
                    for c in CLASSES if (PROC / f"rolling_surprise_class{c}.parquet").exists()]).unique("serial_number")

    out = {"double": DOUBLE, "prior_years": PRIOR_YEARS, "surge_years": SURGE_YEARS, "levels": {}}
    g_all = all_.filter(pl.col("ry").is_between(REG_LO, REG_HI)).join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64).alias("failed"))
    g_all = g_all.join(sc, on="serial_number", how="left").with_columns(
        pl.col("topic_dkl").alias("L"), ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))

    for lab, level in LEVELS.items():
        surge = sh.filter((pl.col("share") >= level) & (pl.col("prior") > 0)
                          & (pl.col("share") >= DOUBLE * pl.col("prior"))).group_by("theme").agg(
            pl.col("fy").min().alias("surge_y"))
        surge = surge.join(sh, left_on=["theme", "surge_y"], right_on=["theme", "fy"], how="left").select(
            "theme", "surge_y", pl.col("share").alias("surge_share"), pl.col("prior").alias("prior_share"))
        log(f"[{lab}] {surge.height} themes surge (double to >= {100*level:.0f}%)")
        g = g_all.join(surge, on="theme", how="left").with_columns(
            (pl.col("surge_y").is_not_null() & (pl.col("fy") >= pl.col("surge_y"))
             & (pl.col("fy") < pl.col("surge_y") + SURGE_YEARS)).alias("in_surge"))
        n_in = int(g["in_surge"].sum())
        rec = {"n_surge_themes": int(surge.height), "n_in_surge_regs": n_in,
               "fail_in_surge": float(g.filter(pl.col("in_surge"))["failed"].mean()),
               "fail_rest": float(g.filter(~pl.col("in_surge"))["failed"].mean())}
        # within class x year, and net of L, A
        g = g.with_columns((pl.col("failed") - pl.col("failed").mean().over(["cls", "ry"])).alias("f_dm"))
        gg = g.filter(pl.col("L").is_finite() & pl.col("A").is_finite())
        X = np.column_stack([np.ones(gg.height), gg["L"].to_numpy(), gg["A"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, gg["f_dm"].to_numpy(), rcond=None)
        gg = gg.with_columns(pl.Series("f_res", gg["f_dm"].to_numpy() - X @ beta))
        for name, df, col in (("within_class_year", g, "f_dm"), ("net_LA", gg, "f_res")):
            a = df.filter(pl.col("in_surge"))[col]
            rec[name] = {"mean": float(a.mean()), "se": float(a.std() / max(len(a), 1) ** 0.5)}
        log(f"  in-surge n={n_in:,} fail {100*rec['fail_in_surge']:.1f}% vs {100*rec['fail_rest']:.1f}%  "
            f"within {100*rec['within_class_year']['mean']:+.2f} (SE {100*rec['within_class_year']['se']:.2f})  "
            f"net L,A {100*rec['net_LA']['mean']:+.2f}")
        # Episode table: growth multiple vs base failure vs within-theme lead contrast.
        episodes = []
        for r in surge.iter_rows(named=True):
            t, y = r["theme"], r["surge_y"]
            sub = g.filter((pl.col("theme") == t) & pl.col("in_surge") & pl.col("L").is_finite())
            if sub.height < 2000:
                continue
            s = sub.sort(["ry", "L", "serial_number"]).with_columns(
                ((pl.col("L").rank("ordinal").over("ry") - 1) * 5 // pl.len().over("ry")).cast(pl.Int8).alias("q"))
            gq = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
            p = [float(v) for v in gq["p"]]; nn = [int(v) for v in gq["n"]]
            if len(p) < 5 or min(nn) < 100:
                continue
            se = ((p[0] * (1 - p[0]) / nn[0]) + (p[4] * (1 - p[4]) / nn[4])) ** 0.5
            episodes.append({"theme": int(t), "surge_year": int(y),
                             "growth_multiple": float(r["surge_share"] / r["prior_share"]),
                             "surge_share": float(r["surge_share"]), "n": int(sub.height),
                             "base_fail": float(sub["failed"].mean()),
                             "lead_lift": p[4] - p[0], "lead_se": se,
                             "words": words[str(int(t))][:6]})
        episodes.sort(key=lambda e: -e["n"])
        rec["episodes"] = episodes
        if len(episodes) >= 4:
            gm = np.array([e["growth_multiple"] for e in episodes]); bf = np.array([e["base_fail"] for e in episodes])
            ll = np.array([e["lead_lift"] for e in episodes])
            rec["corr_growth_basefail"] = float(np.corrcoef(np.log(gm), bf)[0, 1])
            rec["corr_growth_leadlift"] = float(np.corrcoef(np.log(gm), ll)[0, 1])
            rec["mean_within_theme_lead_lift"] = float(np.mean(ll))
            log(f"  episodes n={len(episodes)}: corr(log growth, base fail) {rec['corr_growth_basefail']:+.2f}  "
                f"corr(log growth, lead lift) {rec['corr_growth_leadlift']:+.2f}  mean lead lift {100*rec['mean_within_theme_lead_lift']:+.2f}pp")
        out["levels"][lab] = rec
    (RES / "theme_surge.json").write_text(json.dumps(out, indent=1))
    log("[done] theme_surge.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
