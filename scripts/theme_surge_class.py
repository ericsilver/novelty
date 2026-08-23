"""Class-level surges: a theme doubling to more than 1% of its class, and the gate inside the wave.

theme_surge.py measures surges against the whole corpus and finds five themes
ever doubling to a 1% corpus share; a 500-theme partition leaves most themes
far below that level. The event Eric's question describes happens inside a
class: a theme doubling to more than 1% (and 2%) of its class's filings. This
is that screen. Definitions mirror theme_surge.py with the share measured
within (theme, class):

    share_c(y) >= LEVEL, share_c(y) >= 2 x mean share_c over the prior three
    years, first such year only, surge window [y, y+3).

Reported: pooled in-surge gate effect (raw, within class and year, net of
lead and atypicality); and per episode with at least 1,000 in-surge
registrations, the growth multiple, base failure, and the within-episode lead
contrast. If crowding kills, base failure rises with growth and the
within-episode lead contrast is near zero; if being early kills as such, the
within-episode contrast stays at the corpus level.

Output: paper/results/theme_surge_class.json
"""
from __future__ import annotations

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
MIN_EPISODE = 1000


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    words = json.loads((PROC / "topic_lda_meta_T500.json").read_text())["top_words"]
    names = json.loads((RES / "per_industry_names.json").read_text())
    all_ = pl.concat([pl.read_parquet(CACHE / f"{c}.parquet") for c in CLASSES if (CACHE / f"{c}.parquet").exists()])
    thin = all_.filter(pl.col("thin") & pl.col("fy").is_between(1986, 2024))
    tot = thin.group_by(["cls", "fy"]).len().rename({"len": "N"})
    sh = thin.group_by(["theme", "cls", "fy"]).len().join(tot, on=["cls", "fy"]).with_columns(
        (pl.col("len") / pl.col("N")).alias("share")).sort(["theme", "cls", "fy"])
    # complete the year grid per (theme, cls) so rolling means see zeros
    grid = sh.select("theme", "cls").unique().join(pl.DataFrame({"fy": list(range(1986, 2025))}), how="cross")
    sh = grid.join(sh, on=["theme", "cls", "fy"], how="left").with_columns(pl.col("share").fill_null(0.0)).sort(["theme", "cls", "fy"])
    sh = sh.with_columns(pl.col("share").rolling_mean(window_size=PRIOR_YEARS).shift(1).over(["theme", "cls"]).alias("prior"))

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("cd")
    ).drop_nulls("cd").group_by("serial_number").agg(pl.col("cd").min())
    sc = pl.concat([pl.read_parquet(PROC / f"rolling_surprise_class{c}.parquet",
                                    columns=["serial_number", "topic_kl_vs_past", "topic_kl_vs_future", "topic_dkl"])
                    for c in CLASSES if (PROC / f"rolling_surprise_class{c}.parquet").exists()]).unique("serial_number")
    g_all = all_.filter(pl.col("ry").is_between(REG_LO, REG_HI)).join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("cd") - pl.col("rd")).dt.total_days() / 365.25).alias("age")
    ).with_columns(((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI)).fill_null(False).cast(pl.Float64).alias("failed"))
    g_all = g_all.join(sc, on="serial_number", how="left").with_columns(
        pl.col("topic_dkl").alias("L"), ((pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future")) / 2).alias("A"))

    out = {"double": DOUBLE, "surge_years": SURGE_YEARS, "min_episode": MIN_EPISODE, "levels": {}}
    for lab, level in LEVELS.items():
        surge = sh.filter((pl.col("share") >= level) & (pl.col("prior") > 0)
                          & (pl.col("share") >= DOUBLE * pl.col("prior"))).group_by(["theme", "cls"]).agg(
            pl.col("fy").min().alias("surge_y"))
        surge = surge.join(sh, left_on=["theme", "cls", "surge_y"], right_on=["theme", "cls", "fy"], how="left").select(
            "theme", "cls", "surge_y", pl.col("share").alias("surge_share"), pl.col("prior").alias("prior_share"))
        g = g_all.join(surge, on=["theme", "cls"], how="left").with_columns(
            (pl.col("surge_y").is_not_null() & (pl.col("fy") >= pl.col("surge_y"))
             & (pl.col("fy") < pl.col("surge_y") + SURGE_YEARS)).alias("in_surge"))
        n_in = int(g["in_surge"].sum())
        rec = {"n_surge_episodes": int(surge.height), "n_in_surge_regs": n_in,
               "fail_in_surge": float(g.filter(pl.col("in_surge"))["failed"].mean()),
               "fail_rest": float(g.filter(~pl.col("in_surge"))["failed"].mean())}
        g = g.with_columns((pl.col("failed") - pl.col("failed").mean().over(["cls", "ry"])).alias("f_dm"))
        gg = g.filter(pl.col("L").is_finite() & pl.col("A").is_finite())
        X = np.column_stack([np.ones(gg.height), gg["L"].to_numpy(), gg["A"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, gg["f_dm"].to_numpy(), rcond=None)
        gg = gg.with_columns(pl.Series("f_res", gg["f_dm"].to_numpy() - X @ beta))
        for name, df, col in (("within_class_year", g, "f_dm"), ("net_LA", gg, "f_res")):
            a = df.filter(pl.col("in_surge"))[col]
            rec[name] = {"mean": float(a.mean()), "se": float(a.std() / max(len(a), 1) ** 0.5)}
        log(f"[{lab}] episodes {surge.height}  in-surge regs {n_in:,}  fail {100*rec['fail_in_surge']:.1f}% vs "
            f"{100*rec['fail_rest']:.1f}%  within {100*rec['within_class_year']['mean']:+.2f}  net {100*rec['net_LA']['mean']:+.2f}")
        episodes = []
        for r in surge.iter_rows(named=True):
            sub = g.filter((pl.col("theme") == r["theme"]) & (pl.col("cls") == r["cls"]) & pl.col("in_surge")
                           & pl.col("L").is_finite())
            if sub.height < MIN_EPISODE:
                continue
            s = sub.sort(["ry", "L", "serial_number"]).with_columns(
                ((pl.col("L").rank("ordinal").over("ry") - 1) * 5 // pl.len().over("ry")).cast(pl.Int8).alias("q"))
            gq = s.group_by("q").agg(pl.col("failed").mean().alias("p"), pl.len().alias("n")).sort("q")
            p = [float(v) for v in gq["p"]]; nn = [int(v) for v in gq["n"]]
            if len(p) < 5 or min(nn) < 50:
                continue
            se = ((p[0] * (1 - p[0]) / nn[0]) + (p[4] * (1 - p[4]) / nn[4])) ** 0.5
            cellfail = float(g.filter((pl.col("cls") == r["cls"]) & ~pl.col("in_surge"))["failed"].mean())
            episodes.append({"theme": int(r["theme"]), "cls": r["cls"], "cls_name": names.get(r["cls"], ""),
                             "surge_year": int(r["surge_y"]), "growth_multiple": float(r["surge_share"] / r["prior_share"]),
                             "surge_share": float(r["surge_share"]), "n": int(sub.height),
                             "base_fail": float(sub["failed"].mean()), "class_fail_rest": cellfail,
                             "lead_lift": p[4] - p[0], "lead_se": se, "words": words[str(int(r["theme"]))][:6]})
        episodes.sort(key=lambda e: -e["n"])
        rec["episodes"] = episodes
        if len(episodes) >= 6:
            gm = np.log(np.array([e["growth_multiple"] for e in episodes]))
            bf = np.array([e["base_fail"] - e["class_fail_rest"] for e in episodes])
            ll = np.array([e["lead_lift"] for e in episodes])
            rec["n_episodes_tabled"] = len(episodes)
            rec["corr_growth_excessfail"] = float(np.corrcoef(gm, bf)[0, 1])
            rec["corr_growth_leadlift"] = float(np.corrcoef(gm, ll)[0, 1])
            rec["mean_excess_fail"] = float(np.mean(bf))
            rec["mean_lead_lift"] = float(np.mean(ll))
            rec["share_leadlift_below_1pt"] = float(np.mean(np.abs(ll) < 0.01))
            log(f"  episodes tabled {len(episodes)}: mean excess fail {100*rec['mean_excess_fail']:+.1f}pp  "
                f"mean lead lift {100*rec['mean_lead_lift']:+.2f}pp  corr(log growth, excess fail) {rec['corr_growth_excessfail']:+.2f}  "
                f"corr(log growth, lead lift) {rec['corr_growth_leadlift']:+.2f}")
        out["levels"][lab] = rec
    (RES / "theme_surge_class.json").write_text(json.dumps(out, indent=1))
    log("[done] theme_surge_class.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
