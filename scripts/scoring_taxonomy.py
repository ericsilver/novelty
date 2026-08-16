"""What actually differs between the six ways this corpus has been scored.

Two choices define a scoring. The REPRESENTATION is what a filing's language is
turned into before it is compared to anything -- a distribution over 50 fitted
topics, or a distribution over the class vocabulary's terms. The REFERENCE is
what it is compared against, and that has two sub-choices that are easy to
conflate: the anchoring (one reference per class-year, or one per filing, built
on that filing's own date) and the weighting inside the window (every day equal,
or decaying by half every two years).

Six builds exist on this corpus:

    topics x class-year                 topic_surprise_class*
    topics x per-filing, flat           rolling_surprise_class*     (production)
    topics x per-filing, half-life 2y   decay_surprise_class*
    terms  x class-year, half-life 2y   surprise_class*             (legacy)
    terms  x per-filing, flat           termroll_surprise_class*
    terms  x per-filing, half-life 2y   termrolldecay_surprise_class*

The legacy term build is the one to be careful with: it moves anchoring AND
weighting away from production at the same time, which is why results computed
on it could not be attributed to either choice. That is the reason the two
per-filing term builds exist.

What this reports, on the common set of filings every build scored:

  scale        sd of lead and of atypicality under each build
  agreement    pairwise correlation of lead, and of atypicality, across builds
  decomposition how much of the disagreement is representation, anchoring, and
               weighting, by comparing builds that differ in exactly one
  month        mean lead by calendar month of filing -- the signature of annual
               anchoring, which has no reason to exist and is large when the
               reference is a calendar year
  gate         the first-gate lead contrast under each, so the substantive
               claim can be read off the same table

Output: paper/results/scoring_taxonomy.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

# key -> (file prefix, representation, anchoring, weighting, past col, future col)
BUILDS = {
    "topic_annual": ("topic_surprise", "topics", "class-year", "flat",
                     "topic_kl_vs_past", "topic_kl_vs_future"),
    "topic_flat": ("rolling_surprise", "topics", "per-filing", "flat",
                   "topic_kl_vs_past", "topic_kl_vs_future"),
    "topic_decay": ("decay_surprise", "topics", "per-filing", "half-life 2y",
                    "topic_kl_vs_past", "topic_kl_vs_future"),
    "term_annual_decay": ("surprise", "terms", "class-year", "half-life 2y",
                          "kl_vs_past", "kl_vs_future"),
    "term_flat": ("termroll_surprise", "terms", "per-filing", "flat",
                  "topic_kl_vs_past", "topic_kl_vs_future"),
    "term_decay": ("termrolldecay_surprise", "terms", "per-filing", "half-life 2y",
                   "topic_kl_vs_past", "topic_kl_vs_future"),
}
CLASSES = [f"{i:03d}" for i in range(1, 46)]
REG_LO, REG_HI = 2002, 2018
GATE_LO, GATE_HI = 4.0, 8.5

# pairs differing in exactly one choice, for the decomposition
CONTRASTS = [
    ("anchoring: class-year vs per-filing (topics, flat)", "topic_annual", "topic_flat"),
    ("weighting: flat vs half-life 2y (topics, per-filing)", "topic_flat", "topic_decay"),
    ("weighting: flat vs half-life 2y (terms, per-filing)", "term_flat", "term_decay"),
    ("representation: topics vs terms (per-filing, flat)", "topic_flat", "term_flat"),
    ("representation: topics vs terms (per-filing, decayed)", "topic_decay", "term_decay"),
    ("both at once: production vs legacy term build", "topic_flat", "term_annual_decay"),
]


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def load(key: str) -> pl.DataFrame | None:
    prefix, _, _, _, cp, cf = BUILDS[key]
    parts = []
    for c in CLASSES:
        f = PROC / f"{prefix}_class{c}.parquet"
        if not f.exists():
            continue
        cols = pl.read_parquet_schema(f)
        if cp not in cols or cf not in cols:
            continue
        d = pl.read_parquet(f, columns=["serial_number", cp, cf]).rename(
            {cp: "kp", cf: "kf"})
        parts.append(d.filter(pl.col("kp").is_finite() & pl.col("kf").is_finite()))
        del d
        gc.collect()
    if not parts:
        return None
    d = pl.concat(parts).unique(subset="serial_number")
    del parts
    gc.collect()
    return d.with_columns(
        (pl.col("kp") - pl.col("kf")).alias("L"),
        (0.5 * (pl.col("kp") + pl.col("kf"))).alias("A"),
    ).select("serial_number", "L", "A")


def main() -> int:
    frames = {}
    for k in BUILDS:
        d = load(k)
        if d is None:
            log(f"[{k}] missing -- skipped")
            continue
        frames[k] = d
        log(f"[{k}] {d.height:,} scored")

    keys = list(frames)
    common = frames[keys[0]].select("serial_number")
    for k in keys[1:]:
        common = common.join(frames[k].select("serial_number"), on="serial_number",
                             how="inner")
    log(f"\n[common] {common.height:,} filings scored by all {len(keys)} builds")

    panel = common
    for k in keys:
        panel = panel.join(
            frames[k].rename({"L": f"L_{k}", "A": f"A_{k}"}),
            on="serial_number", how="inner")

    out = {"builds": {k: {"representation": v[1], "anchoring": v[2],
                          "weighting": v[3], "n_scored": int(frames[k].height)}
                      for k, v in BUILDS.items() if k in frames},
           "n_common": int(panel.height), "scale": {}, "corr_lead": {},
           "corr_atypicality": {}, "one_choice_at_a_time": {},
           "month_gradient": {}, "gate": {}}

    log("\nbuild                 sd(lead)  sd(atypicality)")
    for k in keys:
        s = {"sd_lead": float(panel[f"L_{k}"].std()),
             "sd_atypicality": float(panel[f"A_{k}"].std()),
             "mean_atypicality": float(panel[f"A_{k}"].mean())}
        out["scale"][k] = s
        log(f"  {k:<20} {s['sd_lead']:7.4f}   {s['sd_atypicality']:7.4f}")

    def corr(a: str, b: str, method: str = "pearson") -> float:
        return float(panel.select(pl.corr(a, b, method=method)).item())

    log("\npairwise correlation of LEAD (lower triangle)")
    for i, a in enumerate(keys):
        row = []
        for b in keys[:i]:
            r = corr(f"L_{a}", f"L_{b}")
            out["corr_lead"][f"{a}|{b}"] = r
            row.append(f"{r:+.3f}")
        out["corr_atypicality"].update(
            {f"{a}|{b}": corr(f"A_{a}", f"A_{b}") for b in keys[:i]})
        if row:
            log(f"  {a:<20} " + " ".join(f"{x:>7}" for x in row))
    log("  " + " " * 20 + " " + " ".join(f"{k[:7]:>7}" for k in keys[:-1]))

    log("\none choice at a time            corr(lead)  corr(atypicality)")
    for label, a, b in CONTRASTS:
        if a not in frames or b not in frames:
            continue
        rl, ra = corr(f"L_{a}", f"L_{b}"), corr(f"A_{a}", f"A_{b}")
        rs = corr(f"L_{a}", f"L_{b}", "spearman")
        out["one_choice_at_a_time"][label] = {
            "lead_pearson": rl, "lead_spearman": rs, "atypicality_pearson": ra}
        log(f"  {label:<44} {rl:+.3f}      {ra:+.3f}")

    # --- month-of-filing gradient: the fingerprint of annual anchoring ---
    fd = []
    for c in CLASSES:
        f = PROC / f"tm_class{c}.parquet"
        if f.exists():
            fd.append(pl.read_parquet(f, columns=["serial_number", "filing_date"]))
    fdd = pl.concat(fd).filter(
        pl.col("filing_date").fill_null("").str.len_chars() >= 8).with_columns(
        pl.col("filing_date").str.slice(4, 2).cast(pl.Int32).alias("month"))
    del fd
    gc.collect()
    pm = panel.join(fdd.select("serial_number", "month"), on="serial_number",
                    how="inner")
    log("\nmean lead by filing month (Jan -> Dec), standardized within build")
    for k in keys:
        g = pm.group_by("month").agg(pl.col(f"L_{k}").mean().alias("m")).sort("month")
        sd = float(panel[f"L_{k}"].std())
        v = [float(r["m"]) / sd for r in g.iter_rows(named=True)]
        out["month_gradient"][k] = {"by_month_sd_units": v,
                                    "dec_minus_jan_sd": v[-1] - v[0]}
        log(f"  {k:<20} Jan {v[0]:+.3f}  Dec {v[-1]:+.3f}   "
            f"swing {v[-1]-v[0]:+.3f} sd")

    # --- gate contrast under each build ---
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())
    rg = []
    for c in CLASSES:
        f = PROC / f"tm_class{c}.parquet"
        if f.exists():
            rg.append(pl.read_parquet(f, columns=["serial_number", "registration_date"]))
    reg = pl.concat(rg).filter(
        pl.col("registration_date").fill_null("").str.len_chars() >= 8).with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd")
    ).drop_nulls("rd")
    del rg
    gc.collect()
    g = panel.join(reg.select("serial_number", "rd"), on="serial_number", how="inner")
    g = g.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"),
        pl.col("rd").dt.year().alias("ry"))
    g = g.filter(pl.col("ry").is_between(REG_LO, REG_HI)).with_columns(
        ((pl.col("age") >= GATE_LO) & (pl.col("age") < GATE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"))
    log(f"\ngate sample (common filings only): {g.height:,}")
    log("build                 Q5-Q1 lead   Q5-Q1 atypicality")
    for k in keys:
        row = {}
        for axis, col in (("lead", f"L_{k}"), ("atypicality", f"A_{k}")):
            s = g.sort([col, "serial_number"]).with_columns(
                ((pl.col(col).rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
            gg = s.group_by("q").agg(pl.col("failed").mean().alias("p"),
                                     pl.len().alias("n")).sort("q")
            v = [float(r["p"]) for r in gg.iter_rows(named=True)]
            p1, p5 = v[0], v[4]
            n1, n5 = int(gg["n"][0]), int(gg["n"][4])
            se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
            row[axis] = {"quintiles": v, "lift": p5 - p1, "se": se}
        out["gate"][k] = row
        log(f"  {k:<20} {100*row['lead']['lift']:+7.2f}pp    "
            f"{100*row['atypicality']['lift']:+7.2f}pp")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "scoring_taxonomy.json").write_text(json.dumps(out, indent=1))
    log("\n[done] scoring_taxonomy.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
