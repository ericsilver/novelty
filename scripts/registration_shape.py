"""Fine-grained registration profiles on both axes, and what drives the V.

Quintile plots of registration against lead show a sawtooth, which is an
artifact of binning a non-monotone function into five buckets. At twenty bins
the true shape appears: registration falls to a trough just above the median
lead and rises steadily after it.

That trough is compositional. |lead| correlates with atypicality at +0.31, so
the middle of the lead distribution is disproportionately low-atypicality
filings, and those register poorly. Splitting by atypicality quintile removes
the trough everywhere except the lowest-atypicality fifth.

Writes paper/results/registration_shape.json for the figure script.
"""
from __future__ import annotations
import gc, sys
from pathlib import Path
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
CLASSES = [f"{i:03d}" for i in range(1, 46)]


def log(m): print(m, file=sys.stderr, flush=True)


parts = []
for c in CLASSES:
    sp, tp = PROC / f"rolling_surprise_class{c}.parquet", PROC / f"tm_class{c}.parquet"
    if not (sp.exists() and tp.exists()):
        continue
    tm = pl.read_parquet(tp, columns=["serial_number", "filing_date",
                                      "registration_date", "owner_name"]).filter(
        pl.col("owner_name").is_not_null()
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
    sc = pl.read_parquet(sp, columns=["serial_number", "topic_kl_vs_past",
                                      "topic_kl_vs_future", "topic_dkl"]).filter(
        pl.col("topic_dkl").is_finite())
    parts.append(tm.join(sc, on="serial_number", how="inner").with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("fy"),
        pl.lit(c).alias("cls")))
    del tm, sc; gc.collect()

d = pl.concat(parts); del parts; gc.collect()
d = d.sort(["owner_name", "fy", "serial_number"]).unique(subset="owner_name", keep="first")
d = d.filter(pl.col("fy").is_between(1995, 2018)).with_columns(
    (0.5 * (pl.col("topic_kl_vs_past") + pl.col("topic_kl_vs_future"))).alias("atyp"),
    (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
    .cast(pl.Float64).alias("registered"))
log(f"[panel] {d.height:,} debut owners")

# how bad are the ties?
n_tied = d.height - d["topic_dkl"].n_unique()
log(f"[ties] {d['topic_dkl'].n_unique():,} distinct lead values among "
    f"{d.height:,} filings -> {n_tied/d.height*100:.1f}% share a value")
top = d.group_by("topic_dkl").len().sort("len", descending=True).head(3)
log(f"[ties] largest tie groups: {[int(r['len']) for r in top.iter_rows(named=True)]}")

for score, name in (("topic_dkl", "lead L"), ("atyp", "atypicality A")):
    log(f"\n=== registration completion by {name} ===")
    for k in (5, 10, 20):
        dd = d.sort([score, "serial_number"]).with_columns(
            ((pl.col(score).rank("ordinal") - 1) * k // pl.len()).cast(pl.Int16).alias("b"))
        g = dd.group_by("b").agg(pl.col("registered").mean().alias("p")).sort("b")
        v = [float(r["p"]) * 100 for r in g.iter_rows(named=True)]
        log(f"  {k:2d} bins: " + " ".join(f"{x:.2f}" for x in v))

log("\n=== is the lead V just atypicality? profile by lead WITHIN atypicality quintile ===")
d2 = d.sort(["atyp", "serial_number"]).with_columns(
    ((pl.col("atyp").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("aq"))
for aq in range(5):
    sub = d2.filter(pl.col("aq") == aq)
    sub = sub.sort(["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal") - 1) * 10 // pl.len()).cast(pl.Int16).alias("b"))
    g = sub.group_by("b").agg(pl.col("registered").mean().alias("p")).sort("b")
    v = [float(r["p"]) * 100 for r in g.iter_rows(named=True)]
    log(f"  atyp Q{aq+1} (n={sub.height:,}): " + " ".join(f"{x:.1f}" for x in v))

log("\n=== correlation of |lead| with atypicality ===")
c = d.select(pl.corr(pl.col("topic_dkl").abs(), pl.col("atyp")).alias("r")).item()
log(f"  corr(|L|, A) = {c:.4f}")
c2 = d.select(pl.corr("topic_dkl", "atyp").alias("r")).item()
log(f"  corr(L, A)   = {c2:.4f}")


import json
out = {"n": int(d.height), "corr_absL_A": float(c), "corr_L_A": float(c2),
       "tie_share": float(n_tied / d.height), "bins": {}}
for score, key in (("topic_dkl", "lead"), ("atyp", "atyp")):
    dd = d.sort([score, "serial_number"]).with_columns(
        ((pl.col(score).rank("ordinal") - 1) * 20 // pl.len()).cast(pl.Int16).alias("b"))
    g = dd.group_by("b").agg(pl.col("registered").mean().alias("p"),
                             pl.len().alias("n")).sort("b")
    out["bins"][key] = [float(r["p"]) * 100 for r in g.iter_rows(named=True)]
strata = []
for aq in range(5):
    sub = d2.filter(pl.col("aq") == aq).sort(["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal") - 1) * 20 // pl.len()).cast(pl.Int16).alias("b"))
    g = sub.group_by("b").agg(pl.col("registered").mean().alias("p")).sort("b")
    strata.append([float(r["p"]) * 100 for r in g.iter_rows(named=True)])
out["lead_within_atyp_quintile"] = strata
(REPO / "paper" / "results" / "registration_shape.json").write_text(json.dumps(out, indent=1))
log("[done] registration_shape.json")
