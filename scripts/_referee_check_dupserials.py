"""Referee check: multi-class duplication in event_gates_all pooled sample.

Rebuilds the exact event_gates_all.py sample per class but KEEPS serial_number,
then measures:
  - rows vs unique serials (duplication factor)
  - distribution of k = classes per serial in-sample
  - pooled lift Q5-Q1 as published (row level, naive binomial SE)
  - deduplicated lift: one row per serial, topic_dkl = mean across its classes
  - cluster-robust SE at serial level for the row-level estimate
Prints JSON to stdout. Does NOT touch paper/results.
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
REG_LO, REG_HI = 2002, 2018

CLASSES = [f"{i:03d}" for i in range(1, 46)]

c8 = pl.scan_parquet(PROC / "case_events.parquet").filter(
    (pl.col("code") == "C8..") & (pl.col("date") > 19000000)).select(
    "serial_number", "date").collect().with_columns(
    pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
    .alias("c8_d")).drop_nulls("c8_d").group_by("serial_number").agg(
    pl.col("c8_d").min())

parts = []
for cls in CLASSES:
    tp = PROC / f"topic_surprise_class{cls}.parquet"
    if not tp.exists():
        continue
    tm = pl.read_parquet(
        PROC / f"tm_class{cls}.parquet",
        columns=["serial_number", "registration_date"],
    ).filter(
        pl.col("registration_date").fill_null("").str.len_chars() >= 8
    ).with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("reg_d"),
        pl.col("registration_date").str.slice(0, 4)
        .cast(pl.Int32, strict=False).alias("reg_year"),
    ).filter(pl.col("reg_year").is_between(REG_LO, REG_HI)).drop_nulls("reg_d")
    topic = pl.read_parquet(tp).filter(
        pl.col("topic_dkl").is_finite()).select("serial_number", "topic_dkl")
    j = tm.join(topic, on="serial_number", how="inner").join(
        c8, on="serial_number", how="left").with_columns(
        ((pl.col("c8_d") - pl.col("reg_d")).dt.total_days() / 365.25).alias("c8_age")
    ).with_columns(
        ((pl.col("c8_age") >= 4.0) & (pl.col("c8_age") < 8.5))
        .fill_null(False).alias("failed1"),
    ).select("serial_number", "topic_dkl", "failed1")
    parts.append(j)
    print(f"  {cls}: {j.height:,}", file=sys.stderr, flush=True)
    del tm, topic, j
    gc.collect()

df = pl.concat(parts)
del parts; gc.collect()

out = {}
out["rows"] = df.height
uniq = df["serial_number"].n_unique()
out["unique_serials"] = uniq
out["dup_factor"] = df.height / uniq
kdist = (df.group_by("serial_number").len().rename({"len": "k"})
         .group_by("k").agg(pl.len().alias("n_serials"))
         .sort("k").to_dicts())
out["k_distribution"] = {str(r["k"]): r["n_serials"] for r in kdist}


def quint(arr, y):
    cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(cuts, arr)
    d = {}
    for q in range(5):
        m = qi == q
        d[f"q{q+1}"] = float(y[m].mean())
        d[f"q{q+1}_n"] = int(m.sum())
    d["lift"] = d["q5"] - d["q1"]
    p1, n1, p5, n5 = d["q1"], d["q1_n"], d["q5"], d["q5_n"]
    d["se_binom"] = float(np.sqrt(p1*(1-p1)/n1 + p5*(1-p5)/n5))
    return d, qi


arr = df["topic_dkl"].to_numpy()
y = df["failed1"].cast(pl.Float64).to_numpy()
row_q, qi = quint(arr, y)
out["row_level"] = row_q

# cluster-robust SE at serial level for the Q5-Q1 difference:
# treat estimator as p5_hat - p1_hat; variance = sum over serials of
# (contribution to p5 - contribution to p1) residuals.
ser = df["serial_number"].to_numpy()
n1 = row_q["q1_n"]; n5 = row_q["q5_n"]
w = np.zeros(len(y))
in1 = qi == 0; in5 = qi == 4
w[in5] = (y[in5] - row_q["q5"]) / n5
w[in1] = -(y[in1] - row_q["q1"]) / n1
gsum = pl.DataFrame({"s": ser, "w": w}).group_by("s").agg(pl.col("w").sum())["w"].to_numpy()
out["row_level_cluster_se"] = float(np.sqrt((gsum ** 2).sum()))

# dedup: one row per serial, mean topic_dkl across classes; failed1 identical per serial
dd = df.group_by("serial_number").agg(
    pl.col("topic_dkl").mean(), pl.col("failed1").first())
dq, _ = quint(dd["topic_dkl"].to_numpy(), dd["failed1"].cast(pl.Float64).to_numpy())
out["dedup_mean_dkl"] = dq

# dedup variant: random single class per serial (seeded)
dd2 = df.with_columns(pl.int_range(pl.len()).shuffle(seed=7).alias("r")).sort("r") \
        .unique(subset="serial_number", keep="first")
dq2, _ = quint(dd2["topic_dkl"].to_numpy(), dd2["failed1"].cast(pl.Float64).to_numpy())
out["dedup_random_class"] = dq2

print(json.dumps(out, indent=1, default=float))
