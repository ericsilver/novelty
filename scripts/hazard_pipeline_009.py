"""Step 3 (class 009): prosecution telemetry and hazard analyses.

(a) Response latency -> gate survival: for filings with office actions,
    mean days from office action (CNRT/GNRT) to TEAS response (TROA);
    gate outcome by latency quintile, by dKL quintile, and jointly;
    split by attorney presence.
(b) Funded-flailing signatures: death-stage composition (prosecution
    abandonment vs gate cancellation) and filing->death duration, by
    attorney presence; among prosecution deaths, share that had responded
    to at least one office action before dying.
(c) Death-detection hazard by mark age for registered marks, by signed
    dKL tercile (topic and token): the spike-vs-smooth test at the
    Section 8 window (ages 5-7).

Output: paper/results/hazard_009.json
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

OA_CODES = ["CNRT", "GNRT", "CNFR", "GNFR"]  # non-final + final actions
RESP_CODES = ["TROA"]
ABN_CODES = ["ABN2", "ABN6", "ABN1", "ABN5", "ABN9", "ABNF"]
GATE_FAIL_CODES = ["C8.."]
PASS = {"701", "702", "703", "704", "705", "800"}
FAIL = {"710"}
SNAPSHOT = 20251231


def main() -> int:
    # ---- base: class 009 filings with scores + outcomes + attorney ----
    tm = pl.read_parquet(
        PROC / "tm_class009.parquet",
        columns=["serial_number", "filing_date", "registration_date", "status_code"],
    ).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8
         ).alias("registered"),
        pl.col("registration_date").str.slice(0, 4)
          .cast(pl.Int32, strict=False).alias("reg_year"),
        pl.col("status_code").is_in(list(PASS)).alias("st_pass"),
        pl.col("status_code").is_in(list(FAIL)).alias("st_fail"),
    )
    serials = tm["serial_number"]
    sset = set(serials.to_list())

    tok = pl.read_parquet(
        PROC / "surprise_class009.parquet",
        columns=["serial_number", "prospective_kl", "retrospective_kl",
                 "n_ref_prospective", "n_ref_retrospective", "n_terms"],
    ).filter(
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
    ).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("token_dkl"))
    topic = pl.read_parquet(PROC / "topic_surprise_class009.parquet").filter(
        pl.col("topic_dkl").is_finite()).select("serial_number", "topic_dkl")

    extras = pl.scan_parquet(PROC / "case_extras.parquet").filter(
        pl.col("serial_number").is_in(list(sset))).select(
        "serial_number", "attorney_name", "abandonment_date",
        "intent_to_use_in").collect()
    extras = extras.with_columns(
        (pl.col("attorney_name").fill_null("").str.len_chars() > 0
         ).alias("has_attorney"),
        (pl.col("intent_to_use_in") == "T").alias("itu"),
    )
    print(f"[extras] {extras.height:,} matched; attorney share "
          f"{float(extras['has_attorney'].mean()):.3f}", file=sys.stderr, flush=True)

    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("serial_number").is_in(list(sset))
        & (pl.col("date") > 19000000)).select(
        "serial_number", "code", "date").collect()
    print(f"[events] {ev.height:,} rows for class 009", file=sys.stderr, flush=True)

    def to_days(d: pl.Expr) -> pl.Expr:
        return d.cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)

    ev = ev.with_columns(to_days(pl.col("date")).alias("d")).drop_nulls("d")

    # ---- (a) response latency ----
    oa = ev.filter(pl.col("code").is_in(OA_CODES)).select(
        "serial_number", pl.col("d").alias("oa_d"))
    resp = ev.filter(pl.col("code").is_in(RESP_CODES)).select(
        "serial_number", pl.col("d").alias("resp_d"))
    pairs = oa.join(resp, on="serial_number", how="inner").filter(
        pl.col("resp_d") > pl.col("oa_d")).with_columns(
        (pl.col("resp_d") - pl.col("oa_d")).dt.total_days().alias("gap"))
    lat = pairs.group_by(["serial_number", "oa_d"]).agg(
        pl.col("gap").min().alias("latency")).group_by("serial_number").agg(
        pl.col("latency").mean().alias("mean_latency"),
        pl.len().alias("n_oa_responded"))
    n_oa = oa.group_by("serial_number").len().rename({"len": "n_oa"})
    print(f"[latency] {lat.height:,} serials with responded OAs",
          file=sys.stderr, flush=True)

    base = tm.join(tok.select("serial_number", "token_dkl"), on="serial_number",
                   how="inner").join(topic, on="serial_number", how="inner").join(
        extras.select("serial_number", "has_attorney", "itu", "abandonment_date"),
        on="serial_number", how="left").join(
        lat, on="serial_number", how="left").join(
        n_oa, on="serial_number", how="left").with_columns(
        pl.col("has_attorney").fill_null(False),
        pl.col("n_oa").fill_null(0))

    gate = base.filter(pl.col("reg_year").is_between(2016, 2018)
                       & (pl.col("st_pass") | pl.col("st_fail")))
    gl = gate.filter(pl.col("mean_latency").is_not_null())
    print(f"[gate] {gate.height:,} gated; {gl.height:,} with latency",
          file=sys.stderr, flush=True)

    def quintiles(frame, var, outcome):
        arr = frame[var].to_numpy()
        cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
        qi = np.searchsorted(cuts, arr)
        y = frame[outcome].cast(pl.Float64).to_numpy()
        d = {}
        for q in range(5):
            m = qi == q
            d[f"q{q+1}"] = float(y[m].mean()) if m.any() else None
            d[f"q{q+1}_n"] = int(m.sum())
        d["lift_q5_q1"] = d["q5"] - d["q1"]
        return d

    out = {"n_gate": gate.height, "n_gate_with_latency": gl.height}
    out["gate_by_latency"] = quintiles(gl, "mean_latency", "st_pass")
    out["gate_by_latency_attorney"] = quintiles(
        gl.filter(pl.col("has_attorney")), "mean_latency", "st_pass")
    out["gate_by_latency_self"] = quintiles(
        gl.filter(~pl.col("has_attorney")), "mean_latency", "st_pass")
    # joint: dkl gradient within fast/slow halves
    med = float(gl["mean_latency"].median())
    out["latency_median_days"] = med
    out["gate_by_tokendkl_fast"] = quintiles(
        gl.filter(pl.col("mean_latency") <= med), "token_dkl", "st_pass")
    out["gate_by_tokendkl_slow"] = quintiles(
        gl.filter(pl.col("mean_latency") > med), "token_dkl", "st_pass")

    # ---- (b) death stage + duration by attorney ----
    abn = ev.filter(pl.col("code").is_in(ABN_CODES)).group_by("serial_number").agg(
        pl.col("d").min().alias("abn_d"))
    c8 = ev.filter(pl.col("code").is_in(GATE_FAIL_CODES)).group_by(
        "serial_number").agg(pl.col("d").min().alias("c8_d"))
    bb = base.join(abn, on="serial_number", how="left").join(
        c8, on="serial_number", how="left").with_columns(
        to_days(pl.col("filing_date").cast(pl.Int64)).alias("file_d"))
    bb = bb.with_columns(
        pl.when(pl.col("abn_d").is_not_null()).then(pl.lit("prosecution"))
        .when(pl.col("c8_d").is_not_null()).then(pl.lit("gate"))
        .otherwise(pl.lit("alive_or_other")).alias("death_stage"),
        pl.coalesce(pl.col("abn_d"), pl.col("c8_d")).alias("death_d"))
    bb = bb.with_columns(
        ((pl.col("death_d") - pl.col("file_d")).dt.total_days() / 365.25
         ).alias("age_at_death"))
    died = bb.filter(pl.col("death_d").is_not_null()
                     & pl.col("year").is_between(1995, 2015))
    comp = {}
    for att in (True, False):
        sub = died.filter(pl.col("has_attorney") == att)
        responded = sub.filter(pl.col("death_stage") == "prosecution").join(
            lat, on="serial_number", how="left", suffix="_l")
        comp["attorney" if att else "self"] = {
            "n_deaths": sub.height,
            "share_prosecution": float((sub["death_stage"] == "prosecution").mean()),
            "share_gate": float((sub["death_stage"] == "gate").mean()),
            "median_age_at_death": float(sub["age_at_death"].median()),
            "prosec_deaths_responded_first": float(
                responded["mean_latency"].is_not_null().mean()),
        }
    out["death_by_attorney"] = comp

    # ---- (c) hazard by age, dkl terciles, registered marks ----
    regd = bb.filter(pl.col("registered") & pl.col("year").is_between(1995, 2015))
    haz = {}
    for var in ("topic_dkl", "token_dkl"):
        arr = regd[var].to_numpy()
        cuts = np.quantile(arr, [1/3, 2/3])
        ti = np.searchsorted(cuts, arr)
        age_death = regd["age_at_death"].to_numpy()
        file_d = regd["file_d"].cast(pl.Int32).to_numpy()  # days epoch
        snap = (np.datetime64("2025-12-31") - regd["file_d"].to_numpy()
                ).astype("timedelta64[D]").astype(float) / 365.25
        curves = {}
        for t in range(3):
            m = ti == t
            ad = age_death[m]
            cs = snap[m]
            hs = []
            for a in range(0, 15):
                at_risk = ((np.isnan(ad) | (ad >= a)) & (cs >= a)).sum()
                deaths = ((~np.isnan(ad)) & (ad >= a) & (ad < a + 1)).sum()
                hs.append(float(deaths / at_risk) if at_risk > 500 else None)
            curves[f"tercile_{t+1}"] = hs
        haz[var] = curves
    out["hazard_by_age"] = haz
    # spike ratio: gate window (5<=a<7) vs baseline (2<=a<4)
    spikes = {}
    for var, curves in haz.items():
        sp = {}
        for k, hs in curves.items():
            base_h = np.nanmean([h for h in hs[2:4] if h is not None])
            gate_h = np.nanmean([h for h in hs[5:7] if h is not None])
            sp[k] = {"baseline_h": base_h, "gate_h": gate_h,
                     "spike_ratio": gate_h / base_h if base_h else None}
        spikes[var] = sp
    out["spike_ratios"] = spikes

    (RES / "hazard_009.json").write_text(json.dumps(out, indent=1, default=float))
    for k in ("gate_by_latency", "gate_by_tokendkl_fast", "gate_by_tokendkl_slow"):
        d = out[k]
        print(f"  {k}: {[round(d[f'q{i}'],3) for i in range(1,6)]} "
              f"lift {d['lift_q5_q1']:+.3f}", file=sys.stderr, flush=True)
    print("  death_by_attorney:", json.dumps(out["death_by_attorney"], indent=1),
          file=sys.stderr, flush=True)
    print("  spike_ratios:", json.dumps(out["spike_ratios"], indent=1, default=float),
          file=sys.stderr, flush=True)
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
