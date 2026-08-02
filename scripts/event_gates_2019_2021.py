"""Gate-cohort replication: registrations 2019-2021.

Identical pipeline to event_gates_all.py (failed1 = C8.. cancellation at
registration-age 4.0-8.5y), re-run on the post-2018 cohorts to test whether the
forward-leaning first-gate penalty persists beyond the 2016-2018 window that
anchors the paper.

Adds a censoring diagnostic: the first Section 8 gate falls at registration-age
~6 (5th-6th anniversary, grace to 6.5). With C8 events observed only through the
data cut, a cohort is "fully elapsed" only if reg_date + 6.5y <= cut. We report,
per cohort, the base failure rate and the share of the gate window that is
observable, so a depressed base rate is read as censoring, not as a real drop.

Outputs:
  paper/results/event_gates_2019_2021.json
  paper/results/event_gate_cohort_curve_extended.png  (2002-2021, 2019-2021 flagged)
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

NICE_NAMES = {
    "001": "Chemicals", "002": "Paints", "003": "Cosmetics & Cleaning",
    "004": "Lubricants & Fuels", "005": "Pharmaceuticals", "006": "Metal Goods",
    "007": "Machinery", "008": "Hand Tools", "009": "Software & Electronics",
    "010": "Medical Apparatus", "011": "Lighting & Heating", "012": "Vehicles",
    "013": "Firearms", "014": "Jewelry", "015": "Musical Instruments",
    "016": "Paper & Printed Goods", "017": "Rubber & Plastics", "018": "Leather Goods",
    "019": "Building Materials", "020": "Furniture", "021": "Household Utensils",
    "022": "Cordage & Fibers", "023": "Yarns & Threads", "024": "Textiles",
    "025": "Clothing & Footwear", "026": "Lace & Embroidery", "027": "Carpets",
    "028": "Games & Sporting Goods", "029": "Meats & Processed Foods",
    "030": "Staple Foods", "031": "Agricultural Products", "032": "Beer & Soft Drinks",
    "033": "Alcoholic Beverages", "034": "Tobacco", "035": "Advertising & Retail",
    "036": "Insurance & Finance", "037": "Construction & Repair",
    "038": "Telecommunications", "039": "Transport & Storage",
    "040": "Material Treatment", "041": "Education & Entertainment",
    "042": "Scientific & Tech Services", "043": "Hotels & Restaurants",
    "044": "Medical & Beauty Services", "045": "Legal & Personal Services",
}
REG_LO, REG_HI = 2019, 2021


def quint(frame: pl.DataFrame, col: str, outc: str) -> dict:
    arr = frame[col].to_numpy()
    cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(cuts, arr)
    y = frame[outc].cast(pl.Float64).to_numpy()
    d = {}
    for q in range(5):
        m = qi == q
        d[f"q{q+1}"] = float(y[m].mean()) if m.any() else None
        d[f"q{q+1}_n"] = int(m.sum())
    d["lift_q5_q1"] = d["q5"] - d["q1"]
    p1, n1, p5, n5 = d["q1"], d["q1_n"], d["q5"], d["q5_n"]
    d["lift_se"] = float(np.sqrt(p1*(1-p1)/max(n1, 1) + p5*(1-p5)/max(n5, 1)))
    return d


def main() -> int:
    print("[load] C8 events + extras", file=sys.stderr, flush=True)
    c8raw = pl.scan_parquet(PROC / "case_events.parquet").filter(
        (pl.col("code") == "C8..") & (pl.col("date") > 19000000)).select(
        "serial_number", "date").collect()
    cut_int = int(c8raw["date"].max())
    cut = pl.Series([str(cut_int)]).str.strptime(pl.Date, "%Y%m%d")[0]
    print(f"  data cut (max C8 date): {cut}", file=sys.stderr, flush=True)
    c8 = c8raw.with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("c8_d")).drop_nulls("c8_d").group_by("serial_number").agg(
        pl.col("c8_d").min())
    extras = pl.scan_parquet(PROC / "case_extras.parquet").select(
        "serial_number", "attorney_name", "intent_to_use_in").collect(
    ).with_columns(
        (pl.col("attorney_name").fill_null("").str.len_chars() > 0).alias("has_attorney"),
        (pl.col("intent_to_use_in") == "T").alias("itu"),
    ).select("serial_number", "has_attorney", "itu")

    parts = []
    for cls in sorted(NICE_NAMES):
        tp = PROC / f"topic_surprise_class{cls}.parquet"
        if not tp.exists():
            continue
        tm = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "registration_date", "status_code"],
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
        tok = pl.read_parquet(
            PROC / f"surprise_class{cls}.parquet",
            columns=["serial_number", "kl_vs_past", "kl_vs_future",
                     "n_terms", "n_ref_past", "n_ref_future"],
        ).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
        ).with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("token_dkl")
        ).select("serial_number", "token_dkl")
        j = tm.join(topic, on="serial_number", how="inner").join(
            tok, on="serial_number", how="left").join(
            c8, on="serial_number", how="left").join(
            extras, on="serial_number", how="left").with_columns(
            pl.col("has_attorney").fill_null(False),
            pl.col("itu").fill_null(False),
            ((pl.col("c8_d") - pl.col("reg_d")).dt.total_days() / 365.25).alias("c8_age"),
            ((cut - pl.col("reg_d")).dt.total_days() / 365.25).alias("obs_age"),
        ).with_columns(
            ((pl.col("c8_age") >= 4.0) & (pl.col("c8_age") < 8.5))
            .fill_null(False).alias("failed1"),
            # fully-elapsed: the entire 4.0-6.5 core gate window is observable
            (pl.col("obs_age") >= 6.5).alias("elapsed"),
        ).with_columns(pl.lit(cls).alias("cls")).select(
            "cls", "reg_year", "status_code", "topic_dkl", "token_dkl",
            "has_attorney", "itu", "failed1", "obs_age", "elapsed")
        parts.append(j)
        print(f"  {cls}: {j.height:,} regs", file=sys.stderr, flush=True)
        del tm, topic, tok, j
        gc.collect()
    df = pl.concat(parts)
    del parts
    gc.collect()
    print(f"[pool] {df.height:,} registrations 2019-2021", file=sys.stderr, flush=True)

    out = {"reg_window": [REG_LO, REG_HI], "data_cut": str(cut),
           "n_reg": df.height, "p_fail1": float(df["failed1"].mean())}

    # per-cohort base rate, lift, observability
    cohort = {}
    for y in range(REG_LO, REG_HI + 1):
        sub = df.filter(pl.col("reg_year") == y)
        q = quint(sub, "topic_dkl", "failed1")
        cohort[y] = {
            "n": sub.height,
            "p_fail1": float(sub["failed1"].mean()),
            "lift": q["lift_q5_q1"], "se": q["lift_se"],
            "median_obs_age": float(sub["obs_age"].median()),
            "share_elapsed": float(sub["elapsed"].mean()),
        }
    out["cohort_curve"] = cohort

    # primary clean replication: only fully-elapsed registrations (gate window observed)
    elapsed = df.filter(pl.col("elapsed"))
    out["elapsed_only"] = {
        "n": elapsed.height,
        "p_fail1": float(elapsed["failed1"].mean()),
        "by_topic_dkl": quint(elapsed, "topic_dkl", "failed1"),
    }
    detok = elapsed.drop_nulls("token_dkl")
    out["elapsed_only"]["by_token_dkl"] = quint(detok, "token_dkl", "failed1")

    # strata on the elapsed subset
    for name, cond in (("attorney", pl.col("has_attorney")),
                       ("self", ~pl.col("has_attorney")),
                       ("itu", pl.col("itu")),
                       ("use_based", ~pl.col("itu"))):
        sub = elapsed.filter(cond)
        if sub.height >= 1000:
            out[f"elapsed_fail1_{name}"] = {
                "n": sub.height, "p_fail1": float(sub["failed1"].mean()),
                **quint(sub, "topic_dkl", "failed1")}

    # per-class forest on elapsed subset
    per_class = []
    for cls in sorted(NICE_NAMES):
        sub = elapsed.filter(pl.col("cls") == cls)
        if sub.height < 3000:
            continue
        q = quint(sub, "topic_dkl", "failed1")
        per_class.append({"cls": cls, "label": NICE_NAMES[cls], "n": sub.height,
                          "p_fail1": float(sub["p_fail1" if False else "failed1"].mean()),
                          "lift": q["lift_q5_q1"], "se": q["lift_se"]})
    out["per_class"] = per_class
    out["per_class_positive"] = sum(1 for r in per_class if r["lift"] > 0)
    out["per_class_total"] = len(per_class)

    (RES / "event_gates_2019_2021.json").write_text(json.dumps(out, indent=1, default=float))

    # ---- console summary ----
    print("\n=== REPLICATION 2019-2021 ===", file=sys.stderr)
    print(f"data cut {cut}", file=sys.stderr)
    for y, v in cohort.items():
        print(f"  cohort {y}: n={v['n']:,} base_fail1={v['p_fail1']:.3f} "
              f"lift={v['lift']:+.4f}+/-{1.96*v['se']:.4f} "
              f"median_obs_age={v['median_obs_age']:.2f}y "
              f"share_gate_elapsed={v['share_elapsed']:.2f}", file=sys.stderr)
    e = out["elapsed_only"]
    print(f"\nELAPSED-ONLY (clean) n={e['n']:,} base={e['p_fail1']:.3f}", file=sys.stderr)
    print("  topic dKL fail1:",
          {k: round(v, 4) for k, v in e["by_topic_dkl"].items()
           if k.startswith("q") and not k.endswith("_n")}, file=sys.stderr)
    print(f"  lift q5-q1 = {e['by_topic_dkl']['lift_q5_q1']:+.4f} "
          f"+/- {1.96*e['by_topic_dkl']['lift_se']:.4f}", file=sys.stderr)
    print(f"  token dKL lift = {e['by_token_dkl']['lift_q5_q1']:+.4f} "
          f"+/- {1.96*e['by_token_dkl']['lift_se']:.4f}", file=sys.stderr)
    print(f"  per-class positive: {out['per_class_positive']}/{out['per_class_total']}",
          file=sys.stderr)
    for name in ("attorney", "self", "itu", "use_based"):
        k = f"elapsed_fail1_{name}"
        if k in out:
            d = out[k]
            print(f"  {name}: n={d['n']:,} base={d['p_fail1']:.3f} "
                  f"lift={d['lift_q5_q1']:+.4f}", file=sys.stderr)

    # ---- extended cohort curve figure (prior 2002-2018 + new 2019-2021) ----
    prior = json.load(open(RES / "event_gates_all.json"))["cohort_curve"]
    ys_p = sorted(int(y) for y in prior)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(ys_p, [prior[str(y)]["lift"]*100 for y in ys_p],
                yerr=[1.96*prior[str(y)]["se"]*100 for y in ys_p],
                fmt="o-", color="#2b6cb0", lw=1.6, capsize=2, label="2002-2018 (paper)")
    ys_n = [y for y in cohort if cohort[y]["share_elapsed"] >= 0.5]
    ys_c = [y for y in cohort if cohort[y]["share_elapsed"] < 0.5]
    if ys_n:
        ax.errorbar(ys_n, [cohort[y]["lift"]*100 for y in ys_n],
                    yerr=[1.96*cohort[y]["se"]*100 for y in ys_n],
                    fmt="s-", color="#cc4444", lw=1.8, capsize=3,
                    label="2019-2021 replication (gate elapsed)")
    if ys_c:
        ax.errorbar(ys_c, [cohort[y]["lift"]*100 for y in ys_c],
                    yerr=[1.96*cohort[y]["se"]*100 for y in ys_c],
                    fmt="s", color="#cc4444", mfc="white", capsize=3,
                    label="2019-2021 (gate not yet elapsed; censored)")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("Registration cohort year")
    ax.set_ylabel("Gate-1 failure lift, pp\n(topic $\\Delta$KL Q5 $-$ Q1)")
    ax.set_title("First Section 8 gate: forward-leaning penalty by cohort,\n"
                 "event-dated, all classes pooled (2019-2021 replication appended)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "event_gate_cohort_curve_extended.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
