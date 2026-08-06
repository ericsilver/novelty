"""Event-dated gate analysis, all classes, all elapsed cohorts.

Per registration (reg years 2002-2018, first gate fully elapsed):
  failed1 = C8.. cancellation event at registration-age 4-8.5y
  gate2 (cohorts 2002-2013, both gates elapsed): among gate-1 passers
  with terminal status in {800, 710, 900}: failed2 = status 710 or 900
Covariates: topic_dkl (method), token_dkl, attorney presence (case_extras),
intent-to-use basis.

Outputs:
  paper/results/event_gates_all.json
  paper/results/event_gate_cohort_curve.png  (gate-1 lift by cohort year)
  paper/results/event_gate_forest.png        (per-class gate-1 lift)
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"

# Reference-window source. "topic" is the production per-calendar-year scorer;
# "rolling" is the per-filing scorer (scripts/rolling_rescore_all.py), whose
# output carries identical column names, so only the path changes.
SRC = os.environ.get("SURPRISE_SRC", "topic")
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
REG_LO, REG_HI = 2002, 2018
G2_HI = 2013


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
    d["lift_se"] = float(np.sqrt(p1*(1-p1)/max(n1,1) + p5*(1-p5)/max(n5,1)))
    return d


def main() -> int:
    print("[load] global C8 events + extras", file=sys.stderr, flush=True)
    c8 = pl.scan_parquet(PROC / "case_events.parquet").filter(
        (pl.col("code").is_in(["C8..", "C71T"])) & (pl.col("date") > 19000000)).select(
        "serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("c8_d")).drop_nulls("c8_d").group_by("serial_number").agg(
        pl.col("c8_d").min())
    extras = pl.scan_parquet(PROC / "case_extras.parquet").select(
        "serial_number", "attorney_name", "intent_to_use_in").collect(
    ).with_columns(
        (pl.col("attorney_name").fill_null("").str.len_chars() > 0
         ).alias("has_attorney"),
        (pl.col("intent_to_use_in") == "T").alias("itu"),
    ).select("serial_number", "has_attorney", "itu")
    print(f"  c8 {c8.height:,}; extras {extras.height:,}", file=sys.stderr, flush=True)

    parts = []
    for cls in sorted(NICE_NAMES):
        tp = PROC / f"{SRC}_surprise_class{cls}.parquet"
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
            ((pl.col("c8_d") - pl.col("reg_d")).dt.total_days() / 365.25)
            .alias("c8_age"),
        ).with_columns(
            ((pl.col("c8_age") >= 4.0) & (pl.col("c8_age") < 8.5))
            .fill_null(False).alias("failed1"),
        ).with_columns(pl.lit(cls).alias("cls")).select(
            "cls", "reg_year", "status_code", "topic_dkl", "token_dkl",
            "has_attorney", "itu", "failed1")
        parts.append(j)
        print(f"  {cls}: {j.height:,} regs, fail1 {float(j['failed1'].mean()):.3f}",
              file=sys.stderr, flush=True)
        del tm, topic, tok, j
        gc.collect()
    df = pl.concat(parts)
    del parts
    gc.collect()
    df.write_parquet(PROC / "event_gates_all.parquet")
    print(f"[pool] {df.height:,} registrations", file=sys.stderr, flush=True)

    out = {"n_reg": df.height, "p_fail1": float(df["failed1"].mean()),
           "reg_window": [REG_LO, REG_HI]}
    out["fail1_by_topic_dkl"] = quint(df, "topic_dkl", "failed1")
    dtok = df.drop_nulls("token_dkl")
    out["fail1_by_token_dkl"] = quint(dtok, "token_dkl", "failed1")

    # cohort curve
    cohort = {}
    for y in range(REG_LO, REG_HI + 1):
        sub = df.filter(pl.col("reg_year") == y)
        if sub.height >= 20000:
            q = quint(sub, "topic_dkl", "failed1")
            cohort[y] = {"n": sub.height, "p_fail1": float(sub["failed1"].mean()),
                         "lift": q["lift_q5_q1"], "se": q["lift_se"]}
    out["cohort_curve"] = cohort

    # attorney / basis strata
    for name, cond in (("attorney", pl.col("has_attorney")),
                       ("self", ~pl.col("has_attorney")),
                       ("itu", pl.col("itu")),
                       ("use_based", ~pl.col("itu"))):
        sub = df.filter(cond)
        out[f"fail1_{name}"] = {"n": sub.height,
                                "p_fail1": float(sub["failed1"].mean()),
                                **quint(sub, "topic_dkl", "failed1")}

    # per-class forest
    per_class = []
    for cls in sorted(NICE_NAMES):
        sub = df.filter(pl.col("cls") == cls)
        if sub.height < 5000:
            continue
        q = quint(sub, "topic_dkl", "failed1")
        per_class.append({"cls": cls, "label": NICE_NAMES[cls], "n": sub.height,
                          "p_fail1": float(sub["failed1"].mean()),
                          "lift": q["lift_q5_q1"], "se": q["lift_se"]})
    out["per_class"] = per_class
    out["per_class_positive"] = sum(1 for r in per_class if r["lift"] > 0)
    out["per_class_total"] = len(per_class)

    # gate 2 conditional (cohorts 2002-2013)
    g2pool = df.filter((pl.col("reg_year") <= G2_HI) & ~pl.col("failed1")
                       & pl.col("status_code").is_in(["800", "710", "900"])
                       ).with_columns(
        pl.col("status_code").is_in(["710", "900"]).alias("failed2"))
    out["gate2"] = {"n": g2pool.height, "p_fail2": float(g2pool["failed2"].mean()),
                    "by_topic_dkl": quint(g2pool, "topic_dkl", "failed2")}

    (RES / "event_gates_all.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in
                      ("p_fail1", "per_class_positive", "per_class_total")},
                     indent=1), file=sys.stderr, flush=True)
    print("fail1 pooled:", {k: round(v, 4) for k, v in out["fail1_by_topic_dkl"].items()
                            if not k.endswith("_n")}, file=sys.stderr, flush=True)
    print("gate2:", {k: round(v, 4) for k, v in out["gate2"]["by_topic_dkl"].items()
                     if not k.endswith("_n")}, file=sys.stderr, flush=True)
    for name in ("attorney", "self", "itu", "use_based"):
        d = out[f"fail1_{name}"]
        print(f"fail1_{name}: n={d['n']:,} base={d['p_fail1']:.3f} "
              f"lift={d['lift_q5_q1']:+.4f}", file=sys.stderr, flush=True)

    # ---- figures ----
    ys = sorted(cohort)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lifts = [cohort[y]["lift"] * 100 for y in ys]
    ses = [cohort[y]["se"] * 100 for y in ys]
    ax.errorbar(ys, lifts, yerr=[1.96 * s for s in ses], fmt="o-",
                color="#2b6cb0", lw=1.6, capsize=2)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("Registration cohort year")
    ax.set_ylabel("Gate-1 failure lift, pp\n(topic $\\Delta$KL Q5 $-$ Q1)")
    ax.set_title("First Section 8 gate: forward-leaning penalty by cohort,\n"
                 "event-dated, all classes pooled")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "event_gate_cohort_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    pc = sorted(per_class, key=lambda r: r["lift"])
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(pc))))
    yy = np.arange(len(pc))
    lifts = np.array([r["lift"] for r in pc])
    ses = np.array([r["se"] for r in pc])
    ax.barh(yy, lifts, xerr=1.96 * ses,
            color=np.where(lifts > 0, "#cc4444", "#2b6cb0"), alpha=0.85,
            error_kw=dict(lw=0.8))
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{r['label']} ({r['cls']})" for r in pc], fontsize=8)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel("P(failed first gate): topic $\\Delta$KL Q5 minus Q1")
    ax.set_title("Event-dated first-gate failure lift by industry,\n"
                 "registrations 2002-2018")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RES / "event_gate_forest.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
