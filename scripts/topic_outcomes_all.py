"""Outcome analyses under topic-distribution scoring, all classes.

Re-estimates the paper's mark-level results with topic_dkl in place of
token dKL: first Section 8 gate survival (registrations 2016-2018),
registration completion (filings 1995-2018), debut EDGAR gradient, plus
the token-vs-topic comparison on identical samples.

Outputs: paper/results/topic_outcomes_all.{json,png}
         paper/results/topic_s8_forest.png
"""
from __future__ import annotations

import gc
import json
import os
import sys
SUFFIX = os.environ.get("TOPIC_SUFFIX", "")
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

PASS = {"701", "702", "703", "704", "705", "800"}
FAIL = {"710"}
MIN_GATED = 2000

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


def quintile_rates(df: pl.DataFrame, var: str, outcome: str) -> dict:
    arr = df[var].to_numpy()
    cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(cuts, arr)
    g = df.with_columns(pl.Series("q", qi)).group_by("q").agg(
        pl.len().alias("n"),
        pl.col(outcome).cast(pl.Float64).mean().alias("rate")).sort("q")
    d = {}
    for r in g.iter_rows(named=True):
        d[f"q{int(r['q'])+1}"] = r["rate"]
        d[f"q{int(r['q'])+1}_n"] = r["n"]
    d["lift_q5_q1"] = d["q5"] - d["q1"]
    d["inverseU_depth"] = d["q3"] - 0.5 * (d["q1"] + d["q5"])
    p1, n1, p5, n5 = d["q1"], d["q1_n"], d["q5"], d["q5_n"]
    d["lift_se"] = float(np.sqrt(p1*(1-p1)/n1 + p5*(1-p5)/n5))
    return d


def all_classes() -> list[str]:
    return sorted(p.stem.replace("topic_surprise_class", "")
                  for p in PROC.glob("topic_surprise_class???.parquet"))


def load_class(cls: str) -> pl.DataFrame | None:
    tp = PROC / f"topic_surprise_class{cls}{SUFFIX}.parquet"
    sp = PROC / f"surprise_class{cls}.parquet"
    mp = PROC / f"tm_class{cls}.parquet"
    if not (tp.exists() and sp.exists() and mp.exists()):
        return None
    topic = pl.read_parquet(tp).filter(pl.col("topic_dkl").is_finite())
    tok = pl.read_parquet(
        sp, columns=["serial_number", "n_terms", "prospective_kl",
                     "retrospective_kl", "n_ref_prospective", "n_ref_retrospective"],
    ).filter(
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
    ).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("token_dkl"))
    tm = pl.read_parquet(
        mp, columns=["serial_number", "owner_name", "filing_date",
                     "registration_date", "status_code"],
    ).with_columns(
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8
         ).alias("registered"),
        pl.col("registration_date").str.slice(0, 4)
          .cast(pl.Int32, strict=False).alias("reg_year"),
        pl.col("status_code").is_in(list(PASS)).alias("st_pass"),
        pl.col("status_code").is_in(list(FAIL)).alias("st_fail"),
    )
    j = topic.join(tok.select("serial_number", "n_terms", "token_dkl"),
                   on="serial_number", how="inner").filter(pl.col("n_terms") >= 3)
    return j.join(tm, on="serial_number", how="inner")


def main() -> int:
    classes = all_classes()
    print(f"[plan] {len(classes)} classes with topic scores", file=sys.stderr, flush=True)

    gate_parts, reg_parts = [], []
    per_class = []
    for cls in classes:
        d = load_class(cls)
        if d is None or d.height == 0:
            continue
        gate = d.filter(pl.col("reg_year").is_between(2016, 2018)
                        & (pl.col("st_pass") | pl.col("st_fail")))
        regp = d.filter(pl.col("year").is_between(1995, 2018))
        if gate.height >= MIN_GATED:
            row = {"cls": cls, "label": NICE_NAMES.get(cls, cls),
                   "n_gated": gate.height}
            st = quintile_rates(gate, "topic_dkl", "st_pass")
            row.update({f"gate_topic_{k}": v for k, v in st.items()})
            st2 = quintile_rates(gate, "token_dkl", "st_pass")
            row["gate_token_lift"] = st2["lift_q5_q1"]
            per_class.append(row)
        gate_parts.append(gate.select("st_pass", "topic_dkl", "token_dkl"))
        reg_parts.append(regp.select("registered", "topic_dkl", "token_dkl"))
        print(f"  {cls}: gate {gate.height:,} reg {regp.height:,}",
              file=sys.stderr, flush=True)
        del d, gate, regp
        gc.collect()

    gate_all = pl.concat(gate_parts); del gate_parts; gc.collect()
    reg_all = pl.concat(reg_parts); del reg_parts; gc.collect()

    out = {
        "n_gate": gate_all.height, "n_reg": reg_all.height,
        "gate_topic": quintile_rates(gate_all, "topic_dkl", "st_pass"),
        "gate_token_same_sample": quintile_rates(gate_all, "token_dkl", "st_pass"),
        "registration_topic": quintile_rates(reg_all, "topic_dkl", "registered"),
        "registration_token_same_sample": quintile_rates(reg_all, "token_dkl", "registered"),
        "per_class": per_class,
        "per_class_negative_topic": sum(1 for r in per_class
                                        if r["gate_topic_lift_q5_q1"] < 0),
        "per_class_total": len(per_class),
        "corr_token_topic_gate_sample": float(np.corrcoef(
            gate_all["token_dkl"].to_numpy(), gate_all["topic_dkl"].to_numpy())[0, 1]),
    }
    (RES / f"topic_outcomes_all{SUFFIX}.json").write_text(json.dumps(out, indent=1, default=float))
    for k in ("gate_topic", "gate_token_same_sample",
              "registration_topic", "registration_token_same_sample"):
        d = out[k]
        print(f"  {k}: lift {d['lift_q5_q1']:+.4f} invU {d['inverseU_depth']:+.4f}",
              file=sys.stderr, flush=True)
    print(f"  per-class topic gate lift negative: "
          f"{out['per_class_negative_topic']}/{out['per_class_total']}",
          file=sys.stderr, flush=True)

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    xs = np.arange(1, 6)
    for ax, key_t, key_k, title in [
        (axes[0], "gate_topic", "gate_token_same_sample",
         "First Section 8 gate (registrations 2016-2018)"),
        (axes[1], "registration_topic", "registration_token_same_sample",
         "Registration completion (filings 1995-2018)"),
    ]:
        for key, label, color in ((key_t, "topic dKL", "#2b6cb0"),
                                  (key_k, "token dKL", "#cc4444")):
            d = out[key]
            ax.plot(xs, [d[f"q{i}"] for i in xs], "o-", color=color,
                    label=f"{label} (lift {d['lift_q5_q1']*100:+.1f}pp)")
        ax.set_xticks(xs)
        ax.set_xlabel("dKL quintile")
        ax.set_ylabel("rate")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Mark-level outcomes under topic vs token scoring, identical samples")
    fig.tight_layout()
    fig.savefig(RES / f"topic_outcomes_all{SUFFIX}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    ok = sorted(per_class, key=lambda r: r["gate_topic_lift_q5_q1"])
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(ok))))
    ys = np.arange(len(ok))
    lifts = np.array([r["gate_topic_lift_q5_q1"] for r in ok])
    ses = np.array([r["gate_topic_lift_se"] for r in ok])
    ax.barh(ys, lifts, xerr=1.96 * ses,
            color=np.where(lifts > 0, "#2b6cb0", "#cc4444"), alpha=0.85,
            error_kw=dict(lw=0.8))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']} ({r['cls']})" for r in ok], fontsize=8)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel("Gate survival lift (topic dKL Q5 minus Q1)")
    ax.set_title("First-gate Section 8 survival lift by industry, topic scoring")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RES / f"topic_s8_forest{SUFFIX}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
