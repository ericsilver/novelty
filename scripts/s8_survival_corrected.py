"""First-Section-8-gate survival by dKL with CORRECT status codes.

Authoritative code semantics (USPTO status code list):
  700      Registered (gate not yet faced / pending)
  701      Section 8 accepted
  702      Section 8 & 15 accepted
  703      Section 15 acknowledged
  704/705  Partial Section 8 (and 15) accepted
  710      Cancelled - Section 8   <- FAILED the gate
  711-716  Cancelled - other grounds (excluded)
  800      Renewed (passed Section 8 and Section 9)
  900      Expired (registered; not renewed at Section 9)

Clean cohort: registration year 2016-2018 -- exactly one Section 8 gate
(due years 5-6 + grace) has elapsed by the 2025 data snapshot, and the
Section 9 gate has not. PASS = {701,702,703,704,705}, FAIL = {710}.
Everything else in the cohort is counted and reported as excluded.

Outputs per class + pooled: P(pass | faced gate) by dKL quintile.
  paper/results/s8_corrected_summary.{csv,json}
  paper/results/s8_corrected_forest.png
  paper/results/s8_corrected_pooled.png
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

PASS = {"701", "702", "703", "704", "705", "800"}  # 800 = renewed -> passed S8
FAIL = {"710"}
REG_LO, REG_HI = 2016, 2018
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


def quintile_stats(df: pl.DataFrame, var: str, outcome: str) -> dict:
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
    if "q5" in d and "q1" in d:
        d["lift_q5_q1"] = d["q5"] - d["q1"]
        p1, n1, p5, n5 = d["q1"], d["q1_n"], d["q5"], d["q5_n"]
        d["lift_se"] = float(np.sqrt(p1*(1-p1)/n1 + p5*(1-p5)/n5))
    return d


def main() -> int:
    classes = sorted(p.stem.replace("surprise_class", "")
                     for p in PROC.glob("surprise_class???.parquet"))
    rows, pooled_parts = [], []
    for cls in classes:
        s = pl.read_parquet(
            PROC / f"surprise_class{cls}.parquet",
            columns=["serial_number", "year", "kl_vs_past", "kl_vs_future",
                     "n_ref_past", "n_ref_future", "n_terms"],
        ).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
        )
        t = pl.read_parquet(
            PROC / f"tm_class{cls}.parquet",
            columns=["serial_number", "registration_date", "status_code"],
        ).with_columns(
            pl.col("registration_date").str.slice(0, 4)
              .cast(pl.Int32, strict=False).alias("reg_year"),
        ).filter(pl.col("reg_year").is_between(REG_LO, REG_HI))
        j = s.join(t, on="serial_number", how="inner").with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
            pl.col("status_code").is_in(list(PASS)).alias("passed"),
            pl.col("status_code").is_in(list(FAIL)).alias("failed"),
        )
        gated = j.filter(pl.col("passed") | pl.col("failed"))
        n_excl = j.height - gated.height
        label = NICE_NAMES.get(cls, cls)
        if gated.height < MIN_GATED:
            rows.append({"cls": cls, "label": label, "n_gated": gated.height,
                         "skipped": True})
            print(f"  [skip] {cls} {label}: {gated.height} gated", file=sys.stderr, flush=True)
            continue
        st = quintile_stats(gated, "dkl", "passed")
        row = {"cls": cls, "label": label, "skipped": False,
               "n_reg_cohort": j.height, "n_gated": gated.height,
               "n_excluded_codes": n_excl,
               "p_pass": float(gated["passed"].cast(pl.Float64).mean())}
        row.update({f"pass_{k}": v for k, v in st.items()})
        rows.append(row)
        print(f"  [ok] {cls} {label}: gated={gated.height:,} pass={row['p_pass']:.3f} "
              f"lift={st['lift_q5_q1']:+.4f}", file=sys.stderr, flush=True)
        pooled_parts.append(gated.select("dkl", "kl_vs_past", "kl_vs_future",
                                         "passed").with_columns(pl.lit(cls).alias("cls")))
        del s, t, j, gated
        gc.collect()

    pl.from_dicts(rows).write_csv(RES / "s8_corrected_summary.csv")

    ok = [r for r in rows if not r.get("skipped")]
    ok.sort(key=lambda r: r["pass_lift_q5_q1"])

    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(ok))))
    ys = np.arange(len(ok))
    lifts = np.array([r["pass_lift_q5_q1"] for r in ok])
    ses = np.array([r["pass_lift_se"] for r in ok])
    ax.barh(ys, lifts, xerr=1.96 * ses,
            color=np.where(lifts > 0, "#2b6cb0", "#cc4444"), alpha=0.85,
            error_kw=dict(lw=0.8))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']} ({r['cls']})" for r in ok], fontsize=8)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel("P(Section 8 accepted | faced first gate): dKL Q5 minus Q1")
    ax.set_title("First-gate Section 8 survival lift by industry\n"
                 f"(registrations {REG_LO}-{REG_HI}; PASS=701/702/703/704/705, FAIL=710)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RES / "s8_corrected_forest.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    pooled_stats = {}
    if pooled_parts:
        pooled = pl.concat(pooled_parts)
        pooled_stats = {
            "n_gated": pooled.height,
            "p_pass": float(pooled["passed"].cast(pl.Float64).mean()),
            "by_dkl": quintile_stats(pooled, "dkl", "passed"),
            "by_pros": quintile_stats(pooled, "kl_vs_past", "passed"),
            "by_retr": quintile_stats(pooled, "kl_vs_future", "passed"),
        }
        # pooled binned curve figure
        fig, ax = plt.subplots(figsize=(7, 5))
        arr = pooled["dkl"].to_numpy()
        qs = np.linspace(0, 1, 21)
        cuts = np.quantile(arr, qs)
        bins = np.clip(np.searchsorted(cuts[1:-1], arr), 0, 19)
        g = pooled.with_columns(pl.Series("bin", bins)).group_by("bin").agg(
            pl.len().alias("n"),
            pl.col("passed").cast(pl.Float64).mean().alias("rate"),
            pl.col("dkl").mean().alias("mid")).sort("bin").to_pandas()
        z = 1.96
        se = np.sqrt(g["rate"] * (1 - g["rate"]) / g["n"])
        ax.fill_between(g["mid"], g["rate"] - z*se, g["rate"] + z*se,
                        color="#2b6cb0", alpha=0.18)
        ax.plot(g["mid"], g["rate"], "o-", color="#2b6cb0", lw=2, ms=4)
        ax.axhline(pooled_stats["p_pass"], ls="--", color="#444", lw=1)
        ax.set_xlabel("$\\Delta$KL (pros $-$ retr)")
        ax.set_ylabel("P(Section 8 accepted | faced first gate)")
        ax.set_title(f"Pooled first-gate Section 8 survival, all classes\n"
                     f"registrations {REG_LO}-{REG_HI}, n={pooled.height:,}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RES / "s8_corrected_pooled.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    (RES / "s8_corrected_summary.json").write_text(
        json.dumps({"per_class": rows, "pooled": pooled_stats}, indent=1, default=float))
    print(json.dumps(pooled_stats, indent=1, default=float), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
