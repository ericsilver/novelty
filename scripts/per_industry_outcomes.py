"""Per-industry and cross-industry outcome analysis with proper conditioning.

For every NICE class with a surprise parquet:
  - three-panel figure (P(reached registration), P(alive 5y | registration),
    P(owner ever in SEC EDGAR | registration)) by dKL / pros / retr quantile
  - metrics row: n, registration rate, conditional renewal rate, renewal by
    dKL quintile, q5-q1 renewal lift, EDGAR lift

Cross-industry outputs:
  - per_industry_summary.{csv,json}
  - forest plot of per-class conditional-renewal lift (dKL q5 - q1)
  - pooled all-class three-panel figure ("the entire corpus in one go")

Outputs under paper/results/per_industry/ and paper/results/.
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
OUTDIR = RES / "per_industry"

MIN_CLEAN = 5000  # minimum clean filings for a per-class figure

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


def load_class(cls: str, sec: pl.DataFrame) -> pl.DataFrame | None:
    sp = PROC / f"surprise_class{cls}.parquet"
    tp = PROC / f"tm_class{cls}.parquet"
    if not (sp.exists() and tp.exists()):
        return None
    s = pl.read_parquet(
        sp, columns=["serial_number", "year", "prospective_kl", "retrospective_kl",
                     "n_ref_prospective", "n_ref_retrospective", "n_terms"],
    ).filter(
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
        & pl.col("year").is_between(1990, 2020)
    )
    t = pl.read_parquet(
        tp, columns=["serial_number", "owner_name", "status_code"],
    ).with_columns(
        pl.col("status_code").str.slice(0, 1).alias("status_bucket"),
    ).with_columns(
        ((pl.col("status_bucket") == "6") | (pl.col("status_bucket") == "8")
         ).alias("reached_registration"),
        (pl.col("status_bucket") == "6").alias("renewed"),
    )
    j = s.join(t, on="serial_number", how="inner")
    j = j.join(sec, on="owner_name", how="left").with_columns(
        pl.col("in_sec").fill_null(False),
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
    ).select("dkl", "prospective_kl", "retrospective_kl",
             "reached_registration", "renewed", "in_sec")
    del s, t
    gc.collect()
    return j


def quantile_curve(df: pl.DataFrame, var: str, outcome: str, n_bins: int = 20) -> pl.DataFrame:
    arr = df[var].to_numpy()
    qs = np.linspace(0, 1, n_bins + 1)
    cuts = np.quantile(arr, qs)
    bin_idx = np.clip(np.searchsorted(cuts[1:-1], arr), 0, n_bins - 1)
    g = df.with_columns(pl.Series("bin", bin_idx)).group_by("bin").agg([
        pl.len().alias("n"),
        pl.col(outcome).cast(pl.Float64).mean().alias("rate"),
        pl.col(var).mean().alias("mid"),
    ]).sort("bin")
    p = g["rate"].to_numpy(); n = g["n"].to_numpy().astype(np.float64)
    z = 1.96; denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return g.with_columns(pl.Series("lo", centre - half), pl.Series("hi", centre + half))


def quintile_stats(df: pl.DataFrame, outcome: str) -> dict:
    arr = df["dkl"].to_numpy()
    cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(cuts, arr)
    g = df.with_columns(pl.Series("q", qi)).group_by("q").agg([
        pl.len().alias("n"),
        pl.col(outcome).cast(pl.Float64).mean().alias("rate"),
    ]).sort("q")
    rates = {int(r["q"]): (r["rate"], r["n"]) for r in g.iter_rows(named=True)}
    out = {}
    for q in range(5):
        rate, n = rates.get(q, (None, 0))
        out[f"q{q+1}"] = rate
        out[f"q{q+1}_n"] = n
    if out["q5"] is not None and out["q1"] is not None:
        out["lift_q5_q1"] = out["q5"] - out["q1"]
        n1, n5 = out["q1_n"], out["q5_n"]
        p1, p5 = out["q1"], out["q5"]
        se = float(np.sqrt(p1 * (1 - p1) / n1 + p5 * (1 - p5) / n5))
        out["lift_se"] = se
    return out


def render_class(cls: str, df: pl.DataFrame, label: str, out_png: Path) -> None:
    reg = df.filter(pl.col("reached_registration"))
    axes_vars = [("dkl", "$\\Delta$KL", "#2b6cb0"),
                 ("prospective_kl", "Prospective KL", "#cc4444"),
                 ("retrospective_kl", "Retrospective KL", "#229922")]
    panels = [
        (df,  "reached_registration", "A. P(reached registration)"),
        (reg, "renewed",              "B. P(survived 5y | registration)"),
        (reg, "in_sec",               "C. P(in SEC EDGAR | registration)"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (frame, outcome, title) in zip(axs, panels):
        baseline = frame[outcome].cast(pl.Float64).mean()
        for var, var_label, color in axes_vars:
            g = quantile_curve(frame, var, outcome)
            pdf = g.to_pandas()
            ax.plot(pdf["mid"], pdf["rate"], "-o", color=color, lw=2, ms=4, label=var_label)
            ax.fill_between(pdf["mid"], pdf["lo"], pdf["hi"], color=color, alpha=0.16)
        ax.axhline(baseline, ls="--", color="#444", lw=1, label=f"mean={baseline:.3f}")
        ax.set_xlabel("KL value (nats)"); ax.set_ylabel("rate")
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{label} (class {cls}); n={df.height:,}, registered n={reg.height:,}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sec = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name").unique().with_columns(pl.lit(True).alias("in_sec"))

    classes = sorted(p.stem.replace("surprise_class", "")
                     for p in PROC.glob("surprise_class???.parquet")
                     if p.stem.replace("surprise_class", "").isdigit())
    print(f"[plan] {len(classes)} classes with surprise scores", file=sys.stderr, flush=True)

    rows = []
    pooled_parts = []
    for cls in classes:
        label = NICE_NAMES.get(cls, f"class {cls}")
        df = load_class(cls, sec)
        if df is None or df.height < MIN_CLEAN:
            n = 0 if df is None else df.height
            print(f"  [skip] {cls} {label}: {n} clean filings", file=sys.stderr, flush=True)
            rows.append({"cls": cls, "label": label, "n_clean": n, "skipped": True})
            continue
        reg = df.filter(pl.col("reached_registration"))
        row = {
            "cls": cls, "label": label, "skipped": False,
            "n_clean": df.height, "n_registered": reg.height,
            "p_registered": float(df["reached_registration"].cast(pl.Float64).mean()),
            "p_renewed_given_reg": float(reg["renewed"].cast(pl.Float64).mean()),
            "p_sec_given_reg": float(reg["in_sec"].cast(pl.Float64).mean()),
        }
        ren = quintile_stats(reg, "renewed")
        row.update({f"renew_{k}": v for k, v in ren.items()})
        sec_q = quintile_stats(reg, "in_sec")
        row.update({f"sec_{k}": v for k, v in sec_q.items()})
        regq = quintile_stats(df, "reached_registration")
        row.update({f"reg_{k}": v for k, v in regq.items()})
        rows.append(row)
        print(f"  [ok] {cls} {label}: n={df.height:,} renew_lift={ren.get('lift_q5_q1'):+.4f}",
              file=sys.stderr, flush=True)
        render_class(cls, df, label, OUTDIR / f"outcomes_class{cls}.png")
        pooled_parts.append(df.with_columns(pl.lit(cls).alias("cls")))
        del df, reg
        gc.collect()

    # ----- summary table -----
    summary = pl.from_dicts(rows)
    summary.write_csv(RES / "per_industry_summary.csv")
    (RES / "per_industry_summary.json").write_text(json.dumps(rows, indent=1, default=float))

    ok = [r for r in rows if not r.get("skipped") and r.get("renew_lift_q5_q1") is not None]
    ok.sort(key=lambda r: r["renew_lift_q5_q1"])

    # ----- forest plot of conditional-renewal lift -----
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(ok))))
    ys = np.arange(len(ok))
    lifts = np.array([r["renew_lift_q5_q1"] for r in ok])
    ses = np.array([r.get("renew_lift_se", 0.0) for r in ok])
    ax.barh(ys, lifts, xerr=1.96 * ses, color=np.where(lifts > 0, "#2b6cb0", "#cc4444"),
            alpha=0.85, error_kw=dict(lw=0.8))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']} ({r['cls']})" for r in ok], fontsize=8)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel("P(survived 5y | registration): dKL Q5 minus Q1")
    ax.set_title("Conditional-renewal lift by industry (top vs bottom dKL quintile)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RES / "per_industry_renewal_forest.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ----- pooled all-class figure -----
    if pooled_parts:
        pooled = pl.concat(pooled_parts)
        del pooled_parts
        gc.collect()
        print(f"[pooled] {pooled.height:,} filings across {len(ok)} classes",
              file=sys.stderr, flush=True)
        render_class("ALL", pooled.drop("cls"), "All classes pooled",
                     RES / "pooled_outcomes_all_classes.png")
        reg = pooled.filter(pl.col("reached_registration"))
        pooled_stats = {
            "n_clean": pooled.height,
            "n_registered": reg.height,
            "p_registered": float(pooled["reached_registration"].cast(pl.Float64).mean()),
            "p_renewed_given_reg": float(reg["renewed"].cast(pl.Float64).mean()),
            "p_sec_given_reg": float(reg["in_sec"].cast(pl.Float64).mean()),
            "renew": quintile_stats(reg, "renewed"),
            "sec": quintile_stats(reg, "in_sec"),
            "registration": quintile_stats(pooled, "reached_registration"),
        }
        (RES / "pooled_outcomes_all_classes.json").write_text(
            json.dumps(pooled_stats, indent=1, default=float))
        print(json.dumps(pooled_stats["renew"], indent=1, default=float),
              file=sys.stderr, flush=True)

    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
