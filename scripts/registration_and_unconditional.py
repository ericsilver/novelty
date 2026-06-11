"""Two analyses supporting the cross-industry section and the appendix.

Part A: per-industry registration shape. For every class, all clean
filings 1990-2018, outcome = registration_date populated. Reports dKL
quintile rates, the inverse-U depth (q3 minus mean(q1,q5)), and q5-q1.

Part B: the unconditional composite outcome. Filings 2013-2015 (whose
registrations, if any, land by ~2017 and face exactly one Section 8 gate
by the 2025 snapshot). Outcome = registered AND first gate passed
(status 701-705 or 800), measured unconditionally over all filings, vs
the conditional gate outcome on the same filings. Pooled all classes.

Outputs:
  paper/results/registration_by_industry.{csv,json}
  paper/results/registration_inverseU_forest.png
  paper/results/appendix_unconditional.png
  paper/results/appendix_unconditional.json
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

PASS = {"701", "702", "703", "704", "705", "800"}
FAIL = {"710"}
MIN_CLEAN = 5000

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

CLEAN = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
)


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
    nm = 0.5 * (d["q1_n"] + d["q5_n"])
    p3, p_t = d["q3"], 0.5 * (d["q1"] + d["q5"])
    d["inverseU_se"] = float(np.sqrt(p3*(1-p3)/d["q3_n"] + p_t*(1-p_t)/(2*nm)))
    return d


def load_class(cls: str, year_lo: int, year_hi: int) -> pl.DataFrame | None:
    sp = PROC / f"surprise_class{cls}.parquet"
    tp = PROC / f"tm_class{cls}.parquet"
    if not (sp.exists() and tp.exists()):
        return None
    s = pl.read_parquet(
        sp, columns=["serial_number", "year", "prospective_kl", "retrospective_kl",
                     "n_ref_prospective", "n_ref_retrospective", "n_terms"],
    ).filter(CLEAN & pl.col("prospective_kl").is_finite()
             & pl.col("retrospective_kl").is_finite()
             & pl.col("year").is_between(year_lo, year_hi))
    t = pl.read_parquet(
        tp, columns=["serial_number", "registration_date", "status_code"],
    ).with_columns(
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8
         ).alias("registered"),
        pl.col("status_code").is_in(list(PASS)).alias("st_pass"),
        pl.col("status_code").is_in(list(FAIL)).alias("st_fail"),
    )
    return s.join(t, on="serial_number", how="inner").with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"))


def main() -> int:
    # ---------- Part A: per-industry registration shape ----------
    rows = []
    for cls in sorted(NICE_NAMES):
        df = load_class(cls, 1990, 2018)
        if df is None or df.height < MIN_CLEAN:
            rows.append({"cls": cls, "label": NICE_NAMES[cls],
                         "n": 0 if df is None else df.height, "skipped": True})
            continue
        st = quintile_rates(df, "dkl", "registered")
        row = {"cls": cls, "label": NICE_NAMES[cls], "skipped": False,
               "n": df.height,
               "p_registered": float(df["registered"].cast(pl.Float64).mean())}
        row.update({f"reg_{k}": v for k, v in st.items()})
        rows.append(row)
        print(f"[A] {cls} {NICE_NAMES[cls]}: n={df.height:,} "
              f"invU={st['inverseU_depth']:+.4f} q5q1={st['lift_q5_q1']:+.4f}",
              file=sys.stderr, flush=True)
        del df
        gc.collect()

    pl.from_dicts(rows).write_csv(RES / "registration_by_industry.csv")
    (RES / "registration_by_industry.json").write_text(
        json.dumps(rows, indent=1, default=float))

    ok = [r for r in rows if not r.get("skipped")]
    ok.sort(key=lambda r: r["reg_inverseU_depth"])
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(ok))))
    ys = np.arange(len(ok))
    vals = np.array([r["reg_inverseU_depth"] for r in ok])
    ses = np.array([r["reg_inverseU_se"] for r in ok])
    ax.barh(ys, vals, xerr=1.96 * ses,
            color=np.where(vals > 0, "#2b6cb0", "#cc4444"), alpha=0.85,
            error_kw=dict(lw=0.8))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']} ({r['cls']})" for r in ok], fontsize=8)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel("Registration inverse-U depth: q3 rate minus mean(q1, q5)")
    ax.set_title("Registration inverse-U in dKL by industry (filings 1990-2018)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RES / "registration_inverseU_forest.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    n_pos = sum(1 for r in ok if r["reg_inverseU_depth"] > 0)
    n_sig = sum(1 for r in ok
                if r["reg_inverseU_depth"] > 1.96 * r["reg_inverseU_se"])
    print(f"[A] inverse-U positive in {n_pos}/{len(ok)}; significant in {n_sig}",
          file=sys.stderr, flush=True)

    # ---------- Part B: unconditional composite, pooled ----------
    parts = []
    for cls in sorted(NICE_NAMES):
        df = load_class(cls, 2013, 2015)
        if df is None or df.height == 0:
            continue
        parts.append(df.select("dkl", "prospective_kl", "retrospective_kl",
                               "registered", "st_pass", "st_fail"))
        del df
        gc.collect()
    pooled = pl.concat(parts).with_columns(
        (pl.col("registered") & pl.col("st_pass")).alias("uncond_success"))
    gated = pooled.filter(pl.col("registered") & (pl.col("st_pass") | pl.col("st_fail")))

    stats = {
        "n_filings": pooled.height,
        "n_registered": int(pooled["registered"].sum()),
        "n_gated": gated.height,
        "p_uncond_success": float(pooled["uncond_success"].cast(pl.Float64).mean()),
        "p_pass_given_gated": float(gated["st_pass"].cast(pl.Float64).mean()),
        "uncond_by_dkl": quintile_rates(pooled, "dkl", "uncond_success"),
        "registered_by_dkl": quintile_rates(pooled, "dkl", "registered"),
        "cond_by_dkl": quintile_rates(gated, "dkl", "st_pass"),
    }
    (RES / "appendix_unconditional.json").write_text(
        json.dumps(stats, indent=1, default=float))
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)},
                     indent=1), file=sys.stderr, flush=True)
    print("uncond_by_dkl:", {k: round(v, 4) for k, v in stats["uncond_by_dkl"].items()
                             if not k.endswith("_n")}, file=sys.stderr, flush=True)
    print("cond_by_dkl  :", {k: round(v, 4) for k, v in stats["cond_by_dkl"].items()
                             if not k.endswith("_n")}, file=sys.stderr, flush=True)

    # binned curves figure: unconditional (left) vs conditional (right)
    def curve(frame, outcome):
        arr = frame["dkl"].to_numpy()
        cuts = np.quantile(arr, np.linspace(0, 1, 21))
        bins = np.clip(np.searchsorted(cuts[1:-1], arr), 0, 19)
        g = frame.with_columns(pl.Series("bin", bins)).group_by("bin").agg(
            pl.len().alias("n"),
            pl.col(outcome).cast(pl.Float64).mean().alias("rate"),
            pl.col("dkl").mean().alias("mid")).sort("bin").to_pandas()
        g["se"] = np.sqrt(g["rate"] * (1 - g["rate"]) / g["n"])
        return g

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, (frame, outcome, title) in zip(axes, [
        (pooled, "uncond_success",
         "A. P(registered AND passed first gate)\nunconditional, all filings"),
        (gated, "st_pass",
         "B. P(passed first gate | registered and gated)\nconditional"),
    ]):
        g = curve(frame, outcome)
        ax.fill_between(g["mid"], g["rate"] - 1.96*g["se"], g["rate"] + 1.96*g["se"],
                        color="#2b6cb0", alpha=0.18)
        ax.plot(g["mid"], g["rate"], "o-", color="#2b6cb0", lw=2, ms=4)
        base = float(frame[outcome].cast(pl.Float64).mean())
        ax.axhline(base, ls="--", color="#444", lw=1)
        ax.set_xlabel("$\\Delta$KL (pros $-$ retr)")
        ax.set_ylabel("rate")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Composite vs conditional outcome, filings 2013-2015 pooled "
                 f"across classes (n={pooled.height:,}; gated n={gated.height:,})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "appendix_unconditional.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
