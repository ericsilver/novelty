"""Per-class burn-in optimization.

The thin-past contamination signature is a positive bias in mean signed
dKL: with few reference filings behind it, a filing looks spuriously
unusual against its past. As reference mass accumulates, the per-year
mean of signed dKL settles to a stable long-run level. The burn-in end
for a class is the first year from which the per-year mean stays inside
a stability band around the class's long-run level.

Criterion:
  ref(c)   = median of per-year mean dKL over 2005-2015
  band(c)  = max(0.02 nats, 3 * sd of per-year means over 2005-2015)
  burnin_end(c) = first year y (scanning from the class's first scored
                  year) such that |mean(y') - ref| <= band for ALL
                  y' in {y, y+1, y+2}  (three consecutive stable years)

Also reports, per class: yearly volume in the early window and a
slow-moving proxy (stable-period mean prospective KL), and correlates
burn-in length with both.

Outputs:
  paper/results/burnin_by_class.{csv,json}
  paper/results/burnin_by_class.png        (per-class burn-in year)
  paper/results/burnin_convergence_examples.png
"""
from __future__ import annotations

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

STABLE_LO, STABLE_HI = 2005, 2015
BAND_FLOOR = 0.02
CONSEC = 3

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


def main() -> int:
    rows = []
    curves = {}
    for cls in sorted(NICE_NAMES):
        sp = PROC / f"surprise_class{cls}.parquet"
        if not sp.exists():
            continue
        # deliberately NO n_ref floor: the burn-in is about thin references
        df = pl.read_parquet(
            sp, columns=["year", "prospective_kl", "retrospective_kl",
                         "n_ref_prospective", "n_terms"],
        ).filter(
            (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
            & pl.col("year").is_between(1985, 2019)
        ).with_columns(
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"))
        if df.height < 5000:
            rows.append({"cls": cls, "label": NICE_NAMES[cls], "skipped": True,
                         "n": df.height})
            continue
        g = df.group_by("year").agg(
            pl.len().alias("n"),
            pl.col("dkl").mean().alias("mean_dkl"),
            pl.col("prospective_kl").mean().alias("mean_pros"),
            pl.col("n_ref_prospective").mean().alias("mean_nref"),
        ).sort("year").filter(pl.col("n") >= 200)
        years = g["year"].to_list()
        means = dict(zip(years, g["mean_dkl"].to_list()))

        stable = [means[y] for y in years if STABLE_LO <= y <= STABLE_HI]
        if len(stable) < 6:
            rows.append({"cls": cls, "label": NICE_NAMES[cls], "skipped": True,
                         "n": df.height})
            continue
        ref = float(np.median(stable))
        band = max(BAND_FLOOR, 3.0 * float(np.std(stable)))

        burnin_end = None
        for i, y in enumerate(years):
            ok = True
            for k in range(CONSEC):
                yk = y + k
                if yk not in means or abs(means[yk] - ref) > band:
                    ok = False
                    break
            if ok:
                burnin_end = y
                break

        first_year = years[0]
        vol_early = float(np.mean([n for yy, n in zip(years, g["n"].to_list())
                                   if yy <= first_year + 4]))
        pros_stable = float(np.mean(
            [p for yy, p in zip(years, g["mean_pros"].to_list())
             if STABLE_LO <= yy <= STABLE_HI]))
        bias_1990 = means.get(1990)
        rows.append({
            "cls": cls, "label": NICE_NAMES[cls], "skipped": False,
            "n": df.height, "first_scored_year": first_year,
            "burnin_end_year": burnin_end, "ref_level": ref, "band": band,
            "mean_dkl_1990": bias_1990,
            "early_volume_per_year": vol_early,
            "stable_mean_pros": pros_stable,
        })
        curves[cls] = (years, [means[y] for y in years], ref, band)
        print(f"  {cls} {NICE_NAMES[cls]}: burn-in ends {burnin_end} "
              f"(ref {ref:+.3f}, band {band:.3f}, 1990 bias "
              f"{('%+.3f' % bias_1990) if bias_1990 is not None else 'n/a'})",
              file=sys.stderr, flush=True)

    pl.from_dicts(rows).write_csv(RES / "burnin_by_class.csv")
    (RES / "burnin_by_class.json").write_text(json.dumps(rows, indent=1, default=float))

    ok = [r for r in rows if not r.get("skipped") and r["burnin_end_year"]]
    ends = np.array([r["burnin_end_year"] for r in ok])
    print(f"\nburn-in end: min {ends.min()}, median {np.median(ends):.0f}, "
          f"p90 {np.percentile(ends, 90):.0f}, max {ends.max()}",
          file=sys.stderr, flush=True)

    # correlates: volume and slow-moving proxy
    lv = np.log10([r["early_volume_per_year"] for r in ok])
    pr = np.array([r["stable_mean_pros"] for r in ok])
    be = ends.astype(float)
    r_vol = float(np.corrcoef(lv, be)[0, 1])
    r_pros = float(np.corrcoef(pr, be)[0, 1])
    print(f"corr(burn-in end, log10 early volume) = {r_vol:+.3f}",
          file=sys.stderr, flush=True)
    print(f"corr(burn-in end, stable mean prospective KL) = {r_pros:+.3f}",
          file=sys.stderr, flush=True)

    # ---- per-class burn-in year chart ----
    ok.sort(key=lambda r: (r["burnin_end_year"], r["cls"]))
    fig, ax = plt.subplots(figsize=(9, max(6, 0.3 * len(ok))))
    ys = np.arange(len(ok))
    ax.barh(ys, [r["burnin_end_year"] - 1985 for r in ok], left=1985,
            color="#2b6cb0", alpha=0.85)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']} ({r['cls']})" for r in ok], fontsize=8)
    ax.axvline(1995, color="#cc4444", lw=1.2, ls="--", label="1995 (paper's cut)")
    ax.set_xlabel("Burn-in end year (3 consecutive years inside stability band)")
    ax.set_xlim(1985, 2002)
    ax.set_title("Per-class burn-in end under the signal-stability criterion")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RES / "burnin_by_class.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- convergence example curves ----
    examples = [c for c in ("009", "035", "039", "032", "023", "034") if c in curves]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, cls in zip(axes.flat, examples):
        years, ms, ref, band = curves[cls]
        ax.plot(years, ms, "o-", ms=3, lw=1.2, color="#2b6cb0")
        ax.axhline(ref, color="#444", lw=0.8)
        ax.fill_between([min(years), max(years)], ref - band, ref + band,
                        color="#444", alpha=0.12)
        r = next(r for r in ok if r["cls"] == cls)
        ax.axvline(r["burnin_end_year"], color="#cc4444", lw=1.2, ls="--")
        ax.set_title(f"{NICE_NAMES[cls]} ({cls}); burn-in ends "
                     f"{r['burnin_end_year']}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-year mean signed dKL with stability band and burn-in cut")
    fig.tight_layout()
    fig.savefig(RES / "burnin_convergence_examples.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
