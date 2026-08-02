"""For each NICE class, characterise (a) the class's overall novelty
intensity and (b) the SHAPE of the relationship between firm-mean
novelty and survival, then test whether (a) predicts (b).

Hypothesis under test: mature, low-novelty classes (e.g.\ beverages)
have monotone-declining or hump-shaped survival/public-listing curves
in mean_pros (high-novelty firms = small craft entrants), while
high-novelty classes (e.g.\ software) have inverted-U curves
(optimal-distinctiveness pattern peaks at moderate novelty).

For each class:
  - n_filings, n_firms (>=3 clean filings)
  - class_mean_pros, class_mean_retr, class_mean_dkl
  - argmax of P(passed_5y) on firm-mean prospective KL
  - slope on lower half (left of argmax) and upper half (right of argmax)
    of the binned curve
  - argmax / domain-span ratio (0 = leftmost, 1 = rightmost)

Then plot class-mean novelty vs argmax-position (and other shape metrics).

Outputs:
  paper/results/industry_shape_vs_novelty.png
  paper/results/industry_shape_vs_novelty.txt
  paper/results/industry_shape_vs_novelty.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
    & pl.col("kl_vs_past").is_finite()
    & pl.col("kl_vs_future").is_finite()
    & pl.col("owner_name").is_not_null()
)

NICE_LABELS = {  # short labels for the NICE classes (most-used)
    "001": "Chemicals", "002": "Paints", "003": "Cosmetics",
    "004": "Lubricants", "005": "Pharma", "006": "Metals",
    "007": "Machines", "008": "Hand tools", "009": "Software/Electronics",
    "010": "Medical", "011": "Lighting", "012": "Vehicles",
    "013": "Firearms", "014": "Jewelry", "015": "Music",
    "016": "Paper", "017": "Rubber", "018": "Leather",
    "019": "Building", "020": "Furniture", "021": "Housewares",
    "022": "Cordage", "023": "Yarns", "024": "Textiles",
    "025": "Clothing", "026": "Lace", "027": "Carpets",
    "028": "Toys", "029": "Meat", "030": "Coffee",
    "031": "Agriculture", "032": "Beverages", "033": "Wine/Spirits",
    "034": "Tobacco", "035": "Advertising", "036": "Insurance",
    "037": "Construction", "038": "Telecom", "039": "Transport",
    "040": "Materials Treat", "041": "Education/Ent",
    "042": "Software Svcs", "043": "Food Svcs", "044": "Medical Svcs",
    "045": "Legal Svcs",
}


def load_class(cls: str, year_min=1990, year_max=2023) -> pl.DataFrame:
    p = PROC / f"outcomes_class{cls}.parquet"
    if not p.exists(): return pl.DataFrame()
    df = pl.read_parquet(p).filter(
        CLEAN & (pl.col("year") >= year_min) & (pl.col("year") <= year_max)
    ).with_columns(
        (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y"),
        (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
    ).select("owner_name", "kl_vs_past", "kl_vs_future", "dkl",
             "passed_5y", "reached_registration")
    return df


def shape_metrics(panel_pd, x_col, y_col, n_bins=15, q_lo=0.02, q_hi=0.98, min_n=50):
    if len(panel_pd) < n_bins * min_n:
        return None
    lo, hi = panel_pd[x_col].quantile([q_lo, q_hi])
    if hi <= lo: return None
    edges = np.linspace(lo, hi, n_bins + 1)
    centers, ys, ns = [], [], []
    for i in range(n_bins):
        m = (panel_pd[x_col] >= edges[i]) & (panel_pd[x_col] < edges[i + 1])
        sub = panel_pd[m]
        if len(sub) < min_n: continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        ys.append(float(sub[y_col].mean()))
        ns.append(int(len(sub)))
    if len(centers) < 5: return None
    centers = np.array(centers); ys = np.array(ys); ns = np.array(ns)
    argmax = int(np.argmax(ys))
    argmax_pos = (centers[argmax] - centers[0]) / (centers[-1] - centers[0])
    # Slope on the left and right halves
    if argmax > 0:
        left = np.polyfit(centers[:argmax+1], ys[:argmax+1], 1)
        slope_left = float(left[0])
    else:
        slope_left = 0.0
    if argmax < len(centers) - 1:
        right = np.polyfit(centers[argmax:], ys[argmax:], 1)
        slope_right = float(right[0])
    else:
        slope_right = 0.0
    overall = np.polyfit(centers, ys, 1)
    overall_slope = float(overall[0])
    # Quadratic fit (curvature)
    quad = np.polyfit(centers, ys, 2)  # ax^2 + bx + c
    return {
        "n_bins": len(centers),
        "argmax": float(centers[argmax]),
        "argmax_pos": float(argmax_pos),
        "y_max": float(ys[argmax]),
        "y_at_low": float(ys[0]),
        "y_at_high": float(ys[-1]),
        "slope_left": slope_left,
        "slope_right": slope_right,
        "overall_slope": overall_slope,
        "quad_a": float(quad[0]),  # negative => inverted-U; positive => U
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2023)
    args = ap.parse_args()

    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik").unique(subset=["owner_name"])

    classes = sorted(p.stem.replace("outcomes_class","")
                     for p in PROC.glob("outcomes_class*.parquet")
                     if p.stem.replace("outcomes_class","").isdigit())

    rows = []
    for cls in classes:
        df = load_class(cls, args.year_min, args.year_max)
        if df.is_empty(): continue
        firm = df.group_by("owner_name").agg(
            pl.col("kl_vs_past").mean().alias("mean_pros"),
            pl.col("kl_vs_future").mean().alias("mean_retr"),
            pl.col("dkl").mean().alias("mean_dkl"),
            pl.col("passed_5y").mean().alias("firm_passed5"),
            pl.len().alias("n_filings"),
        ).filter(pl.col("n_filings") >= 3)
        if firm.is_empty(): continue
        firm = firm.join(cw, on="owner_name", how="left").with_columns(
            pl.col("cik").is_not_null().alias("ever_public"))
        elig = firm.filter(pl.col("n_filings") >= 10)
        firm_pd = firm.to_pandas()
        elig_pd = elig.to_pandas()

        m_surv_pros = shape_metrics(firm_pd, "mean_pros", "firm_passed5")
        m_pub_pros = shape_metrics(elig_pd, "mean_pros", "ever_public", min_n=20)

        cls_mean_pros = float(df["kl_vs_past"].mean())
        cls_mean_retr = float(df["kl_vs_future"].mean())
        cls_mean_dkl = float(df["dkl"].mean())

        rec = {
            "nice_class": cls,
            "label": NICE_LABELS.get(cls, cls),
            "n_filings": df.height,
            "n_firms": firm.height,
            "n_firms_eligible": elig.height,
            "class_mean_pros": cls_mean_pros,
            "class_mean_retr": cls_mean_retr,
            "class_mean_dkl": cls_mean_dkl,
            "ever_public_rate": float(elig["ever_public"].mean()) if elig.height > 0 else float("nan"),
        }
        if m_surv_pros:
            rec.update({
                "surv_argmax_pos": m_surv_pros["argmax_pos"],
                "surv_quad_a": m_surv_pros["quad_a"],
                "surv_slope_left": m_surv_pros["slope_left"],
                "surv_slope_right": m_surv_pros["slope_right"],
                "surv_overall_slope": m_surv_pros["overall_slope"],
            })
        if m_pub_pros:
            rec.update({
                "pub_argmax_pos": m_pub_pros["argmax_pos"],
                "pub_quad_a": m_pub_pros["quad_a"],
                "pub_slope_left": m_pub_pros["slope_left"],
                "pub_slope_right": m_pub_pros["slope_right"],
                "pub_overall_slope": m_pub_pros["overall_slope"],
            })
        rows.append(rec)
        print(f"  [{cls} {NICE_LABELS.get(cls,cls)[:18]:<18s}] "
              f"n_filings={df.height:>9,}  cls_mean_pros={cls_mean_pros:.2f}  "
              f"surv_argmax_pos={rec.get('surv_argmax_pos', float('nan')):.2f}  "
              f"surv_quad_a={rec.get('surv_quad_a', float('nan')):+.4f}", flush=True)

    df_out = pl.DataFrame(rows)
    df_out.write_csv(RESULTS / "industry_shape_vs_novelty.csv")
    print(f"\n[done] wrote {RESULTS}/industry_shape_vs_novelty.csv", flush=True)

    # ---- Figure: class-mean novelty vs shape characteristics ----
    pdf = df_out.to_pandas()
    pdf = pdf.dropna(subset=["surv_quad_a", "class_mean_pros"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    # (a) class novelty vs shape curvature (quad_a)
    ax = axes[0, 0]
    ax.scatter(pdf["class_mean_pros"], pdf["surv_quad_a"],
               s=np.sqrt(pdf["n_firms"]) * 0.3, alpha=0.7,
               color="#2b6cb0", edgecolor="black", linewidth=0.5)
    for _, r in pdf.iterrows():
        if r["n_firms"] > 1000:  # only annotate larger classes
            ax.annotate(r["label"][:14], (r["class_mean_pros"], r["surv_quad_a"]),
                        fontsize=7, alpha=0.7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4, lw=0.7)
    ax.set_xlabel(r"Class-mean prospective KL, vs. past (novelty intensity)")
    ax.set_ylabel(r"Quadratic curvature of P(survival) vs mean_pros\n(<0 = inverted-U; >0 = U)")
    ax.set_title("(a) Does class novelty predict survival-curve shape?")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(pdf["class_mean_pros"], pdf["surv_argmax_pos"],
               s=np.sqrt(pdf["n_firms"]) * 0.3, alpha=0.7,
               color="#2b6cb0", edgecolor="black", linewidth=0.5)
    for _, r in pdf.iterrows():
        if r["n_firms"] > 1000:
            ax.annotate(r["label"][:14], (r["class_mean_pros"], r["surv_argmax_pos"]),
                        fontsize=7, alpha=0.7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel(r"Class-mean prospective KL (vs. past)")
    ax.set_ylabel(r"Position of survival peak (0=left, 1=right of bin range)")
    ax.set_title("(b) Where does the survival curve peak?")
    ax.grid(True, alpha=0.3)

    pdf2 = pdf.dropna(subset=["pub_quad_a"])
    ax = axes[1, 0]
    ax.scatter(pdf2["class_mean_pros"], pdf2["pub_quad_a"],
               s=np.sqrt(pdf2["n_firms_eligible"]) * 0.6, alpha=0.7,
               color="#cc4444", edgecolor="black", linewidth=0.5)
    for _, r in pdf2.iterrows():
        if r["n_firms_eligible"] > 200:
            ax.annotate(r["label"][:14], (r["class_mean_pros"], r["pub_quad_a"]),
                        fontsize=7, alpha=0.7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4, lw=0.7)
    ax.set_xlabel(r"Class-mean prospective KL (vs. past)")
    ax.set_ylabel(r"Quadratic curvature of P(public) vs mean_pros\n(<0 = inverted-U)")
    ax.set_title("(c) Going-public curve shape")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(pdf2["class_mean_pros"], pdf2["pub_overall_slope"],
               s=np.sqrt(pdf2["n_firms_eligible"]) * 0.6, alpha=0.7,
               color="#cc4444", edgecolor="black", linewidth=0.5)
    for _, r in pdf2.iterrows():
        if r["n_firms_eligible"] > 200:
            ax.annotate(r["label"][:14], (r["class_mean_pros"], r["pub_overall_slope"]),
                        fontsize=7, alpha=0.7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4, lw=0.7)
    ax.set_xlabel(r"Class-mean prospective KL (vs. past)")
    ax.set_ylabel(r"Overall slope of P(public) vs mean_pros")
    ax.set_title("(d) Going-public slope direction")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Industry novelty intensity vs shape of outcome curves\n"
        f"({len(pdf)} classes, point size $\\propto \\sqrt{{\\text{{n\\_firms}}}}$)",
        fontsize=12)
    fig.tight_layout()
    out_png = RESULTS / "industry_shape_vs_novelty.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_png}", flush=True)

    # ---- Numerical table ----
    out_txt = RESULTS / "industry_shape_vs_novelty.txt"
    with out_txt.open("w") as f:
        f.write("Industry novelty vs shape of survival/public curves.\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"{'cls':<5s}{'label':<22s}{'n_firms':>10s}{'cls_pros':>10s}"
                f"{'cls_dkl':>10s}{'surv_argmax_pos':>16s}{'surv_quad_a':>14s}"
                f"{'pub_quad_a':>13s}{'pub_slope':>11s}{'P(pub)':>9s}\n")
        for _, r in pdf.sort_values("class_mean_pros").iterrows():
            f.write(f"{r['nice_class']:<5s}{r['label'][:20]:<22s}"
                    f"{r['n_firms']:>10,}{r['class_mean_pros']:>10.3f}"
                    f"{r['class_mean_dkl']:>+10.4f}{r['surv_argmax_pos']:>16.3f}"
                    f"{r['surv_quad_a']:>+14.4f}"
                    f"{r.get('pub_quad_a',float('nan')):>+13.4f}"
                    f"{r.get('pub_overall_slope',float('nan')):>+11.4f}"
                    f"{r.get('ever_public_rate',float('nan')):>9.3f}\n")
        f.write("\nKey:\n")
        f.write("  cls_pros = class-mean prospective KL (novelty intensity of the industry)\n")
        f.write("  surv_argmax_pos = where the survival curve peaks (0=left edge, 1=right)\n")
        f.write("  surv_quad_a = curvature of P(survive) vs mean_pros (<0 inverted-U)\n")
        f.write("  pub_quad_a = same for P(ever_public)\n")
        f.write("  pub_slope = overall slope of P(ever_public) vs mean_pros\n")
        # Cross-correlations
        f.write("\nCross-correlations across classes:\n")
        for x in ["class_mean_pros", "class_mean_dkl"]:
            for y in ["surv_argmax_pos", "surv_quad_a", "surv_overall_slope",
                      "pub_quad_a", "pub_overall_slope"]:
                xx = pdf[x].dropna(); yy = pdf[y].dropna()
                idx = pdf[[x, y]].dropna().index
                if len(idx) >= 5:
                    r = float(np.corrcoef(pdf.loc[idx, x], pdf.loc[idx, y])[0, 1])
                    f.write(f"  corr({x:<22s}, {y:<22s}) = {r:+.3f}  (n={len(idx)})\n")
    print(f"[done] wrote {out_txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
