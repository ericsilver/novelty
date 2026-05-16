"""Audit the lower-left commonplace-penalty against two confounds.

H1 — length confound:
    Short G/S descriptions have mechanically smaller KL in both
    directions (the per-filing distribution lives over fewer tokens,
    so the divergence from the class-corpus reference is structurally
    suppressed).  Test: regress per-filing pros_kl and retr_kl on
    log(n_terms), take residuals, re-aggregate to firm-level, replot.
    If the lower-left valley shrinks substantially, the penalty is
    explained by length.

H5 — firm-quality confound:
    Low-KL filings may disproportionately come from unsophisticated
    firms (single-product owners that use generic descriptions and
    don't bother with Section 8 maintenance).  Test: restrict to
    firms with >=10 clean filings (procedurally sophisticated).  If
    the lower-left valley flattens, the penalty is firm-quality, not
    novelty.

Both controls applied jointly = panel D.  If the penalty survives
all four panels with similar magnitude, the substantive reading
(optimal distinctiveness — a trademark needs at least one direction
of distinctiveness to be a defensible asset) gains support.

Focal class: 009 (Electronics / scientific apparatus — largest single
class by clean-filing count, so the panels remain dense).

Output:
  paper/results/lower_left_audit_class009.png
  paper/results/lower_left_audit_class009.txt   (numerical summary)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

import argparse


def clean_filter(year_min: int, year_max: int) -> pl.Expr:
    """CLEAN expression parameterized by year window.

    Year cap was 2018 historically (5y survival look-ahead).  At 2018 the
    post-cutoff portion of the corpus is 4.95M filings (~39% of clean
    total), much larger than the ~1.2M raw pre-1990 burn-in.  Cap of 2023
    leaves ~1.45M post-cutoff filings, roughly matching the pre-window
    buffer.  Recent filings get truncated survival observation, but
    `passed_5y = reached_registration AND currently_live` is already a
    "ever-registered & still alive today" metric, so the censoring is on
    the same axis it always was.
    """
    return (
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= year_min) & (pl.col("year") <= year_max)
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
    )

N_BINS = 36
MIN_FIRMS_BASELINE = 3
MIN_FIRMS_SOPHISTICATED = 10


def load_filings(classes: list[str], year_min: int, year_max: int) -> pl.DataFrame:
    parts = []
    clean = clean_filter(year_min, year_max)
    for cls in classes:
        path = PROC / f"outcomes_class{cls}.parquet"
        if not path.exists():
            continue
        df = (pl.read_parquet(path)
              .filter(clean & pl.col("owner_name").is_not_null())
              .with_columns(
                  (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y"),
                  pl.col("n_terms").log().alias("log_n_terms"),
                  pl.lit(cls).alias("nice_class"),
              )
              .select("owner_name", "nice_class", "prospective_kl", "retrospective_kl",
                      "n_terms", "log_n_terms", "reached_registration", "passed_5y"))
        parts.append(df)
    return pl.concat(parts)


def residualize_kls(df: pl.DataFrame, within_class: bool = False) -> tuple[pl.DataFrame, dict]:
    """Replace pros/retr KL with OLS residuals after regressing on log(n_terms).

    If within_class, fit a separate OLS per nice_class so that the
    between-class component of KL is also removed.  This matters for
    the all-classes pool, where a class-level shift in mean KL would
    otherwise dominate the lower-left.
    """
    diag = {}
    if not within_class:
        x = df["log_n_terms"].to_numpy().astype(np.float64)
        for col in ("prospective_kl", "retrospective_kl"):
            y = df[col].to_numpy().astype(np.float64)
            b, a = np.polyfit(x, y, 1)
            resid = y - (a + b * x)
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            diag[col] = {"slope": float(b), "intercept": float(a),
                         "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0}
            df = df.with_columns(pl.Series(col + "_resid", resid))
        return df, diag

    # Within-class residualization
    classes = sorted(df["nice_class"].unique().to_list())
    pros = df["prospective_kl"].to_numpy().astype(np.float64).copy()
    retr = df["retrospective_kl"].to_numpy().astype(np.float64).copy()
    cls_arr = df["nice_class"].to_numpy()
    x_all = df["log_n_terms"].to_numpy().astype(np.float64)
    pros_resid = np.empty_like(pros)
    retr_resid = np.empty_like(retr)
    for cls in classes:
        mask = (cls_arr == cls)
        if mask.sum() < 5:
            pros_resid[mask] = pros[mask] - pros[mask].mean()
            retr_resid[mask] = retr[mask] - retr[mask].mean()
            continue
        xx = x_all[mask]
        for col_name, y, out in [("prospective_kl", pros, pros_resid),
                                  ("retrospective_kl", retr, retr_resid)]:
            yy = y[mask]
            b, a = np.polyfit(xx, yy, 1)
            out[mask] = yy - (a + b * xx)
    # diagnostics: pooled R²
    for col, y, r in [("prospective_kl", pros, pros_resid),
                      ("retrospective_kl", retr, retr_resid)]:
        ss_res = float(np.sum(r ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        diag[col] = {"slope": float("nan"), "intercept": float("nan"),
                     "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
                     "note": "within-class OLS; R² is pooled"}
    df = df.with_columns(
        pl.Series("prospective_kl_resid", pros_resid),
        pl.Series("retrospective_kl_resid", retr_resid),
    )
    return df, diag


def make_firm_panel(df: pl.DataFrame, kl_p: str, kl_r: str,
                    min_filings: int) -> pd.DataFrame:
    return (df.group_by("owner_name").agg(
        pl.col(kl_p).mean().alias("mean_pros"),
        pl.col(kl_r).mean().alias("mean_retr"),
        pl.col("passed_5y").mean().alias("mean_passed5"),
        pl.col("reached_registration").mean().alias("mean_reach"),
        pl.len().alias("n_filings"),
    ).filter(pl.col("n_filings") >= min_filings)).to_pandas()


def topomap_panel(ax, panel: pd.DataFrame, title: str, n_bins: int = N_BINS):
    """Render a per-cell mean_passed5 heatmap; return (im, p_edges, r_edges, H)."""
    p_lo, p_hi = panel["mean_pros"].quantile([0.02, 0.98])
    r_lo, r_hi = panel["mean_retr"].quantile([0.02, 0.98])
    p_edges = np.linspace(p_lo, p_hi, n_bins + 1)
    r_edges = np.linspace(r_lo, r_hi, n_bins + 1)

    panel = panel.copy()
    panel["p_bin"] = np.digitize(panel["mean_pros"], p_edges) - 1
    panel["r_bin"] = np.digitize(panel["mean_retr"], r_edges) - 1
    panel = panel[(panel["p_bin"] >= 0) & (panel["p_bin"] < n_bins)
                  & (panel["r_bin"] >= 0) & (panel["r_bin"] < n_bins)]

    H = np.full((n_bins, n_bins), np.nan)
    grouped = panel.groupby(["r_bin", "p_bin"])["mean_passed5"].agg(["mean", "size"]).reset_index()
    for _, row in grouped.iterrows():
        if row["size"] >= 3:  # min 3 firms per cell so means are stable
            H[int(row["r_bin"]), int(row["p_bin"])] = row["mean"]

    vals = H[~np.isnan(H)]
    if vals.size > 0:
        vmin, vmax = np.percentile(vals, [5, 95])
    else:
        vmin, vmax = 0, 1
    im = ax.pcolormesh(p_edges, r_edges, H, cmap="viridis",
                       shading="flat", vmin=vmin, vmax=vmax)
    diag_lo = max(p_lo, r_lo)
    diag_hi = min(p_hi, r_hi)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", linewidth=0.7, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"Mean prospective KL  (firm)")
    ax.set_ylabel(r"Mean retrospective KL  (firm)")
    ax.set_xlim(p_lo, p_hi)
    ax.set_ylim(r_lo, r_hi)
    ax.text(0.02, 0.02, "commonplace", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#444", alpha=0.7))
    ax.text(0.99, 0.98, "novel", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#1f3a7a",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f3a7a", alpha=0.7))
    return im


def lower_left_penalty(panel: pd.DataFrame, q: float = 0.25) -> dict:
    """Compare mean passed_5y in the lower-left q-quantile box vs the rest.

    'Lower-left' = mean_pros below the q-th percentile AND mean_retr below
    the q-th percentile of THIS panel's firm distribution.
    """
    pq = panel["mean_pros"].quantile(q)
    rq = panel["mean_retr"].quantile(q)
    ll = (panel["mean_pros"] < pq) & (panel["mean_retr"] < rq)
    ll_mean = float(panel.loc[ll, "mean_passed5"].mean()) if ll.any() else float("nan")
    rest_mean = float(panel.loc[~ll, "mean_passed5"].mean()) if (~ll).any() else float("nan")
    # Also a tighter 10th-percentile box for the deepest lower-left
    pq10 = panel["mean_pros"].quantile(0.10)
    rq10 = panel["mean_retr"].quantile(0.10)
    ll10 = (panel["mean_pros"] < pq10) & (panel["mean_retr"] < rq10)
    ll10_mean = float(panel.loc[ll10, "mean_passed5"].mean()) if ll10.any() else float("nan")
    return {
        "n_total": int(len(panel)),
        "ll25_n": int(ll.sum()),
        "ll25_passed5": ll_mean,
        "rest25_passed5": rest_mean,
        "penalty25": rest_mean - ll_mean,
        "ll10_n": int(ll10.sum()),
        "ll10_passed5": ll10_mean,
        "penalty10": rest_mean - ll10_mean,  # against the same rest pool
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="009",
                    help="comma-separated NICE class codes, or 'all'")
    ap.add_argument("--label", default=None, help="label for output file stem")
    ap.add_argument("--within-class-resid", action="store_true",
                    help="when pooling multiple classes, residualize within each class")
    ap.add_argument("--year-min", type=int, default=1990,
                    help="earliest filing year to include (default 1990)")
    ap.add_argument("--year-max", type=int, default=2023,
                    help="latest filing year to include (default 2023; previously 2018)")
    args = ap.parse_args()

    if args.classes == "all":
        classes = sorted(p.stem.replace("outcomes_class", "")
                         for p in PROC.glob("outcomes_class*.parquet")
                         if p.stem.replace("outcomes_class", "").isdigit())
    else:
        classes = [c.strip().zfill(3) for c in args.classes.split(",")]
    label = args.label or ("all" if len(classes) > 5 else "_".join(classes))

    print(f"[load] {len(classes)} class(es), years {args.year_min}-{args.year_max}…", flush=True)
    raw = load_filings(classes, args.year_min, args.year_max)
    print(f"  {raw.height:,} clean filings (post-CLEAN, finite KLs, owner present)", flush=True)

    # Sanity: per-filing correlation between log(n_terms) and KL
    x = raw["log_n_terms"].to_numpy()
    for col in ("prospective_kl", "retrospective_kl"):
        y = raw[col].to_numpy()
        rho = float(np.corrcoef(x, y)[0, 1])
        print(f"  corr(log_n_terms, {col}) = {rho:+.3f}", flush=True)

    within = args.within_class_resid and len(classes) > 1
    mode = "within-class" if within else "pooled"
    print(f"\n[resid] regressing pros_kl, retr_kl ~ log(n_terms)  [{mode}]…", flush=True)
    res, diag = residualize_kls(raw, within_class=within)
    for col, d in diag.items():
        if "slope" in d and not np.isnan(d.get("slope", 0)):
            print(f"  {col}: slope={d['slope']:+.4f}  "
                  f"intercept={d['intercept']:+.4f}  R²={d['R2']:.4f}", flush=True)
        else:
            print(f"  {col}: pooled R² = {d['R2']:.4f}  ({d.get('note','')})", flush=True)

    # Four firm-level panels
    print("\n[panel] building firm panels…", flush=True)
    panel_A = make_firm_panel(raw, "prospective_kl", "retrospective_kl",
                              MIN_FIRMS_BASELINE)
    panel_B = make_firm_panel(res, "prospective_kl_resid", "retrospective_kl_resid",
                              MIN_FIRMS_BASELINE)
    panel_C = make_firm_panel(raw, "prospective_kl", "retrospective_kl",
                              MIN_FIRMS_SOPHISTICATED)
    panel_D = make_firm_panel(res, "prospective_kl_resid", "retrospective_kl_resid",
                              MIN_FIRMS_SOPHISTICATED)

    rows = []
    for name, p in [("A (baseline)", panel_A),
                    ("B (residualized)", panel_B),
                    ("C (>=10 filings)", panel_C),
                    ("D (both controls)", panel_D)]:
        q = lower_left_penalty(p)
        print(f"\n  {name:20s}  n_firms={q['n_total']:>7,}", flush=True)
        print(f"    LL@25%: n={q['ll25_n']:>6,}  passed5={q['ll25_passed5']:.3f}  "
              f"rest={q['rest25_passed5']:.3f}  penalty={q['penalty25']:+.3f}", flush=True)
        print(f"    LL@10%: n={q['ll10_n']:>6,}  passed5={q['ll10_passed5']:.3f}  "
              f"penalty={q['penalty10']:+.3f}", flush=True)
        rows.append((name, q))

    # ---------- Plot ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panels = [
        (panel_A, "A: baseline — raw KLs, firms ≥3 filings"),
        (panel_B, "B: H1 — KLs residualized on log(n_terms), firms ≥3"),
        (panel_C, "C: H5 — raw KLs, firms ≥10 filings"),
        (panel_D, "D: H1+H5 — residualized & firms ≥10"),
    ]
    for ax, (p, t) in zip(axes.flat, panels):
        im = topomap_panel(ax, p, t)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    scope = f"all 45 classes pooled" if len(classes) > 5 else f"class(es) {','.join(classes)}"
    resid_note = "within-class" if within else "pooled"
    fig.suptitle(
        f"{scope} — lower-left commonplace-penalty audit "
        f"(residualization: {resid_note})\n"
        f"Cells coloured by per-firm mean Still-Live rate (registered \\& not cancelled).  "
        f"Colour clipped to 5--95\\% of cell values; min 3 firms per cell.\n"
        f"If H1 explains the valley, panel B looks flat.  If H5 explains it, panel C looks flat.  "
        f"If both fail, panel D still shows a lower-left valley.",
        y=1.03, fontsize=10,
    )
    fig.tight_layout()
    out_png = RESULTS / f"lower_left_audit_{label}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {out_png}", flush=True)

    # ---------- Numerical summary file ----------
    txt = RESULTS / f"lower_left_audit_{label}.txt"
    with txt.open("w") as f:
        f.write(f"Lower-left commonplace-penalty audit — {scope}\n")
        f.write(f"Residualization: {resid_note}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"N clean filings: {raw.height:,}\n\n")
        f.write("Per-filing correlations with log(n_terms):\n")
        for col in ("prospective_kl", "retrospective_kl"):
            rho = float(np.corrcoef(raw['log_n_terms'].to_numpy(),
                                    raw[col].to_numpy())[0, 1])
            f.write(f"  corr(log_n_terms, {col}) = {rho:+.3f}\n")
        f.write(f"\nResidualization model (OLS, per-filing, {resid_note}):\n")
        for col, d in diag.items():
            if "slope" in d and not np.isnan(d.get("slope", 0)):
                f.write(f"  {col} = {d['intercept']:+.4f} + {d['slope']:+.4f} · log(n_terms)   "
                        f"R² = {d['R2']:.4f}\n")
            else:
                f.write(f"  {col} pooled R² = {d['R2']:.4f}  "
                        f"({d.get('note', '')})\n")
        f.write("\nFirm-level lower-left penalty by panel:\n")
        f.write(f"  {'Panel':<22s}{'n_firms':>10s}{'LL@25 n':>10s}"
                f"{'LL@25 surv':>12s}{'rest surv':>11s}{'penalty':>10s}"
                f"{'LL@10 n':>10s}{'LL@10 surv':>12s}\n")
        for name, q in rows:
            f.write(f"  {name:<22s}{q['n_total']:>10,}{q['ll25_n']:>10,}"
                    f"{q['ll25_passed5']:>12.3f}{q['rest25_passed5']:>11.3f}"
                    f"{q['penalty25']:>+10.3f}{q['ll10_n']:>10,}"
                    f"{q['ll10_passed5']:>12.3f}\n")
        f.write("\nInterpretation:\n")
        f.write("  H1 supported if panel B penalty25 << panel A penalty25.\n")
        f.write("  H5 supported if panel C penalty25 << panel A penalty25.\n")
        f.write("  H4 (optimal distinctiveness) supported if panel D penalty25\n")
        f.write("     remains similar in magnitude to panel A.\n")
    print(f"[done] wrote {txt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
