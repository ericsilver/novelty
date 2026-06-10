"""Diffusion Phase 1: RQ1 (theme-arrival rate), RQ2 (breadth + class-flow),
RQ3 (Bass p/q/m), RQ4 (entrant vs incumbent provenance).

Uses the panels built by diff3:
  data/processed/filing_topics.parquet   (per-filing topic assignment)
  data/processed/theme_panel.parquet     (theme x class x year prevalence)
And the four class records for owner_name (RQ4).

RQ5 (margin/revenue) and RQ6 (does borrowed novelty pay?) need external data
not on disk and remain blocked.

Output: paper/results/diff4_phase1.json (+ stdout)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT_JSON = REPO / "paper" / "results" / "diff4_phase1.json"
CLASSES = {"009": "software", "042": "tech-svcs", "039": "transport", "035": "advert-retail"}
FLOOR = 0.01      # prevalence floor for "theme present"
N_EARLY = 50      # number of earliest carriers per theme for provenance


def bass_F(t, p, q):
    """Bass cumulative-fraction adoption."""
    e = np.exp(-(p + q) * t)
    return (1 - e) / (1 + (q / max(p, 1e-9)) * e)


def fit_bass(years, cum_prev, max_prev):
    """Fit p,q to a normalized cumulative curve. Returns (p,q,r2) or None."""
    if len(years) < 5 or max_prev <= 0:
        return None
    y = (cum_prev / max_prev).clip(0, 1)
    t = (years - years.min()).astype(float)
    try:
        popt, _ = curve_fit(bass_F, t, y, p0=[0.01, 0.3],
                            bounds=([1e-5, 1e-5], [1.0, 5.0]), maxfev=2000)
    except Exception:
        return None
    p, q = popt
    yhat = bass_F(t, p, q)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return float(p), float(q), float(r2)


def main() -> int:
    t0 = time.time()
    panel = pl.read_parquet(DATA / "theme_panel.parquet")
    topics = pl.read_parquet(DATA / "filing_topics.parquet")
    T = int(panel["theme"].max()) + 1
    print(f"[load] {panel.height:,} (theme,class,year) cells, T={T}", file=sys.stderr)

    res = {"T": T, "classes": CLASSES}

    # --------- RQ1: theme-arrival rate (Heaps-style on themes-as-objects) ---
    # Theme "arrives" the first year it crosses FLOOR prevalence in ANY class
    # using share_top. Cumulative arrived themes vs cumulative filings.
    arrivals = (panel.filter(pl.col("share_top") >= FLOOR)
                     .group_by("theme").agg(pl.col("year").min().alias("arrival_year")))
    n_arrived = arrivals.height
    print(f"\n[RQ1] {n_arrived}/{T} themes arrived above 1% prevalence at least once",
          file=sys.stderr)
    arrival_curve = (arrivals.group_by("arrival_year").len()
                            .sort("arrival_year").to_pandas())
    arrival_curve["cum_themes"] = arrival_curve["len"].cumsum()
    # Cumulative filings by year (across all 4 classes)
    fy = (topics.group_by("year").len().sort("year").to_pandas())
    fy["cum_filings"] = fy["len"].cumsum()
    merged = arrival_curve.merge(fy[["year", "cum_filings"]],
                                 left_on="arrival_year", right_on="year", how="left")
    # Heaps' exponent fit: log(cum_themes) = a + b * log(cum_filings)
    mask = (merged["cum_themes"] > 0) & (merged["cum_filings"] > 0)
    if mask.sum() >= 3:
        lx = np.log(merged.loc[mask, "cum_filings"].values)
        ly = np.log(merged.loc[mask, "cum_themes"].values)
        slope = float(np.polyfit(lx, ly, 1)[0])
    else:
        slope = float("nan")
    print(f"     Heaps slope d log(themes)/d log(filings) = {slope:+.3f}  "
          f"(decelerating if < 1)", file=sys.stderr)
    res["RQ1"] = {"n_arrived": int(n_arrived), "heaps_slope": slope,
                  "curve": [{"year": int(r["arrival_year"]), "cum_themes": int(r["cum_themes"]),
                             "cum_filings": int(r["cum_filings"])}
                            for _, r in merged.iterrows() if not np.isnan(r["cum_filings"])]}

    # --------- RQ2: breadth + class-flow ------------------------------------
    # breadth: per theme, count of classes where it crossed FLOOR.
    arrivals_per_class = (panel.filter(pl.col("share_top") >= FLOOR)
                                .group_by(["theme", "nice"])
                                .agg(pl.col("year").min().alias("y_in"))
                                .sort(["theme", "y_in"]))
    breadth = arrivals_per_class.group_by("theme").len().rename({"len": "breadth"})
    bdist = breadth.group_by("breadth").len().sort("breadth")
    print(f"\n[RQ2] theme breadth across 4 classes:")
    for r in bdist.iter_rows(named=True):
        print(f"     {r['breadth']} classes reached: {r['len']:>3} themes")
    res["RQ2"] = {"breadth_dist": bdist.to_dicts()}

    # class-flow: ordered pairs (origin -> next class arrival) per theme
    flow = {(a, b): 0 for a in CLASSES for b in CLASSES if a != b}
    multi = 0
    for k, g in arrivals_per_class.group_by("theme"):
        rows = list(g.iter_rows(named=True))
        if len(rows) < 2:
            continue
        multi += 1
        origin = rows[0]["nice"]
        for r in rows[1:]:
            flow[(origin, r["nice"])] = flow.get((origin, r["nice"]), 0) + 1
    res["RQ2"]["class_flow"] = {f"{a}->{b}": v for (a, b), v in flow.items()}
    print(f"     {multi} themes reached >=2 classes; class-flow (origin->next):")
    for (a, b), v in sorted(flow.items(), key=lambda x: -x[1])[:8]:
        if v:
            print(f"     {CLASSES[a]:>14} -> {CLASSES[b]:<14}  {v}")

    # --------- RQ3: Bass fits per (theme, class) ----------------------------
    fits = []
    panel_pd = panel.to_pandas()
    panel_pd = panel_pd.sort_values(["theme", "nice", "year"])
    for (k, c), g in panel_pd.groupby(["theme", "nice"]):
        if g["share_top"].max() < FLOOR or len(g) < 6:
            continue
        cum = g["share_top"].cumsum().values
        if cum[-1] <= 0:
            continue
        fit = fit_bass(g["year"].values, cum, cum[-1])
        if fit is None:
            continue
        p, q, r2 = fit
        if r2 < 0.7:  # only retain reasonable Bass shapes
            continue
        fits.append({"theme": int(k), "nice": c, "p": p, "q": q, "qp": q / max(p, 1e-9),
                     "n_years": int(len(g)), "max_prev": float(g["share_top"].max()),
                     "r2": r2})
    print(f"\n[RQ3] Bass fits (R^2>=0.7): {len(fits)} (theme,class) cells")
    if fits:
        qp = np.array([f["qp"] for f in fits])
        ps = np.array([f["p"] for f in fits])
        qs = np.array([f["q"] for f in fits])
        print(f"     median p={np.median(ps):.4f}  median q={np.median(qs):.3f}  "
              f"median q/p={np.median(qp):.0f}")
        # high q/p = strong word-of-mouth (imitation-heavy diffusion = fad-shape risk
        # but also platform-replication); high p with low q/p = steady innovation pull.
        # crude fad vs platform: fad = high q/p AND short n_years above floor
        plat = [f for f in fits if f["qp"] < 20]
        fads = [f for f in fits if f["qp"] >= 50]
        print(f"     'platform-like' (q/p<20): {len(plat)};  'fad-like' (q/p>=50): {len(fads)}")
    res["RQ3"] = {"n_fits": len(fits), "fits": fits[:60]}  # cap

    # --------- RQ4: entrant vs incumbent provenance -------------------------
    # Build owner -> first grant year across all 4 classes.
    print(f"\n[RQ4] building owner -> first-grant-year map...", file=sys.stderr)
    owner_first = {}
    for c in CLASSES:
        df = pl.read_parquet(DATA / f"tm_class{c}.parquet",
                             columns=["owner_name", "filing_date", "registration_date"])
        df = df.with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")
        ).filter(
            pl.col("year").is_not_null() & (pl.col("registration_date").fill_null("").str.len_chars() >= 4)
            & pl.col("owner_name").is_not_null()
        )
        for own, yr in df.group_by("owner_name").agg(pl.col("year").min()).iter_rows():
            if own:
                owner_first[own] = min(owner_first.get(own, 99999), int(yr))
    print(f"     {len(owner_first):,} distinct owners with at least one grant", file=sys.stderr)

    # For each theme, take the earliest N carriers (by year), classify entrant vs incumbent.
    rec_owners = []
    for c in CLASSES:
        d = pl.read_parquet(DATA / f"tm_class{c}.parquet",
                             columns=["serial_number", "owner_name"])
        rec_owners.append(d)
    own_map = pl.concat(rec_owners)
    full = topics.join(own_map, on="serial_number", how="inner")

    prov_rows = []
    for k in range(T):
        cell = full.filter((pl.col("top_topic") == k) & (pl.col("top_weight") >= 0.20))
        if cell.height < N_EARLY:
            continue
        cell = cell.sort("year").head(N_EARLY)
        entrant = 0
        for own, yr in cell.select("owner_name", "year").iter_rows():
            fy = owner_first.get(own)
            if fy is not None and fy >= int(yr) - 1:
                entrant += 1
        prov_rows.append({"theme": int(k), "n_early": cell.height,
                          "entrant_share": entrant / max(cell.height, 1)})
    if prov_rows:
        es = np.array([r["entrant_share"] for r in prov_rows])
        print(f"\n[RQ4] {len(prov_rows)} themes scored; mean entrant share = {es.mean():.1%}; "
              f"median = {np.median(es):.1%}")
        print(f"     themes mostly born from entrants (Schumpeter I, entrant_share>=0.5): "
              f"{int((es>=0.5).sum())}/{len(es)}")
        print(f"     themes mostly born from incumbents (Schumpeter II, entrant_share<=0.2): "
              f"{int((es<=0.2).sum())}/{len(es)}")
    res["RQ4"] = {"n_themes": len(prov_rows),
                  "mean_entrant_share": float(es.mean()) if prov_rows else None,
                  "median_entrant_share": float(np.median(es)) if prov_rows else None,
                  "share_entrant_dominant": float((es >= 0.5).mean()) if prov_rows else None,
                  "share_incumbent_dominant": float((es <= 0.2).mean()) if prov_rows else None,
                  "by_theme": prov_rows}

    OUT_JSON.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[done] -> {OUT_JSON}  ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
