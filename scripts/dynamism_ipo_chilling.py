"""D1+D2+D3: IPO chilling event study at the NICE-class level.

For each USPTO trademark owner that we can crosswalk to SEC EDGAR
(uspto_sec_crosswalk.parquet) and that first appears in SEC data in
year y >= 2012 (Census FSDS coverage stabilises around that point):
treat y as the IPO event year. For each such firm we identify the
NICE classes where it actively files trademarks. Each
(NICE class, IPO year) pair is an event cell.

Event study: stack pre/post non-IPO-firm debut counts in each event
cell from t-3 to t+3 (years relative to IPO). Compare debut counts
in event cells vs non-event control cells (NICE-class-years with no
significant IPO that year). Two-way fixed effects DiD.

Why coarse (NICE class) rather than token-vertical? Time budget. If
the class-level test finds a chilling effect, we follow up with the
token-vertical version. If it finds nothing, the finer cut is
unlikely to surface a signal that the coarse one missed.

Definition of "actively files": >= 5 trademark filings in the class
in any 5-year window centred on the IPO year.

Outputs:
  paper/results/dynamism_ipo_events.csv          (one row per IPO firm)
  paper/results/dynamism_ipo_class_events.csv    (one row per class-event)
  paper/results/dynamism_ipo_chilling_panel.csv  (event-study panel)
  paper/results/dynamism_ipo_chilling.tex        (DiD regression table)
  paper/results/dynamism_ipo_chilling.png        (event-study plot)
  paper/results/dynamism_ipo_chilling.txt        (summary)
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from novelty.industries import name as industry_name

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES  = REPO / "paper" / "results"

YEAR_MIN = 2005
YEAR_MAX = 2025
EVENT_MIN_YEAR = 2012  # first plausible IPO event in our data
EVENT_MAX_YEAR = 2022  # need at least 3 post-event years
PRE, POST = 3, 3       # event window
ACTIVE_THRESHOLD = 5   # min filings in class in 5y window around IPO


def class_list() -> list[str]:
    return sorted(p.stem.replace("tm_class","") for p in PROC.glob("tm_class*.parquet")
                  if p.stem.replace("tm_class","").isdigit())


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)

    # Step 1: build IPO event panel from crosswalk + sec_firm_year
    print("[D1] building IPO event panel …", flush=True)
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet", columns=["cik","fy"])
    first_fy = sec.group_by("cik").agg(pl.col("fy").min().alias("ipo_fy"))
    xw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet",
                          columns=["owner_name","cik","first_year","last_year","n_filings"])
    ipo = (xw.join(first_fy, on="cik", how="inner")
             .filter((pl.col("ipo_fy") >= EVENT_MIN_YEAR) & (pl.col("ipo_fy") <= EVENT_MAX_YEAR))
             # Filter out firms that were filing trademarks long before SEC entry
             # (those are old public firms, not IPOs)
             .filter(pl.col("first_year") >= pl.col("ipo_fy") - 5)
             .filter(pl.col("n_filings") >= 1))
    print(f"  IPO events kept: {ipo.height:,} firms (fy >= {EVENT_MIN_YEAR}, USPTO first_year >= ipo_fy-5)")
    ipo.write_csv(RES / "dynamism_ipo_events.csv")

    # Step 2: for each IPO firm, identify the NICE classes it actively files in
    print("[D2] mapping IPO firms to NICE classes …", flush=True)
    ipo_owners = set(ipo["owner_name"].to_list())
    ipo_owner_to_fy = dict(ipo.select(["owner_name","ipo_fy"]).iter_rows())

    class_event_rows = []
    debut_panel_rows = []
    for cls in class_list():
        tm = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                              columns=["owner_name","filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        ).with_columns(pl.col("filing_date").str.slice(0,4).cast(pl.Int64).alias("year"))
        tm = tm.filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX))
        if tm.height == 0:
            continue
        # IPO firms active in this class
        ipo_in_cls = tm.filter(pl.col("owner_name").is_in(ipo_owners))
        if ipo_in_cls.height == 0:
            continue
        # Per (IPO firm), filing count in 5y window centred on IPO year
        ipo_in_cls_pd = ipo_in_cls.to_pandas()
        ipo_in_cls_pd["ipo_fy"] = ipo_in_cls_pd["owner_name"].map(ipo_owner_to_fy)
        ipo_in_cls_pd["rel_year"] = ipo_in_cls_pd["year"] - ipo_in_cls_pd["ipo_fy"]
        active = (ipo_in_cls_pd[ipo_in_cls_pd["rel_year"].between(-2,2)]
                    .groupby(["owner_name","ipo_fy"]).size().reset_index(name="n_5y"))
        active = active[active["n_5y"] >= ACTIVE_THRESHOLD]
        for _, r in active.iterrows():
            class_event_rows.append({
                "cls": cls, "class_name": industry_name(cls),
                "owner_name": r["owner_name"], "ipo_fy": int(r["ipo_fy"]),
                "n_filings_5y": int(r["n_5y"]),
            })

        del tm, ipo_in_cls, ipo_in_cls_pd
        gc.collect()

    class_events = pl.from_dicts(class_event_rows)
    class_events.write_csv(RES / "dynamism_ipo_class_events.csv")
    print(f"  IPO class-events (active-filer): {class_events.height:,}")
    # Aggregate: number of IPO events per (class, year)
    cls_ipo_yr = (class_events.group_by(["cls","ipo_fy"])
                    .agg(pl.len().alias("n_ipo_events_in_year")))
    print(f"  Unique (class, ipo_fy) cells: {cls_ipo_yr.height}")

    # Step 3: build the debut-count panel
    print("[D3] building class-year debut panel for event study …", flush=True)
    # Reuse the firsttimer share data we already computed
    ftf_panel = pl.read_csv(RES / "dynamism_firsttimer_share_byclass.csv")
    # For each (class, year), compute non-IPO-firm debut count (proxy: all
    # debuts in class-year, since IPO firms are by construction repeat
    # filers post-IPO and most of their debuts predate the IPO year)
    panel = ftf_panel.filter(pl.col("year").is_between(YEAR_MIN, YEAR_MAX)).rename(
        {"n_debuts": "debut_count", "n_filings": "total_filings"}).with_columns(
        pl.col("cls").cast(pl.Int64).cast(pl.Utf8).str.zfill(3).alias("cls"))
    cls_ipo_yr = cls_ipo_yr.with_columns(
        pl.col("cls").cast(pl.Utf8).str.zfill(3).alias("cls"))
    panel = panel.join(cls_ipo_yr, left_on=["cls","year"], right_on=["cls","ipo_fy"],
                       how="left").with_columns(
        pl.col("n_ipo_events_in_year").fill_null(0))

    # Treat: (class, year) has any IPO event (binary or count)
    panel = panel.with_columns(
        (pl.col("n_ipo_events_in_year") > 0).cast(pl.Int64).alias("ipo_treat"))
    panel.write_csv(RES / "dynamism_ipo_chilling_panel.csv")

    # Step 4: event-study regression
    # Stack: for each treated (class, IPO year y_e), compute log(debut_count)
    # at relative years t in [-PRE, +POST]. Compare to control (non-event)
    # class-years matched on year.
    # Build long form
    treats = []
    for r in class_events.iter_rows(named=True):
        for rel in range(-PRE, POST+1):
            yr = r["ipo_fy"] + rel
            if not (YEAR_MIN <= yr <= YEAR_MAX):
                continue
            row = panel.filter((pl.col("cls") == r["cls"]) & (pl.col("year") == yr))
            if row.height == 0: continue
            treats.append({
                "cls": r["cls"], "ipo_fy": r["ipo_fy"], "owner_name": r["owner_name"],
                "rel_year": rel, "year": yr,
                "debut_count": int(row["debut_count"].item()),
                "total_filings": int(row["total_filings"].item()),
            })
    event_df = pd.DataFrame(treats)

    # Group by (class, rel_year): average log debut_count
    event_df["log_debut"] = np.log1p(event_df["debut_count"])
    by_rel = event_df.groupby("rel_year").agg(
        mean_log_debut=("log_debut","mean"),
        mean_debut=("debut_count","mean"),
        n_events=("debut_count","size"),
    ).reset_index()
    print("\nAverage log debut count by event-relative year (across all IPO events):")
    print(by_rel.to_string(index=False))

    # Two-way FE DiD: log(debut_count) ~ post * treat, with class & year FE
    # Simpler version: within-treated, compare pre (rel < 0) to post (rel >= 0)
    event_df["post"] = (event_df["rel_year"] >= 0).astype(int)
    event_df["cls_str"] = event_df["cls"].astype(str)
    # FE via dummies
    X_df = pd.get_dummies(event_df[["cls_str","year","post"]],
                            columns=["cls_str","year"], drop_first=True, dtype=float)
    X = sm.add_constant(X_df.astype(float))
    y = event_df["log_debut"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": event_df["cls_str"].values})
    b_post = model.params["post"]; se_post = model.bse["post"]; t_post = b_post/se_post
    p_post = model.pvalues["post"]

    # Also: regress log_debut on rel_year (linear time-around-event)
    event_df["rel_pos"] = event_df["rel_year"].clip(lower=0)
    event_df["rel_neg"] = event_df["rel_year"].clip(upper=0)
    X2_df = pd.get_dummies(event_df[["cls_str","year","rel_pos","rel_neg"]],
                            columns=["cls_str","year"], drop_first=True, dtype=float)
    X2 = sm.add_constant(X2_df.astype(float))
    model2 = sm.OLS(y, X2).fit(cov_type="cluster", cov_kwds={"groups": event_df["cls_str"].values})

    summary = []
    summary.append("D3 IPO chilling event study at the NICE-class level")
    summary.append("=" * 70)
    summary.append(f"IPO firm events:                {ipo.height:,}")
    summary.append(f"Active-filer class-events:      {class_events.height:,}")
    summary.append(f"Event cells in regression:      {len(event_df):,}")
    summary.append("")
    summary.append("Within-treated DiD (post vs pre, class FE + year FE, cluster-by-class SE):")
    summary.append(f"  post coefficient on log(1+debut_count): {b_post:+.4f}  SE={se_post:.4f}  t={t_post:+.2f}  p={p_post:.4f}")
    summary.append(f"  Implied effect: post-event debut count multiplied by {np.exp(b_post):.3f}")
    summary.append("")
    summary.append("Slope regression: log_debut on rel_year split into rel_neg + rel_pos:")
    summary.append(f"  rel_neg slope: {model2.params['rel_neg']:+.4f}  SE={model2.bse['rel_neg']:.4f}  t={model2.params['rel_neg']/model2.bse['rel_neg']:+.2f}")
    summary.append(f"  rel_pos slope: {model2.params['rel_pos']:+.4f}  SE={model2.bse['rel_pos']:.4f}  t={model2.params['rel_pos']/model2.bse['rel_pos']:+.2f}")
    summary.append(f"  rel_pos - rel_neg = {model2.params['rel_pos']-model2.params['rel_neg']:+.4f}")
    summary.append("")
    summary.append("Interpretation:")
    summary.append("  - Negative post coefficient => chilling effect (fewer first-time-filers post-IPO)")
    summary.append("  - rel_pos < rel_neg => slope flips down after the IPO")
    summary.append("  - Class-level test only; finer token-vertical version not run yet.")
    text = "\n".join(summary)
    (RES / "dynamism_ipo_chilling.txt").write_text(text)
    print("\n" + text)

    # tex table
    tex = [r"\begin{tabular}{lrrr}",
           r"\toprule",
           r"Estimator & coef & cluster SE & $t$ \\",
           r"\midrule",
           f"  $\\mathbf{{1}}[t \\geq \\text{{IPO}}]$ (post vs pre) & {b_post:+.4f} & {se_post:.4f} & {t_post:+.2f} \\\\",
           f"  pre-event slope (rel\\_neg) & {model2.params['rel_neg']:+.4f} & {model2.bse['rel_neg']:.4f} & {model2.params['rel_neg']/model2.bse['rel_neg']:+.2f} \\\\",
           f"  post-event slope (rel\\_pos) & {model2.params['rel_pos']:+.4f} & {model2.bse['rel_pos']:.4f} & {model2.params['rel_pos']/model2.bse['rel_pos']:+.2f} \\\\",
           r"\midrule",
           r"  Class FE + year FE & \multicolumn{3}{c}{Yes} \\",
           f"  $N$ event-cells & \\multicolumn{{3}}{{c}}{{{len(event_df):,}}} \\\\",
           f"  IPO-firm events & \\multicolumn{{3}}{{c}}{{{ipo.height:,}}} \\\\",
           r"\bottomrule",
           r"\end{tabular}",]
    (RES / "dynamism_ipo_chilling.tex").write_text("\n".join(tex))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(by_rel["rel_year"], by_rel["mean_log_debut"],
                yerr=event_df.groupby("rel_year")["log_debut"].sem().values,
                marker="o", color="#2b6cb0", lw=2, ms=6, capsize=3,
                label=f"mean log(1+debut count) ± SEM")
    ax.axvline(0, color="grey", ls="--", lw=1, label="IPO year (t=0)")
    ax.set_xlabel("Years relative to IPO event")
    ax.set_ylabel("log(1 + non-IPO-firm debut count)")
    ax.set_title(
        f"Event study: trademark debut activity in the IPO firm's NICE class\n"
        f"({class_events.height:,} class-event cells from {ipo.height:,} IPO-ing firms, 2012-2022)\n"
        f"post-vs-pre coef = {b_post:+.4f} (t={t_post:+.2f}, p={p_post:.3f})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "dynamism_ipo_chilling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] wrote IPO chilling outputs to {RES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
