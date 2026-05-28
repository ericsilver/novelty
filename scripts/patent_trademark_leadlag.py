"""Patent-to-trademark lead-lag cross-correlation, addressing the
pipeline hypothesis: a firm's patents (the invention stage) lead its
trademark filings (the brand-launch stage) by k > 0 years, with
high-ΔKL trademarks following patents most strongly because a
genuinely novel patented product needs genuinely new launch
vocabulary.

The contemporaneous correlation between firm-year ΔKL and firm-year
patent counts is +0.01 -- but that null is exactly what a
patent→product→trademark pipeline predicts, because the two stages
"block each other out" in any given year. The signal, if present,
lives in the lead-lag structure.

Two timing bases:
  * GRANT-date patent counts -- already in patent_firm_year.parquet.
    Grant lags application by ~2.5 years, so trademark launches
    (often during patent pendency) can PRECEDE the grant. Robustness
    check; ambiguous direction for the pipeline test.
  * APPLICATION-date patent counts -- rebuilt from g_application.tsv.
    This is the invention-timing proxy: pipeline predicts patent_app(t)
    → trademark_ΔKL(t+k), k>0.

Builds outer-join firm × year panel so patent-only and trademark-only
years are visible (the existing firm_dkl_vs_patents.parquet is an
inner join and biases the lead-lag).

Tests:
  T1. Contemporaneous correlation (replicates r=+0.01)
  T2. Lead-lag corr(patents(t), mean_dkl(t+k)) for k = -5..+5
  T3. Decomposition: corr(patents(t), n_positive_dkl_filings(t+k))
      -- does the positive-ΔKL subset of trademarks specifically
      follow patents?
  T4. Within-firm regression with firm fixed effects:
      mean_dkl(t) = α_firm + Σ_k β_k * patents(t-k) + ε

Outputs:
  paper/results/leadlag_firm_year.parquet   -- outer-join panel
  paper/results/leadlag_correlations.csv    -- k, corr (grant + app)
  paper/results/leadlag.png                  -- 4-panel summary
  paper/results/leadlag.txt                  -- headline numbers
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
PV = REPO / "data" / "raw" / "patentsview"
OUT = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(p.stem.replace("firm_year_class","") for p in PROC.glob("firm_year_class*.parquet")
                  if p.stem.replace("firm_year_class","").isdigit())


def trademark_firm_year() -> pl.DataFrame:
    """Aggregate trademark ΔKL across NICE classes per (firm, year).
    Restrict to clean filings (n_terms>=3, finite KLs, n_ref both
    sides >= 1000 -- joined from surprise parquets where available)."""
    print("[tm] aggregating per-class firm-year files …", flush=True)
    parts = []
    for cls in class_list():
        fy = pl.read_parquet(PROC / f"firm_year_class{cls}.parquet",
                              columns=["owner_name","year","n_filings","n_filings_clean","mean_dkl"])
        # We want firm × year, summed across classes
        parts.append(fy.with_columns(pl.lit(cls).alias("class_id")))
    df = pl.concat(parts)
    # Weighted aggregation across classes: weight by n_filings_clean
    fy = df.with_columns(
        (pl.col("mean_dkl") * pl.col("n_filings_clean").cast(pl.Float64)).alias("dkl_sum_w")
    ).group_by(["owner_name","year"]).agg(
        pl.col("n_filings").sum().alias("n_filings"),
        pl.col("n_filings_clean").sum().alias("n_filings_clean"),
        pl.col("dkl_sum_w").sum().alias("dkl_sum_w"),
    ).with_columns(
        (pl.col("dkl_sum_w") / pl.col("n_filings_clean").cast(pl.Float64)).alias("mean_dkl")
    ).drop("dkl_sum_w")
    return fy


def trademark_firm_year_with_positive() -> pl.DataFrame:
    """Same as trademark_firm_year but also count positive-ΔKL filings
    per firm-year (T3 decomposition). Goes back to per-filing surprise
    data."""
    print("[tm] computing positive-ΔKL filing counts per firm-year …", flush=True)
    # Build owner -> serial_number -> dkl
    parts = []
    for cls in class_list():
        rec = PROC / f"tm_class{cls}.parquet"
        sur = PROC / f"surprise_class{cls}.parquet"
        if not (rec.exists() and sur.exists()): continue
        r = pl.read_parquet(rec, columns=["serial_number","owner_name","filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        )
        s = pl.read_parquet(sur, columns=["serial_number","prospective_kl","retrospective_kl",
                                            "n_ref_prospective","n_ref_retrospective","n_terms"]).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
        ).with_columns(
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl")
        ).select("serial_number","dkl")
        j = r.join(s, on="serial_number", how="inner").with_columns(
            pl.col("filing_date").str.slice(0,4).cast(pl.Int32).alias("year"),
            (pl.col("dkl") > 0).cast(pl.Int32).alias("is_pos"),
        )
        parts.append(j.select("owner_name","year","dkl","is_pos"))
        del r, s, j
        gc.collect()
    df = pl.concat(parts)
    return df.group_by(["owner_name","year"]).agg(
        pl.len().alias("n_filings_clean"),
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.col("is_pos").sum().alias("n_positive_dkl_filings"),
    )


def patent_firm_year(date_basis: str = "application") -> pl.DataFrame:
    """Build firm × year patent counts keyed to either application or
    grant date. Uses the disambiguated assignee organization name.
    """
    print(f"[pv] building patent firm-year on {date_basis} date …", flush=True)
    ass = pl.read_csv(PV / "g_assignee_disambiguated.tsv", separator="\t",
                       null_values=[""],
                       schema_overrides={"patent_id": pl.Utf8, "assignee_sequence": pl.Int64,
                                          "assignee_type": pl.Int64}).select(
        "patent_id","assignee_sequence","disambig_assignee_organization"
    ).filter((pl.col("assignee_sequence") == 0) & pl.col("disambig_assignee_organization").is_not_null())

    if date_basis == "grant":
        dates = pl.read_csv(PV / "g_patent.tsv", separator="\t", null_values=[""],
                             schema_overrides={"patent_id": pl.Utf8}).select(
            "patent_id","patent_date"
        ).with_columns(pl.col("patent_date").str.slice(0,4).cast(pl.Int32, strict=False).alias("year")).drop_nulls("year")
    else:  # application
        apps = pl.read_csv(PV / "g_application.tsv", separator="\t", null_values=[""],
                            schema_overrides={"patent_id": pl.Utf8, "application_id": pl.Utf8}).select(
            "patent_id","filing_date"
        )
        # filing_date is YYYY-MM-DD; some old patents have wrong-format dates (e.g. 1074-09-23).
        # Filter to plausible 19xx/20xx years.
        apps = apps.with_columns(
            pl.col("filing_date").str.slice(0,4).cast(pl.Int32, strict=False).alias("year")
        ).filter(pl.col("year").is_between(1900, 2030)).drop_nulls("year")
        dates = apps.select("patent_id","year")

    j = ass.join(dates, on="patent_id", how="inner").filter(pl.col("year").is_between(1980, 2024))
    pfy = j.group_by(["disambig_assignee_organization","year"]).agg(pl.len().alias(f"n_patents_{date_basis}"))
    pfy = pfy.with_columns(pl.col("disambig_assignee_organization").str.to_uppercase().alias("norm"))
    return pfy


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: trademark firm-year with positive-ΔKL count
    tm = trademark_firm_year_with_positive()
    tm = tm.with_columns(pl.col("owner_name").str.to_uppercase().alias("norm"))
    print(f"      tm firm-years: {tm.height:,}")

    # ----- Step 2: patent firm-year on BOTH date bases
    pv_grant = patent_firm_year("grant")
    pv_app = patent_firm_year("application")
    print(f"      pv firm-years (grant):       {pv_grant.height:,}")
    print(f"      pv firm-years (application): {pv_app.height:,}")

    # ----- Step 3: name crosswalk — use the existing one from
    # firm_dkl_vs_patents.parquet so we don't redo name matching
    xw = pl.read_parquet(PROC / "firm_dkl_vs_patents.parquet",
                          columns=["owner_name","patentsview_name","norm"]).unique()
    print(f"      name crosswalk: {xw.height:,} firm-pairs")

    # Use the same `norm` field on both sides, mapped via xw
    # The xw has owner_name → patentsview_name; pv side has disambig_assignee_org → norm (uppercased).
    # The norm on the crosswalk matches the uppercase of patentsview_name (we'll verify).
    xw2 = xw.with_columns(
        pl.col("patentsview_name").str.to_uppercase().alias("pv_norm")
    ).select("norm","pv_norm")

    # Map pv firm-years to the `norm` space via pv_norm
    pv_grant_n = pv_grant.join(xw2, left_on="norm", right_on="pv_norm", how="inner").select(
        pl.col("norm_right").alias("norm"), "year", "n_patents_grant"
    )
    pv_app_n = pv_app.join(xw2, left_on="norm", right_on="pv_norm", how="inner").select(
        pl.col("norm_right").alias("norm"), "year", "n_patents_application"
    )
    print(f"      pv firm-years on tm-norm space (grant): {pv_grant_n.height:,}")
    print(f"      pv firm-years on tm-norm space (app):   {pv_app_n.height:,}")

    # Outer-join: keep all (norm, year) cells that appear in EITHER tm or pv (either basis)
    print("[merge] outer-join firm × year …", flush=True)
    all_norms = pl.concat([
        tm.select("norm").unique(),
        pv_grant_n.select("norm").unique(),
        pv_app_n.select("norm").unique(),
    ]).unique()
    print(f"      distinct firms in any side: {all_norms.height:,}")

    # Build the panel by joining all three
    panel = (tm.select("norm","year","n_filings_clean","mean_dkl","n_positive_dkl_filings")
                .join(pv_grant_n, on=["norm","year"], how="full", coalesce=True)
                .join(pv_app_n,   on=["norm","year"], how="full", coalesce=True))
    # Fill missing counts with 0; mean_dkl stays NA if no trademarks that year
    panel = panel.with_columns(
        pl.col("n_filings_clean").fill_null(0),
        pl.col("n_positive_dkl_filings").fill_null(0),
        pl.col("n_patents_grant").fill_null(0),
        pl.col("n_patents_application").fill_null(0),
    ).filter(pl.col("year").is_between(1990, 2024))
    panel.write_parquet(OUT / "leadlag_firm_year.parquet")
    print(f"      panel rows: {panel.height:,}")

    # ----- Step 4: Test T1 -- contemporaneous correlation (replicate r=+0.01)
    pdf = panel.to_pandas()
    pdf_has_tm = pdf[pdf["n_filings_clean"] >= 1]
    pdf_has_dkl = pdf[~pdf["mean_dkl"].isna()]
    print()
    print("=== T1: Contemporaneous (replicates the r=+0.01 finding) ===")
    print(f"  Corr(n_patents_grant, mean_dkl):       r = {pdf_has_dkl['n_patents_grant'].corr(pdf_has_dkl['mean_dkl']):+.4f}")
    print(f"  Corr(n_patents_application, mean_dkl): r = {pdf_has_dkl['n_patents_application'].corr(pdf_has_dkl['mean_dkl']):+.4f}")

    # ----- Step 5: Test T2 -- lead-lag cross-correlation
    print()
    print("=== T2: Lead-lag cross-correlation ===")
    # For each k in -5..+5 we want corr(patents(t), mean_dkl(t+k))
    # Build by self-joining on shifted years
    correlations = []
    pdf_sorted = pdf.sort_values(["norm","year"])
    for k in range(-5, 6):
        # Build (norm, year) → mean_dkl(year+k) by merging shifted
        shifted = pdf_sorted[["norm","year","mean_dkl","n_positive_dkl_filings","n_filings_clean"]].copy()
        shifted["year_orig"] = shifted["year"] - k  # so when we merge on year, the patents at year_orig pair with dkl at year_orig+k
        m = pdf_sorted[["norm","year","n_patents_grant","n_patents_application"]].merge(
            shifted, left_on=["norm","year"], right_on=["norm","year_orig"], how="inner",
            suffixes=("","_shifted"))
        # Keep rows where the shifted side has a mean_dkl (i.e., the firm had a TM filing at year+k)
        m = m.dropna(subset=["mean_dkl"])
        if len(m) < 100: continue
        r_grant = m["n_patents_grant"].corr(m["mean_dkl"])
        r_app   = m["n_patents_application"].corr(m["mean_dkl"])
        r_grant_pos = m["n_patents_grant"].corr(m["n_positive_dkl_filings"])
        r_app_pos   = m["n_patents_application"].corr(m["n_positive_dkl_filings"])
        correlations.append({"k": k, "n": len(m),
                              "r_dkl_grant": r_grant, "r_dkl_app": r_app,
                              "r_pos_grant": r_grant_pos, "r_pos_app": r_app_pos})
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(OUT / "leadlag_correlations.csv", index=False)
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ----- Step 6: Test T4 -- within-firm regression with firm FE
    # mean_dkl(t) = α_firm + β·patents_app(t-1) + β·patents_app(t-2) + β·patents_app(t-3) + ε
    print()
    print("=== T4: Within-firm regression with firm FE (application date) ===")
    # Build the regression panel: for each (norm, year) where mean_dkl exists,
    # attach patents_app(t-1), patents_app(t-2), patents_app(t-3)
    # Construct lags via self-merge
    p = pdf_sorted[["norm","year","mean_dkl","n_patents_application"]].copy()
    for lag in [0, 1, 2, 3]:
        s = p[["norm","year","n_patents_application"]].copy()
        s["year"] = s["year"] + lag  # so when merged at year, this carries patents at year-lag
        s = s.rename(columns={"n_patents_application": f"pat_lag{lag}"})
        p = p.merge(s[["norm","year",f"pat_lag{lag}"]], on=["norm","year"], how="left")
    p = p.dropna(subset=["mean_dkl"])
    # Restrict to firms with >=5 firm-years with TM ΔKL
    counts = p.groupby("norm")["mean_dkl"].count()
    keep = counts[counts >= 5].index
    p_reg = p[p["norm"].isin(keep)].copy()
    # Lags 0..3 with fill_na 0
    for lag in [0, 1, 2, 3]:
        p_reg[f"pat_lag{lag}"] = p_reg[f"pat_lag{lag}"].fillna(0)
    # Within transform: subtract firm mean from each variable
    firm_means = p_reg.groupby("norm")[["mean_dkl","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]].transform("mean")
    p_w = p_reg[["mean_dkl","pat_lag0","pat_lag1","pat_lag2","pat_lag3"]] - firm_means
    p_w = p_w.dropna()
    X = sm.add_constant(p_w[["pat_lag0","pat_lag1","pat_lag2","pat_lag3"]])
    y = p_w["mean_dkl"]
    mod = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": p_reg.loc[p_w.index, "norm"]})
    print(mod.summary().tables[1])
    print(f"  N firms = {p_reg['norm'].nunique():,}, N firm-years = {len(p_w):,}")

    # ----- Step 7: 4-panel summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: cross-correlation, mean_dkl
    ax = axes[0, 0]
    ax.plot(corr_df["k"], corr_df["r_dkl_grant"], "-o", color="#cc4444", lw=2, ms=6,
            label="patents (grant date) → mean ΔKL")
    ax.plot(corr_df["k"], corr_df["r_dkl_app"], "-s", color="#2b6cb0", lw=2, ms=6,
            label="patents (application date) → mean ΔKL")
    ax.axvline(0, color="#444", ls=":", lw=0.8)
    ax.axhline(0, color="#444", ls="--", lw=0.8)
    ax.set_xlabel("Lag k (years; patent at t, ΔKL at t+k)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("T2: Patents → trademark mean ΔKL")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Panel B: cross-correlation, n_positive_dkl_filings (the "new vocabulary" trademarks)
    ax = axes[0, 1]
    ax.plot(corr_df["k"], corr_df["r_pos_grant"], "-o", color="#cc4444", lw=2, ms=6,
            label="patents (grant) → n positive-ΔKL filings")
    ax.plot(corr_df["k"], corr_df["r_pos_app"], "-s", color="#2b6cb0", lw=2, ms=6,
            label="patents (application) → n positive-ΔKL filings")
    ax.axvline(0, color="#444", ls=":", lw=0.8)
    ax.axhline(0, color="#444", ls="--", lw=0.8)
    ax.set_xlabel("Lag k (years)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("T3: Patents → count of positive-ΔKL trademark filings")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Panel C: within-firm FE regression coefficients
    ax = axes[1, 0]
    coefs = mod.params[["pat_lag0","pat_lag1","pat_lag2","pat_lag3"]].values
    ses = mod.bse[["pat_lag0","pat_lag1","pat_lag2","pat_lag3"]].values
    lags = [0, 1, 2, 3]
    ax.errorbar(lags, coefs, yerr=1.96*ses, fmt="o-", color="#2b6cb0", lw=2, ms=8,
                capsize=4, capthick=1.2, label="β coefficients (clustered SE)")
    ax.axhline(0, color="#444", ls="--", lw=0.8)
    ax.set_xlabel(r"Lag of patent applications relative to mean $\Delta KL$")
    ax.set_ylabel(r"β: effect on mean $\Delta KL$ of 1 extra application at $t-\mathrm{lag}$")
    ax.set_title("T4: Within-firm FE regression (clustered SE)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Panel D: dual time series for one firm where the pipeline should be visible
    # Pick a firm with lots of both -- e.g., the top firm by patents+filings
    ax = axes[1, 1]
    pdf_sub = panel.to_pandas()
    pdf_sub["both_score"] = pdf_sub["n_patents_application"] + pdf_sub["n_filings_clean"]
    top_norm = pdf_sub.groupby("norm")["both_score"].sum().sort_values(ascending=False).head(8).index.tolist()
    # Pick one with a clean visual story
    pick = top_norm[0]
    one = pdf_sub[pdf_sub["norm"] == pick].sort_values("year")
    ax2 = ax.twinx()
    ax.bar(one["year"], one["n_patents_application"], color="#cc4444", alpha=0.5, label="patent apps (left axis)")
    ax2.plot(one["year"], one["mean_dkl"], "-o", color="#2b6cb0", lw=2, ms=5, label="mean ΔKL (right axis)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Patent applications", color="#cc4444")
    ax2.set_ylabel(r"Mean $\Delta KL$ across firm's filings", color="#2b6cb0")
    ax.set_title(f"Illustrative firm trajectory: {pick}")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Patents lead trademarks: lead-lag cross-correlation between firm-year patent applications and trademark mean ΔKL",
                  fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "leadlag.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----- Write headline summary text
    text = []
    text.append("Patent-to-trademark lead-lag cross-correlation")
    text.append("=" * 70)
    text.append("")
    text.append("PANEL: outer-join firm × year, 1990-2024.")
    text.append(f"  Total firm-years: {panel.height:,}")
    text.append(f"  Firm-years with ≥1 trademark filing (clean):  {len(pdf_has_tm):,}")
    text.append(f"  Firm-years with non-NaN mean_dkl:             {len(pdf_has_dkl):,}")
    text.append("")
    text.append("T1: CONTEMPORANEOUS CORRELATION (replicates r=+0.01)")
    text.append(f"  Corr(n_patents_grant,       mean_dkl) = {pdf_has_dkl['n_patents_grant'].corr(pdf_has_dkl['mean_dkl']):+.4f}")
    text.append(f"  Corr(n_patents_application, mean_dkl) = {pdf_has_dkl['n_patents_application'].corr(pdf_has_dkl['mean_dkl']):+.4f}")
    text.append("")
    text.append("T2: LEAD-LAG CORRELATION (negative k = trademark leads patents)")
    text.append(f"  {'k':>3}  {'n':>9}  {'r_dkl_grant':>12}  {'r_dkl_app':>12}  {'r_pos_grant':>12}  {'r_pos_app':>12}")
    for _, r in corr_df.iterrows():
        text.append(f"  {int(r['k']):>3}  {int(r['n']):>9,}  {r['r_dkl_grant']:>+12.4f}  {r['r_dkl_app']:>+12.4f}  {r['r_pos_grant']:>+12.4f}  {r['r_pos_app']:>+12.4f}")
    text.append("")
    text.append("T4: WITHIN-FIRM FE REGRESSION (application date, lags 0..3)")
    text.append(f"  N firms = {p_reg['norm'].nunique():,}, N firm-years = {len(p_w):,}")
    text.append(str(mod.summary().tables[1]))
    text.append("")
    text.append("Interpretation guide:")
    text.append("  - Pipeline prediction: patents lead high-ΔKL trademarks → r peaks at k>0.")
    text.append("  - If application date r peaks at k=0 or k<0 instead, the pipeline interpretation fails.")
    text.append("  - Within-firm β at lag>0 being POSITIVE and significant is direct evidence")
    text.append("    that a firm's prior-year patent applications predict that firm's later")
    text.append("    mean-ΔKL after controlling for firm-level fixed effects.")
    summary = "\n".join(text)
    (OUT / "leadlag.txt").write_text(summary)
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
