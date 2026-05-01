"""Join USPTO trademark novelty scores to SEC firm-year financials and ask:
does mean prospective/retrospective surprise predict gross margin?

Outputs:
    paper/results/financials_table.tex
    paper/results/financials_scatter.png
    paper/results/financials_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    cw = pl.read_parquet(REPO_ROOT / "data/processed/uspto_sec_crosswalk.parquet").select(
        "owner_name", "cik", "sec_name"
    )

    fy_parts = []
    for cls in ("009", "032"):
        fy = pl.read_parquet(REPO_ROOT / f"data/processed/firm_year_class{cls}.parquet")
        fy_parts.append(fy.with_columns(pl.lit(cls).alias("nice_class")))
    tm_fy = pl.concat(fy_parts).join(cw, on="owner_name", how="inner")

    tm_cy = (
        tm_fy.group_by(["cik", "year"])
        .agg(
            pl.col("n_filings").sum().alias("n_filings_yr"),
            pl.col("n_filings_clean").sum().alias("n_filings_clean_yr"),
            pl.col("mean_dkl").mean().alias("mean_dkl_yr"),
            pl.col("max_dkl").max().alias("max_dkl_yr"),
            pl.col("mean_pros_kl").mean().alias("mean_pros_yr"),
            pl.col("mean_retr_kl").mean().alias("mean_retr_yr"),
        )
        .sort(["cik", "year"])
    )

    tm_cy_pd = tm_cy.to_pandas().sort_values(["cik", "year"])
    rolling = tm_cy_pd.groupby("cik").rolling(3, on="year", min_periods=1)[[
        "n_filings_yr", "mean_dkl_yr", "max_dkl_yr", "mean_pros_yr", "mean_retr_yr"
    ]].mean().reset_index()
    rolling.columns = ["cik", "year", "n_filings_3y", "mean_dkl_3y", "max_dkl_3y", "mean_pros_3y", "mean_retr_3y"]

    sec = pl.read_parquet(REPO_ROOT / "data/processed/sec_firm_year.parquet").select(
        "cik", "name", "sic", "fy", "revenue", "gross_margin", "rd_expense", "net_income", "operating_income"
    ).rename({"fy": "year", "name": "sec_name"})
    panel = pl.from_pandas(rolling).join(sec, on=["cik", "year"], how="inner")
    panel = panel.filter(
        pl.col("gross_margin").is_not_null()
        & (pl.col("gross_margin") > -1)
        & (pl.col("gross_margin") < 1.5)
        & (pl.col("revenue") > 0)
        & (pl.col("mean_dkl_3y").is_not_null())
    ).with_columns(
        pl.col("revenue").log().alias("log_revenue"),
        ((pl.col("rd_expense") / pl.col("revenue")).fill_null(0)).alias("rd_intensity"),
        (pl.col("net_income") / pl.col("revenue")).alias("net_margin"),
        (pl.col("operating_income") / pl.col("revenue")).alias("op_margin"),
        pl.col("sic").str.slice(0, 2).alias("sic2"),
    )

    print(f"[panel] {panel.height:,} firm-years across {panel['cik'].n_unique():,} CIKs", flush=True)
    pdf = panel.to_pandas()

    def fit(outcome: str, predictors: list[str]) -> dict | None:
        cols = predictors + ["log_revenue", "rd_intensity"]
        df = pdf.dropna(subset=[outcome, *cols, "year"]).copy()
        if len(df) < 200:
            return None
        for c in predictors:
            df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
        df["year_c"] = df["year"] - df["year"].mean()
        df["year_c2"] = df["year_c"] ** 2
        X = sm.add_constant(df[[c + "_z" for c in predictors] + ["log_revenue", "rd_intensity", "year_c", "year_c2"]])
        m = sm.OLS(df[outcome], X).fit(cov_type="cluster", cov_kwds={"groups": df["cik"]})
        out = {
            "outcome": outcome,
            "predictors": predictors,
            "n": int(len(df)),
            "rsq": float(m.rsquared),
        }
        for c in predictors:
            out[f"coef_{c}"] = float(m.params[c + "_z"])
            out[f"se_{c}"] = float(m.bse[c + "_z"])
        out["coef_log_revenue"] = float(m.params["log_revenue"])
        out["se_log_revenue"] = float(m.bse["log_revenue"])
        out["coef_rd_intensity"] = float(m.params["rd_intensity"])
        out["se_rd_intensity"] = float(m.bse["rd_intensity"])
        return out

    rows = []
    for outcome in ("gross_margin", "net_margin", "op_margin"):
        rows.append(fit(outcome, ["mean_dkl_3y"]))
        rows.append(fit(outcome, ["mean_pros_3y", "mean_retr_3y"]))
        rows.append(fit(outcome, ["max_dkl_3y", "n_filings_3y"]))
    rows = [r for r in rows if r]

    outcome_label = {
        "gross_margin": "Gross margin",
        "net_margin": "Net margin",
        "op_margin": "Operating margin",
    }
    pred_label = {
        "mean_dkl_3y": "3y trailing mean novelty $\\Delta KL$",
        "mean_pros_3y": "3y trailing prospective surprise (novel vs.\\ past)",
        "mean_retr_3y": "3y trailing retrospective surprise (novel vs.\\ future)",
        "max_dkl_3y": "3y trailing max novelty",
        "n_filings_3y": "3y trailing average filing count",
    }

    by_outcome: dict[str, list[dict]] = {}
    for r in rows:
        by_outcome.setdefault(r["outcome"], []).append(r)

    body = ""
    for outcome in ("gross_margin", "net_margin", "op_margin"):
        if outcome not in by_outcome:
            continue
        body += (
            f"\\midrule\n"
            f"\\multicolumn{{4}}{{l}}{{\\textit{{Outcome: {outcome_label[outcome]}}}}} \\\\\n"
            f"\\midrule\n"
        )
        for r in by_outcome[outcome]:
            for p in r["predictors"]:
                body += (
                    f"{pred_label.get(p, p)} & "
                    f"{r[f'coef_{p}']:+.3f} & {r[f'se_{p}']:.3f} & "
                    f"{r['n']:,} \\\\\n"
                )
            body += "\\addlinespace[1pt]\n"
    tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Predictor (z-scored unless noted) & Coef.\\ & SE & N \\\\\n"
        + body
        + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "financials_table.tex").write_text(tex)

    # Restrict to firms with Class 9 (software & electronics) trademark filings
    sw_owners = pl.read_parquet(REPO_ROOT / "data/processed/firm_year_class009.parquet")["owner_name"].unique()
    cw = pl.read_parquet(REPO_ROOT / "data/processed/uspto_sec_crosswalk.parquet").filter(
        pl.col("owner_name").is_in(sw_owners)
    )
    sw_ciks = set(cw["cik"].to_list())
    sw_panel = pdf[pdf["cik"].isin(sw_ciks)].copy()
    print(f"[scatter] software-only panel: {len(sw_panel):,} firm-years across {sw_panel['cik'].nunique():,} CIKs")

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.0))
    s = sw_panel.dropna(subset=["mean_dkl_3y", "gross_margin"]).copy()
    s = s[(s["gross_margin"] >= 0) & (s["gross_margin"] <= 1.0)]
    s = s[(s["mean_dkl_3y"] >= s["mean_dkl_3y"].quantile(0.01)) &
          (s["mean_dkl_3y"] <= s["mean_dkl_3y"].quantile(0.99))]
    ax.scatter(s["mean_dkl_3y"], s["gross_margin"], s=8, alpha=0.18, color="#888")

    bins = np.quantile(s["mean_dkl_3y"], np.linspace(0, 1, 16))
    s["bin"] = np.digitize(s["mean_dkl_3y"], bins[1:-1])
    rng = np.random.default_rng(7)
    bin_summary = []
    for b, group in s.groupby("bin"):
        if len(group) < 30:
            continue
        x_mean = float(group["mean_dkl_3y"].mean())
        y_mean = float(group["gross_margin"].mean())
        # Bootstrap CI for bin mean
        boot = np.array([
            rng.choice(group["gross_margin"].values, size=len(group), replace=True).mean()
            for _ in range(400)
        ])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        bin_summary.append((x_mean, y_mean, float(lo), float(hi), len(group)))
    bin_summary.sort()
    xs = np.array([b[0] for b in bin_summary])
    ys = np.array([b[1] for b in bin_summary])
    los = np.array([b[2] for b in bin_summary])
    his = np.array([b[3] for b in bin_summary])
    ax.fill_between(xs, los, his, color="#d62728", alpha=0.18,
                    label="bin mean: 95% bootstrap CI")
    ax.plot(xs, ys, "o-", color="#d62728", linewidth=2.0, markersize=7, label="bin mean")

    # Label notable software firms by exact normalized name match
    label_targets = {
        "MICROSOFT": "Microsoft",
        "APPLE": "Apple",
        "ALPHABET": "Alphabet",
        "GOOGLE": "Google",
        "META PLATFORMS": "Meta",
        "FACEBOOK": "Meta (Facebook)",
        "AMAZON COM": "Amazon",
        "ADOBE": "Adobe",
        "ORACLE": "Oracle",
        "SALESFORCE": "Salesforce",
        "NVIDIA": "NVIDIA",
        "CISCO SYSTEMS": "Cisco",
        "INTEL": "Intel",
        "TESLA": "Tesla",
        "UBER TECHNOLOGIES": "Uber",
        "SNOWFLAKE": "Snowflake",
        "NETFLIX": "Netflix",
        "AIRBNB": "Airbnb",
    }

    sec_recent = sw_panel.dropna(subset=["mean_dkl_3y", "gross_margin"]).copy()
    sec_recent = sec_recent[(sec_recent["year"] >= 2020) & (sec_recent["year"] <= 2024)]
    if "sec_name" in sec_recent.columns:
        sec_recent["sec_norm"] = sec_recent["sec_name"].fillna("").str.upper()
    placed = []
    for needle, label in label_targets.items():
        cand = sec_recent[sec_recent["sec_norm"].str.contains(needle, na=False)]
        if cand.empty:
            continue
        # Take median across years per firm for stable point
        agg = cand.groupby("cik").agg({"mean_dkl_3y": "mean", "gross_margin": "mean"}).reset_index()
        for _, row in agg.iterrows():
            x, y = float(row["mean_dkl_3y"]), float(row["gross_margin"])
            if not (s["mean_dkl_3y"].min() <= x <= s["mean_dkl_3y"].max()): continue
            if not (0 <= y <= 1): continue
            ax.scatter([x], [y], s=60, color="#1f77b4", edgecolor="black", linewidth=0.8, zorder=4)
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, zorder=5)
            placed.append(label)
            break
    print(f"[scatter] placed labels: {placed}")

    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel(r"3-year trailing mean $\Delta KL$  (firm-year)")
    ax.set_ylabel("Gross margin")
    ax.set_title("Software firms only: novelty vs. gross margin")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "financials_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Variance regression: does innovation raise margin variability? ----------
    by_firm = pdf.dropna(subset=["mean_dkl_3y", "gross_margin"]).copy()
    firm_agg = by_firm.groupby("cik").agg(
        firm_mean_dkl=("mean_dkl_3y", "mean"),
        firm_n_filings=("n_filings_3y", "mean"),
        firm_log_rev=("log_revenue", "mean"),
        margin_std=("gross_margin", "std"),
        margin_mean=("gross_margin", "mean"),
        n_years=("gross_margin", "size"),
    ).reset_index()
    firm_agg = firm_agg[(firm_agg["n_years"] >= 4) & firm_agg["margin_std"].notna() & (firm_agg["margin_mean"] > 0)]
    print(f"[variance] {len(firm_agg)} firms with >=4 years for variance regression", flush=True)

    if len(firm_agg) >= 200:
        firm_agg["dkl_z"] = (firm_agg["firm_mean_dkl"] - firm_agg["firm_mean_dkl"].mean()) / firm_agg["firm_mean_dkl"].std()
        Xv = sm.add_constant(firm_agg[["dkl_z", "firm_log_rev"]])
        var_model = sm.OLS(firm_agg["margin_std"], Xv).fit()
        var_result = {
            "n_firms": int(len(firm_agg)),
            "coef_dkl_z": float(var_model.params["dkl_z"]),
            "se_dkl_z": float(var_model.bse["dkl_z"]),
            "coef_log_rev": float(var_model.params["firm_log_rev"]),
            "se_log_rev": float(var_model.bse["firm_log_rev"]),
            "rsq": float(var_model.rsquared),
        }
    else:
        var_result = None

    # Variance regression mini-table
    if var_result is not None:
        var_tex = (
            "\\begin{tabular}{lrrr}\n\\toprule\n"
            "Predictor (z-scored unless noted) & Coef.\\ & SE & N firms \\\\\n"
            "\\midrule\n"
            f"3y trailing mean novelty $\\Delta KL$ (firm avg) & "
            f"{var_result['coef_dkl_z']:+.4f} & {var_result['se_dkl_z']:.4f} & {var_result['n_firms']:,} \\\\\n"
            f"log revenue (firm avg) & "
            f"{var_result['coef_log_rev']:+.4f} & {var_result['se_log_rev']:.4f} & {var_result['n_firms']:,} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
        (RESULTS / "variance_table.tex").write_text(var_tex)

    (RESULTS / "financials_metrics.json").write_text(
        json.dumps({
            "panel_n": panel.height,
            "panel_ciks": int(panel['cik'].n_unique()),
            "regressions": rows,
            "variance": var_result,
        }, indent=2, default=str)
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
