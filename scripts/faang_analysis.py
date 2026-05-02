"""Are the stock-return results driven by a few mega-cap winners
(FAANG-and-friends), or do they survive removing them?

Case-study plot of FAANG firms' per-year mean dKL and cumulative
stock returns; also re-run the returns regression EXCLUDING the
mega-cap cohort to test whether the headline effect is robust.

Outputs (under paper/results/):
  faang_case_study.png
  faang_returns_table.tex
  faang_metrics.json
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

# CIKs of the mega-cap tech / FAANG cohort (and a couple of obvious peers)
MEGACAP = {
    320193: "Apple",
    789019: "Microsoft",
    1018724: "Amazon",
    1326801: "Meta",
    1652044: "Alphabet (Google)",
    1065280: "Netflix",
    1045810: "NVIDIA",
    1318605: "Tesla",
}


def load_filings_with_dkl() -> pl.DataFrame:
    parts = []
    for path in sorted((REPO_ROOT / "data/processed").glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        parts.append(pl.read_parquet(path))
    return pl.concat(parts).filter(
        pl.col("owner_name").is_not_null()
        & (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
        & (pl.col("year") >= 1995) & (pl.col("year") <= 2020)
    )


def main() -> int:
    fil = load_filings_with_dkl()
    cw = pl.read_parquet(REPO_ROOT / "data/processed/uspto_sec_crosswalk.parquet").select("owner_name", "cik")
    rets = pl.read_parquet(REPO_ROOT / "data/processed/stock_returns.parquet")

    matched = fil.join(cw, on="owner_name", how="inner")
    fy = matched.group_by(["cik", "year"]).agg(
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.len().alias("n_filings_yr"),
    ).rename({"year": "base_year"})

    # ---- (1) FAANG case study ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MEGACAP)))
    cumret_traces = {}

    for color, (cik, name) in zip(colors, MEGACAP.items()):
        sub_dkl = fy.filter(pl.col("cik") == cik).sort("base_year")
        sub_ret = rets.filter(pl.col("cik") == cik).sort("base_year")
        if sub_dkl.is_empty() or sub_ret.is_empty():
            continue

        # dKL panel
        axes[0].plot(
            sub_dkl["base_year"], sub_dkl["mean_dkl"],
            "o-", color=color, label=name, linewidth=1.6, markersize=4,
        )

        # Cumulative return panel: chain ret_1y from earliest base_year forward
        ret_pd = sub_ret.select("base_year", "ret_1y").drop_nulls().to_pandas().sort_values("base_year")
        ret_pd["cum"] = (1 + ret_pd["ret_1y"]).cumprod()
        if len(ret_pd) > 0:
            cumret_traces[name] = ret_pd
            axes[1].plot(
                ret_pd["base_year"] + 1,
                ret_pd["cum"],
                "o-", color=color, label=name, linewidth=1.6, markersize=4,
            )

    axes[0].axhline(0, color="grey", linewidth=0.6, linestyle="--")
    axes[0].set_ylabel(r"Per-firm-year mean $\Delta KL$")
    axes[0].set_title("FAANG-era mega-cap cohort: novelty trajectory and cumulative price")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False)

    axes[1].set_yscale("log")
    axes[1].set_ylabel("Cumulative price (log scale)\nstart year per firm = 1.0")
    axes[1].set_xlabel("Calendar year")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].axhline(1.0, color="grey", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    fig.savefig(RESULTS / "faang_case_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- (2) Re-run returns regression EXCLUDING the cohort ----
    panel_a = fy.join(rets, on=["cik", "base_year"], how="inner")
    for k in (1, 2, 3, 4, 5):
        panel_a = panel_a.with_columns(
            (pl.col(f"ret_{k}y") - pl.col(f"sp500_ret_{k}y")).alias(f"excess_{k}y")
        )

    debut_firm = (
        matched.sort("filing_date")
        .unique(subset=["owner_name"], keep="first")
        .group_by("cik")
        .agg(
            pl.col("dkl").mean().alias("mean_dkl"),
            pl.col("year").min().alias("base_year"),
        )
    )
    panel_b = debut_firm.join(rets, on=["cik", "base_year"], how="inner")
    for k in (1, 2, 3, 4, 5):
        panel_b = panel_b.with_columns(
            (pl.col(f"ret_{k}y") - pl.col(f"sp500_ret_{k}y")).alias(f"excess_{k}y")
        )

    def run(panel: pl.DataFrame, exclude_megacap: bool, predictor: str = "mean_dkl"):
        df = panel
        if exclude_megacap:
            df = df.filter(~pl.col("cik").is_in(list(MEGACAP)))
        rows = []
        for k in (1, 2, 3, 4, 5):
            d = df.select(predictor, f"excess_{k}y", "base_year", "cik").drop_nulls().to_pandas()
            if len(d) < 200:
                continue
            lo, hi = d[f"excess_{k}y"].quantile([0.01, 0.99])
            d[f"excess_{k}y"] = d[f"excess_{k}y"].clip(lo, hi)
            d[f"{predictor}_z"] = (d[predictor] - d[predictor].mean()) / d[predictor].std()
            d["year_c"] = d["base_year"] - d["base_year"].mean()
            d["year_c2"] = d["year_c"] ** 2
            X = sm.add_constant(d[[f"{predictor}_z", "year_c", "year_c2"]])
            try:
                m = sm.OLS(d[f"excess_{k}y"], X).fit(cov_type="cluster", cov_kwds={"groups": d["cik"]})
                rows.append({
                    "horizon": k,
                    "n": int(len(d)),
                    "n_clusters": int(d["cik"].nunique()),
                    "coef": float(m.params[f"{predictor}_z"]),
                    "se": float(m.bse[f"{predictor}_z"]),
                })
            except Exception as e:
                rows.append({"horizon": k, "error": str(e)})
        return rows

    fy_full = run(panel_a, False)
    fy_excl = run(panel_a, True)
    db_full = run(panel_b, False)
    db_excl = run(panel_b, True)

    body = ""
    for k in (1, 2, 3, 4, 5):
        a = next((r for r in fy_full if r.get("horizon") == k and "coef" in r), None)
        b = next((r for r in fy_excl if r.get("horizon") == k and "coef" in r), None)
        c = next((r for r in db_full if r.get("horizon") == k and "coef" in r), None)
        d = next((r for r in db_excl if r.get("horizon") == k and "coef" in r), None)
        body += (
            f"{k}y & "
            f"{a['coef']:+.4f} ({a['se']:.4f}) & {b['coef']:+.4f} ({b['se']:.4f}) & "
            f"{c['coef']:+.4f} ({c['se']:.4f}) & {d['coef']:+.4f} ({d['se']:.4f}) \\\\\n"
        )
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Horizon & Firm-year (full) & Firm-year (excl.\\ mega-cap) & "
        "Debut (full) & Debut (excl.\\ mega-cap) \\\\\n\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}%\n}\n"
    )
    (RESULTS / "faang_returns_table.tex").write_text(tex)

    print("\n[FAANG/mega-cap robustness check]")
    print(f"firms in cohort: {sorted(MEGACAP.values())}")
    print(f"\n{'horizon':<8}{'fy full':<22}{'fy ex-mega':<22}{'debut full':<22}{'debut ex-mega':<22}")
    for k in (1, 2, 3, 4, 5):
        a = next((r for r in fy_full if r.get("horizon") == k and "coef" in r), None)
        b = next((r for r in fy_excl if r.get("horizon") == k and "coef" in r), None)
        c = next((r for r in db_full if r.get("horizon") == k and "coef" in r), None)
        d = next((r for r in db_excl if r.get("horizon") == k and "coef" in r), None)
        print(f"{k}y      {a['coef']:+.4f} (n={a['n']:>5,})   {b['coef']:+.4f} (n={b['n']:>5,})   {c['coef']:+.4f} (n={c['n']:>4,})   {d['coef']:+.4f} (n={d['n']:>4,})")

    (RESULTS / "faang_metrics.json").write_text(json.dumps({
        "megacap": MEGACAP,
        "fy_full": fy_full, "fy_excl": fy_excl,
        "debut_full": db_full, "debut_excl": db_excl,
    }, indent=2, default=str))
    print(f"\n[done] wrote faang_case_study.png + faang_returns_table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
