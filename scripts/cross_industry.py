"""Per-industry summary, ranking, and discussion-figure builder.

Outputs (under paper/results/):
    industry_summary.tex        per-class summary table
    industry_ranks.png          mean firm-level dKL by industry, sorted
    industry_survival_betas.png logit coefficient on dKL for each outcome, by class
    industry_face_validity.tex  top-1 highest-dKL mark per industry
    industry_variance.png       margin variance vs dKL across firms (for matched panel)
    cross_industry_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})


def discover_classes() -> list[str]:
    out = sorted(
        p.stem.replace("outcomes_class", "")
        for p in (REPO_ROOT / "data/processed").glob("outcomes_class*.parquet")
        if p.stem.replace("outcomes_class", "").isdigit()
    )
    return out


def industry_summary(classes: list[str]) -> pl.DataFrame:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    rows: list[dict] = []
    CLEAN = (
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
    )
    for cls in classes:
        df = pl.read_parquet(REPO_ROOT / f"data/processed/outcomes_class{cls}.parquet")
        clean = df.filter(CLEAN & (pl.col("year") >= 1995) & (pl.col("year") <= 2020))
        if clean.is_empty():
            continue
        rows.append(
            {
                "cls": cls,
                "industry": industry_name(cls),
                "n_filings": df.height,
                "n_clean": clean.height,
                "n_owners": clean["owner_name"].n_unique(),
                "year_min": int(clean["year"].min()),
                "year_max": int(clean["year"].max()),
                "median_dkl": float(clean["dkl"].median()),
                "mean_dkl": float(clean["dkl"].mean()),
                "iqr_dkl": float(clean["dkl"].quantile(0.75) - clean["dkl"].quantile(0.25)),
                "p95_dkl": float(clean["dkl"].quantile(0.95)),
                "reach_rate": float(clean["reached_registration"].mean()),
                "surv5_rate": float(clean["survived_5y"].mean()),
            }
        )
    return pl.DataFrame(rows).sort("mean_dkl", descending=True)


def write_summary_tex(summary: pl.DataFrame) -> None:
    """Per-industry summary, with top-3 firms by firm-mean dKL appended.

    A firm is included in the per-industry top-3 if it has at least
    three clean filings in that industry between 1995 and 2020 (so the
    mean isn't driven by a single filing).
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name  # noqa

    CLEAN = (
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
    )

    top_by_cls: dict[str, str] = {}
    for cls in summary["cls"].to_list():
        path = REPO_ROOT / f"data/processed/outcomes_class{cls}.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path).filter(
            CLEAN & (pl.col("year") >= 1995) & (pl.col("year") <= 2020)
            & pl.col("owner_name").is_not_null()
        )
        firm_means = (
            df.group_by("owner_name")
            .agg(
                pl.col("dkl").mean().alias("firm_mean_dkl"),
                pl.len().alias("n_filings"),
            )
            .filter(pl.col("n_filings") >= 3)
            .sort("firm_mean_dkl", descending=True)
            .head(3)
        )
        names = []
        for r in firm_means.iter_rows(named=True):
            display = (r["owner_name"] or "")[:40].replace("&", r"\&")
            names.append(f"{display} ($+{r['firm_mean_dkl']:.2f}$)")
        top_by_cls[cls] = "; ".join(names) if names else "---"

    L = r">{\raggedright\arraybackslash}"
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l rrrr rr rr " + L + "p{8.5cm}}\n\\toprule\n"
        "Industry (NICE) & Filings & Clean & Owners & Years & "
        "$\\Delta KL$ med. & $\\Delta KL$ mean & 5y reg.\\ rate & Reg.\\ rate & "
        "Top 3 firms by firm-mean $\\Delta KL$ ($\\geq 3$ filings) \\\\\n"
        "\\midrule\n"
    )
    for r in summary.iter_rows(named=True):
        ind = r["industry"].replace("&", r"\&")
        top3 = top_by_cls.get(r["cls"], "---")
        tex += (
            f"{ind} ({r['cls']}) & {r['n_filings']:,} & {r['n_clean']:,} & {r['n_owners']:,} & "
            f"{r['year_min']}--{r['year_max']} & {r['median_dkl']:+.2f} & {r['mean_dkl']:+.2f} & "
            f"{r['surv5_rate']*100:.0f}\\% & {r['reach_rate']*100:.0f}\\% & {top3} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}%\n}\n"
    (RESULTS / "industry_summary.tex").write_text(tex)


def fig_industry_ranks(summary: pl.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 11))
    s = summary.sort("mean_dkl", descending=False)
    labels = [f"{r['industry']} ({r['cls']})" for r in s.iter_rows(named=True)]
    means = s["mean_dkl"].to_numpy()
    colors = ["#117a3a" if v > 0 else "#7a1111" for v in means]
    ax.barh(range(len(labels)), means, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.7)
    ax.set_xlabel(r"Industry mean filing-level $\Delta KL$ (1995--2020 cohorts)")
    ax.set_title(r"Industries ranked by mean novelty $\Delta KL$")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(RESULTS / "industry_ranks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_class_survival_beta(classes: list[str]) -> pl.DataFrame:
    rows: list[dict] = []
    CLEAN = (
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 3)
    )
    for cls in classes:
        df = pl.read_parquet(REPO_ROOT / f"data/processed/outcomes_class{cls}.parquet").filter(
            CLEAN & (pl.col("year") >= 1995)
        )
        for outcome, cap in (("reached_registration", 2025), ("survived_5y", 2020), ("survived_10y", 2015)):
            d = df.filter(pl.col("year") <= cap)
            pdf = (
                d.select(
                    pl.col(outcome).cast(pl.Int8),
                    pl.col("dkl"),
                    pl.col("n_terms").log().alias("log_n_terms"),
                    pl.col("year"),
                ).drop_nulls().to_pandas()
            )
            if len(pdf) < 500:
                continue
            pdf["dkl_z"] = (pdf["dkl"] - pdf["dkl"].mean()) / pdf["dkl"].std()
            pdf["year_c"] = pdf["year"] - pdf["year"].mean()
            pdf["year_c2"] = pdf["year_c"] ** 2
            X = sm.add_constant(pdf[["dkl_z", "log_n_terms", "year_c", "year_c2"]])
            try:
                m = sm.GLM(pdf[outcome], X, family=sm.families.Binomial()).fit(disp=False, maxiter=200)
            except Exception as e:
                print(f"  fit failed cls={cls} outcome={outcome}: {e}", file=sys.stderr)
                continue
            rows.append(
                {
                    "cls": cls,
                    "outcome": outcome,
                    "n": int(len(pdf)),
                    "beta_dkl": float(m.params["dkl_z"]),
                    "se_dkl": float(m.bse["dkl_z"]),
                }
            )
    return pl.DataFrame(rows)


def fig_industry_betas(survival_betas: pl.DataFrame, summary: pl.DataFrame) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    fig, axes = plt.subplots(1, 3, figsize=(16, 11), sharey=True)
    panel_outcomes = [
        ("reached_registration", "Completed registration", axes[0]),
        ("survived_5y", "Renewed 5y registration", axes[1]),
        ("survived_10y", "Renewed 10y registration", axes[2]),
    ]
    sort_classes = summary.sort("mean_dkl", descending=False)["cls"].to_list()
    label_for = {c: f"{industry_name(c)} ({c})" for c in sort_classes}
    y_pos = list(range(len(sort_classes)))

    for outcome, title, ax in panel_outcomes:
        rows = survival_betas.filter(pl.col("outcome") == outcome)
        beta_by_cls = {r["cls"]: (r["beta_dkl"], r["se_dkl"]) for r in rows.iter_rows(named=True)}
        xs = []
        es = []
        cs = []
        for cls in sort_classes:
            b, se = beta_by_cls.get(cls, (np.nan, np.nan))
            xs.append(b)
            es.append(1.96 * se if not np.isnan(se) else 0)
            cs.append("#117a3a" if (not np.isnan(b)) and b > 0 else "#7a1111" if (not np.isnan(b)) else "#bbb")
        ax.barh(y_pos, xs, xerr=es, color=cs, alpha=0.85, error_kw={"elinewidth": 0.6, "ecolor": "#444"})
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([label_for[c] for c in sort_classes], fontsize=7)
        ax.set_title(title)
        ax.set_xlabel(r"$\beta_{\Delta KL}^z$ (log-odds per $\sigma$)")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(r"Per-industry effect of novelty $\Delta KL$ on each survival outcome", y=1.005)
    fig.tight_layout()
    fig.savefig(RESULTS / "industry_survival_betas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_face_validity(classes: list[str]) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from novelty.industries import name as industry_name

    CLEAN = (
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms") >= 5)
    )
    rows = []
    for cls in classes:
        sp = pl.read_parquet(REPO_ROOT / f"data/processed/surprise_class{cls}.parquet").with_columns(
            (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl")
        ).filter(CLEAN & (pl.col("year") >= 2010) & (pl.col("year") <= 2020))
        if sp.is_empty():
            continue
        top = sp.sort("dkl", descending=True).head(1)
        if top.is_empty():
            continue
        r = top.row(0, named=True)
        rows.append(
            {
                "cls": cls,
                "industry": industry_name(cls),
                "year": r["year"],
                "mark": (r["mark_identification"] or "")[:32],
                "owner": (r["owner_name"] or "")[:30],
                "dkl": float(r["dkl"]),
            }
        )

    rows.sort(key=lambda r: -r["dkl"])
    L = r">{\raggedright\arraybackslash}"
    tex = (
        "\\begin{tabular}{"
        + f"{L}p{{4.0cm}}rr{L}p{{4.5cm}}{L}p{{4.5cm}}"
        + "}\n\\toprule\n"
        "Industry (NICE) & Year & $\\Delta KL$ & Mark & Owner at filing \\\\\n\\midrule\n"
    )
    for r in rows:
        ind = r["industry"].replace("&", r"\&")
        mark = r["mark"].replace("&", r"\&")
        owner = r["owner"].replace("&", r"\&")
        tex += f"{ind} ({r['cls']}) & {r['year']} & {r['dkl']:+.2f} & {mark} & {owner} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    (RESULTS / "industry_face_validity.tex").write_text(tex)


def main() -> int:
    classes = discover_classes()
    print(f"[plan] {len(classes)} industry classes", file=sys.stderr)

    summary = industry_summary(classes)
    write_summary_tex(summary)
    fig_industry_ranks(summary)

    print("[betas] fitting per-industry logits…", file=sys.stderr, flush=True)
    survival_betas = per_class_survival_beta(classes)
    fig_industry_betas(survival_betas, summary)

    print("[face]  top-dKL marks per industry…", file=sys.stderr, flush=True)
    write_face_validity(classes)

    metrics = {
        "n_classes": len(classes),
        "summary": summary.to_dicts(),
        "survival_betas": survival_betas.to_dicts(),
    }
    (RESULTS / "cross_industry_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[done] wrote cross-industry artifacts to {RESULTS}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
