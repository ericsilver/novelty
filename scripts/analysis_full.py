"""End-to-end analysis: produces every figure and table the paper consumes.

Outputs land in paper/results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

from novelty.industries import INDUSTRY_NAME, INDUSTRY_SHORT
from novelty.token_attribution import attribute

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "paper" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

CLASSES = ["009", "032"]
CLEAN = (
    (pl.col("n_ref_prospective") >= 1000)
    & (pl.col("n_ref_retrospective") >= 1000)
    & (pl.col("n_terms") >= 3)
)


def _load_outcomes(cls: str) -> pl.DataFrame:
    return pl.read_parquet(REPO_ROOT / f"data/processed/outcomes_class{cls}.parquet")


def _load_surprise(cls: str) -> pl.DataFrame:
    return pl.read_parquet(REPO_ROOT / f"data/processed/surprise_class{cls}.parquet").with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl")
    )


# ---------- Figure: per-industry 3-panel time series ----------
def fig_industry_time_series():
    for cls in CLASSES:
        sp = _load_surprise(cls).filter(CLEAN & pl.col("year").is_not_null() & (pl.col("year") >= 1980))
        agg = (
            sp.group_by("year")
            .agg(
                pl.len().alias("n"),
                pl.col("prospective_kl").median().alias("pros_med"),
                pl.col("retrospective_kl").median().alias("retr_med"),
                pl.col("dkl").median().alias("dkl_med"),
                pl.col("prospective_kl").quantile(0.25).alias("pros_q1"),
                pl.col("prospective_kl").quantile(0.75).alias("pros_q3"),
                pl.col("retrospective_kl").quantile(0.25).alias("retr_q1"),
                pl.col("retrospective_kl").quantile(0.75).alias("retr_q3"),
                pl.col("dkl").quantile(0.25).alias("dkl_q1"),
                pl.col("dkl").quantile(0.75).alias("dkl_q3"),
            )
            .sort("year")
            .filter(pl.col("n") >= 50)
        )
        fig, axes = plt.subplots(3, 1, figsize=(8, 7.6), sharex=True)
        years = agg["year"].to_numpy()
        for ax, kind, color in zip(axes,
                                   ["pros", "retr", "dkl"],
                                   ["#1f77b4", "#d62728", "#2ca02c"]):
            med = agg[f"{kind}_med"].to_numpy()
            q1 = agg[f"{kind}_q1"].to_numpy()
            q3 = agg[f"{kind}_q3"].to_numpy()
            ax.axvspan(1984, 1995, color="#fde0a8", alpha=0.55, zorder=0)
            ax.axvspan(2021, max(years.max() + 1, 2027), color="#cfe5ff", alpha=0.55, zorder=0)
            ax.fill_between(years, q1, q3, alpha=0.45, color=color, zorder=1,
                            label="25--75% range across filings in year")
            ax.plot(years, med, color=color, linewidth=1.8, zorder=2,
                    label="median across filings in year")
            ax.grid(alpha=0.3, zorder=0)
            ax.set_xlim(1980, max(years.max() + 1, 2027))
        axes[0].set_ylabel("Prospective KL\n(novel vs. past)")
        axes[1].set_ylabel("Retrospective KL\n(novel vs. future)")
        axes[2].set_ylabel(r"$\Delta KL$ (pros$-$retr)")
        axes[2].axhline(0, color="grey", linewidth=0.7, linestyle="--", zorder=1)
        axes[0].set_title(INDUSTRY_NAME[cls])
        axes[2].set_xlabel("Filing year")
        # legend on the third panel only, kept off-plot
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(RESULTS / f"timeseries_{INDUSTRY_SHORT[cls]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------- Figure: 1980s burn-in investigation ----------
def fig_burnin():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.axvspan(1984, 1995, color="#fde0a8", alpha=0.95, zorder=0,
               label="Past-side burn-in (no 5y look-behind available)")
    ax.axvspan(2021, 2027, color="#cfe5ff", alpha=0.95, zorder=0,
               label="Future-side burn-in (no 5y look-ahead available)")
    for cls, color in zip(CLASSES, ["#1f77b4", "#d62728"]):
        sp = _load_surprise(cls).filter(CLEAN & pl.col("year").is_not_null())
        agg = sp.group_by("year").agg(pl.len().alias("n")).sort("year").filter(pl.col("year") >= 1980)
        ax.plot(agg["year"], agg["n"], label=INDUSTRY_NAME[cls], color=color, linewidth=1.6, zorder=2)
    ax.set_yscale("log")
    ax.set_xlim(1980, 2027)
    ax.set_xlabel("Filing year")
    ax.set_ylabel("Filings with both 5y windows populated")
    ax.set_title("Reference-window density by year (log scale)")
    ax.grid(alpha=0.3, zorder=1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "burnin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- Figure: 2x2 quadrant scatter with labelled marks ----------
def fig_quadrant():
    examples = {
        "009": [
            ("OPENAI", 2016),
            ("CLAUDE", 2023),
            ("INSTAGRAM", 2011),
            ("ETHEREUM", 2018),
            ("KUBERNETES", 2014),
            ("CHATGPT", 2022),
            ("UBER", 2014),
            ("WINDOWS", 1990),
            ("PHOTOSHOP", 1992),
            ("AIRPODS", 2015),
            ("KINDLE", 2010),
            ("IPOD", 2001),
        ],
        "032": [
            ("LIQUID DEATH", 2019),  # canned water
            ("LIQUID DEATH", 2023),
            ("WHITE CLAW SELTZER WORKS", 2016),
            ("ROCKSTAR", 2002),
            ("HARD MTN DEW", 2021),
            ("SARATOGA", 2010),       # ordinary bottled water (if present)
            ("FIJI", 2005),           # ordinary bottled water
            ("SMARTWATER", 2010),
            ("ESSENTIA", 2010),
            ("KOMBUCHA", 1997),
            ("POWERADE", 1990),
            ("BUDWEISER", 2010),
            ("MICHELOB", 2005),
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, cls in zip(axes, CLASSES):
        sp = _load_surprise(cls).filter(CLEAN)
        sample = sp.sample(min(15000, sp.height), seed=7)
        ax.scatter(sample["prospective_kl"], sample["retrospective_kl"],
                   s=3, alpha=0.08, color="#888")

        lo, hi = 2.0, 10.0
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, linestyle="--")
        ax.text(hi - 0.1, hi - 0.1, "diagonal", fontsize=8,
                ha="right", va="top", color="#444")

        # Resolve hits, then place labels with simple anti-overlap policy
        placed: list[tuple[float, float, str]] = []  # (x, y, side)
        for mark_q, year_q in examples[cls]:
            hits = (
                sp.filter(
                    (pl.col("year") == year_q)
                    & pl.col("mark_identification").str.to_uppercase().str.contains(
                        f"^{mark_q}$|^{mark_q} "
                    )
                )
                .sort("dkl", descending=True)
                .head(1)
            )
            if hits.is_empty():
                continue
            r = hits.row(0, named=True)
            x, y = float(r["prospective_kl"]), float(r["retrospective_kl"])
            if not (lo <= x <= hi and lo <= y <= hi):
                continue
            ax.scatter([x], [y], s=46, color="#d62728",
                       edgecolor="black", linewidth=0.7, zorder=4)
            label = f"{mark_q} ({year_q})"

            # Try a few offset positions; pick the one with no other label within ~0.5 nat
            offsets = [
                ("ur", 8, 6), ("ul", -8, 6), ("lr", 8, -10), ("ll", -8, -10),
                ("u", 0, 12), ("d", 0, -16), ("r", 14, 0), ("l", -14, 0),
                ("urf", 18, 14), ("ulf", -18, 14), ("lrf", 18, -18), ("llf", -18, -18),
            ]
            best = None
            for tag, dx, dy in offsets:
                # convert pixel offsets to data-coords for collision check
                tx = x + dx * 0.012
                ty = y + dy * 0.012
                if not (lo <= tx <= hi and lo <= ty <= hi):
                    continue
                clash = any((px - tx) ** 2 + (py - ty) ** 2 < 0.16 for px, py, _ in placed)
                if not clash:
                    best = (dx, dy, tx, ty, tag)
                    break
            if best is None:
                best = (offsets[0][1], offsets[0][2], x + 0.1, y + 0.1, "ur")
            dx, dy, tx, ty, tag = best
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"
            ax.annotate(
                label,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                ha=ha,
                va=va,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="#888", linewidth=0.5,
                                shrinkA=2, shrinkB=2) if abs(dx) + abs(dy) > 14 else None,
            )
            placed.append((tx, ty, tag))

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(
            r"Prospective KL --- novelty at filing time"
            "\n(higher $\\rightarrow$ less like the previous 5 years' filings in this category)"
        )
        ax.set_ylabel(
            r"Retrospective KL --- novelty in hindsight"
            "\n(higher $\\rightarrow$ less like the next 5 years' filings in this category)"
        )
        ax.set_title(INDUSTRY_NAME[cls])
        ax.text(0.98, 0.02,
                "innovator\n(below diagonal:\nfuture resembles\nthis filing)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color="#117a3a", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#117a3a", alpha=0.7))
        ax.text(0.02, 0.98,
                "tired\n(above diagonal:\npast resembles\nthis filing)",
                transform=ax.transAxes, va="top", fontsize=8.5, color="#7a1111",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7a1111", alpha=0.7))
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "quadrant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- Figure: outcome bins by KL axis (3 panels per industry) ----------
def fig_outcome_curves():
    outcomes = [
        ("reached_registration", "Completed registration", 2025),
        ("survived_5y", "Renewed 5y registration", 2020),
        ("survived_10y", "Renewed 10y registration", 2015),
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    kl_edges = np.array([0.5, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0])
    kl_centers = 0.5 * (kl_edges[:-1] + kl_edges[1:])
    d_edges = np.array([-2.0, -1.0, -0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])

    def wilson(p, n, z=1.96):
        if n <= 0:
            return p, p
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return center - half, center + half

    out_means = {}
    for cls in CLASSES:
        df = _load_outcomes(cls).filter(CLEAN)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
        panel_specs = [
            ("prospective_kl", "Prospective KL  (novel vs. past)", kl_edges, kl_centers),
            ("retrospective_kl", "Retrospective KL  (novel vs. future)", kl_edges, kl_centers),
            ("dkl", r"$\Delta KL$ (pros $-$ retr)  (innovator $\rightarrow$ right)", d_edges, d_centers),
        ]
        for ax, (xcol, xlabel, edges, centers) in zip(axes, panel_specs):
            for (col, label, max_year), color in zip(outcomes, colors):
                d = df.filter(pl.col("year") <= max_year)
                if d.is_empty():
                    continue
                mean_rate = float(d[col].mean())
                out_means.setdefault(cls, {})[col] = mean_rate
                d_pd = d.select(pl.col(xcol), pl.col(col).cast(pl.Int8)).drop_nulls().to_pandas()
                d_pd["bin"] = np.digitize(d_pd[xcol], edges) - 1
                d_pd = d_pd[(d_pd["bin"] >= 0) & (d_pd["bin"] < len(centers))]
                grp = d_pd.groupby("bin").agg(rate=(col, "mean"), n=(col, "size"))
                grp = grp[grp["n"] >= 50]
                xs = centers[grp.index]
                ys = grp["rate"].values
                lo_arr = np.array([wilson(y, n)[0] for y, n in zip(ys, grp["n"].values)])
                hi_arr = np.array([wilson(y, n)[1] for y, n in zip(ys, grp["n"].values)])
                ax.fill_between(xs, lo_arr, hi_arr, color=color, alpha=0.22, linewidth=0)
                ax.plot(xs, ys, "o-", color=color,
                        label=f"{label} (mean {mean_rate*100:.1f}%)",
                        linewidth=1.6, markersize=5)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Outcome rate")
        axes[2].axvline(0, color="grey", linewidth=0.7, linestyle="--")
        axes[1].set_title(INDUSTRY_NAME[cls])
        axes[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(RESULTS / f"outcome_curves_{INDUSTRY_SHORT[cls]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_means


# ---------- Logistic regressions on all four outcomes ----------
def regressions():
    outcomes = ("reached_registration", "survived_5y", "survived_10y", "currently_live")
    eligibility_cap = {"reached_registration": 2025, "survived_5y": 2020, "survived_10y": 2015, "currently_live": 2025}
    rows = []

    def fit(pdf, predictors, outcome):
        pdf = pdf.copy()
        pdf["year_c"] = pdf["year"] - pdf["year"].mean()
        pdf["year_c2"] = pdf["year_c"] ** 2
        cols = list(predictors) + ["year_c", "year_c2"]
        X = sm.add_constant(pdf[cols])
        return sm.GLM(pdf[outcome], X, family=sm.families.Binomial()).fit(disp=False, maxiter=200)

    for cls in CLASSES:
        df = _load_outcomes(cls).filter(CLEAN).filter(pl.col("year") >= 1995)
        for outcome in outcomes:
            pdf = (
                df.filter(pl.col("year") <= eligibility_cap[outcome])
                .select(
                    pl.col(outcome).cast(pl.Int8),
                    pl.col("dkl"),
                    pl.col("prospective_kl"),
                    pl.col("retrospective_kl"),
                    pl.col("n_terms").log().alias("log_n_terms"),
                    pl.col("year"),
                )
                .drop_nulls()
                .to_pandas()
            )
            if len(pdf) < 1000:
                continue
            for col in ("dkl", "prospective_kl", "retrospective_kl"):
                pdf[f"{col}_z"] = (pdf[col] - pdf[col].mean()) / pdf[col].std()
            m_dkl = fit(pdf, ["dkl_z", "log_n_terms"], outcome)
            m_pr = fit(pdf, ["prospective_kl_z", "retrospective_kl_z", "log_n_terms"], outcome)
            rows.append(
                {
                    "cls": cls,
                    "industry": INDUSTRY_NAME[cls],
                    "outcome": outcome,
                    "n": int(len(pdf)),
                    "rate": float(pdf[outcome].mean()),
                    "coef_dkl_z": float(m_dkl.params["dkl_z"]),
                    "se_dkl_z": float(m_dkl.bse["dkl_z"]),
                    "coef_pros_z": float(m_pr.params["prospective_kl_z"]),
                    "se_pros_z": float(m_pr.bse["prospective_kl_z"]),
                    "coef_retr_z": float(m_pr.params["retrospective_kl_z"]),
                    "se_retr_z": float(m_pr.bse["retrospective_kl_z"]),
                    "coef_log_n_terms": float(m_dkl.params["log_n_terms"]),
                    "se_log_n_terms": float(m_dkl.bse["log_n_terms"]),
                }
            )
    return rows


# ---------- LaTeX tables ----------
def write_summary(out_means: dict):
    rows = []
    for cls in CLASSES:
        df = _load_outcomes(cls).filter(CLEAN)
        rows.append(
            {
                "cls": cls,
                "industry": INDUSTRY_NAME[cls],
                "n_filings": pl.read_parquet(REPO_ROOT / f"data/processed/tm_class{cls}.parquet").height,
                "n_clean": df.height,
                "year_min": int(df["year"].min()),
                "year_max": int(df["year"].max()),
                "n_owners": df["owner_name"].n_unique(),
                "median_dkl": float(df["dkl"].median()),
                "p5_dkl": float(df["dkl"].quantile(0.05)),
                "p95_dkl": float(df["dkl"].quantile(0.95)),
            }
        )
    tex = (
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        "Industry (NICE) & Filings & Clean & Years & Owners & "
        "$\\Delta KL$ med. & p5 & p95 \\\\\n\\midrule\n"
    )
    for r in rows:
        industry = r["industry"].replace("&", r"\&")
        tex += (
            f"{industry} ({r['cls']}) & {r['n_filings']:,} & {r['n_clean']:,} & "
            f"{r['year_min']}--{r['year_max']} & {r['n_owners']:,} & "
            f"{r['median_dkl']:+.2f} & {r['p5_dkl']:+.2f} & {r['p95_dkl']:+.2f} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n"
    (RESULTS / "summary_stats.tex").write_text(tex)
    return rows


def write_regression_table(rows):
    label = {
        "reached_registration": "Completed registration",
        "survived_5y": "Renewed 5y registration",
        "survived_10y": "Renewed 10y registration",
        "currently_live": "Currently live",
    }
    eligibility = {
        "reached_registration": "all cohorts",
        "survived_5y": "filed $\\le$ 2020",
        "survived_10y": "filed $\\le$ 2015",
        "currently_live": "all cohorts",
    }
    rows = [r for r in rows if r["outcome"] != "currently_live"]
    tex = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Industry & Outcome (cohort) & N & "
        "$\\beta$ for $\\Delta KL$ & "
        "$\\beta$ for pros (joint) & $\\beta$ for retr (joint) \\\\\n"
        "\\midrule\n"
    )
    for r in rows:
        industry = r["industry"].replace("&", r"\&")
        tex += (
            f"{industry} & {label[r['outcome']]} ({eligibility[r['outcome']]}) & "
            f"{r['n']:,} & {r['coef_dkl_z']:+.3f} ({r['se_dkl_z']:.3f}) & "
            f"{r['coef_pros_z']:+.3f} ({r['se_pros_z']:.3f}) & "
            f"{r['coef_retr_z']:+.3f} ({r['se_retr_z']:.3f}) \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}%\n}\n"
    (RESULTS / "regression_table.tex").write_text(tex)


def write_token_attribution_table():
    targets = [
        ("009", "OPENAI", 2016),
        ("009", "CLAUDE", 2023),
        ("009", "INSTAGRAM", 2011),
        ("009", "PHOTOSHOP", 1992),
        ("009", "KUBERNETES", 2014),
        ("032", "WHITE CLAW SELTZER WORKS", 2016),
        ("032", "LIQUID DEATH", 2023),
        ("032", "HARD MTN DEW", 2021),
        ("032", "ROCKSTAR", 2002),
    ]

    serials_per_cls: dict[str, list[str]] = {"009": [], "032": []}
    record_meta: dict[str, dict] = {}
    for cls, mark, year in targets:
        sp = _load_surprise(cls)
        rec = pl.read_parquet(REPO_ROOT / f"data/processed/tm_class{cls}.parquet").with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year")
        )
        hits = sp.filter(
            (pl.col("year") == year)
            & pl.col("mark_identification").str.to_uppercase().str.contains(f"^{mark}$|^{mark} ")
        ).sort("dkl", descending=True).head(1)
        if hits.is_empty():
            print(f"  miss: {cls} {mark} {year}")
            continue
        sn = hits.row(0, named=True)["serial_number"]
        serials_per_cls[cls].append(sn)
        record_meta[sn] = {"cls": cls, "mark_query": mark}

    attribs: list = []
    for cls, sns in serials_per_cls.items():
        if not sns:
            continue
        attribs.extend(
            attribute(
                REPO_ROOT / f"data/processed/tm_class{cls}.parquet",
                REPO_ROOT / f"data/processed/surprise_class{cls}.parquet",
                REPO_ROOT / f"data/processed/vocab_class{cls}.parquet",
                sns,
            )
        )

    tex = (
        "\\begin{tabular}{p{2.4cm}p{2.6cm}rrr p{4.5cm} p{4.5cm}}\n\\toprule\n"
        "Mark / Year & Owner at filing & $KL_{\\text{pros}}$ & $KL_{\\text{retr}}$ & "
        "$\\Delta KL$ & Top tokens driving prospective surprise & "
        "Top tokens driving retrospective surprise \\\\\n\\midrule\n"
    )
    body = ""
    text_dump = []
    for a in attribs:
        mark = (a.mark or "").replace("&", r"\&")[:30]
        owner = (a.owner or "").replace("&", r"\&")[:28]
        amp = r"\&"
        pros_terms = [t.replace("&", amp) for t, _ in a.top_pros_terms[:5]]
        retr_terms = [t.replace("&", amp) for t, _ in a.top_retr_terms[:5]]
        pros = ", ".join(r"\texttt{" + t + "}" for t in pros_terms)
        retr = ", ".join(r"\texttt{" + t + "}" for t in retr_terms)
        body += (
            f"{mark} ({a.year}) & {owner} & {a.pros_kl:.2f} & {a.retr_kl:.2f} & "
            f"{a.dkl:+.2f} & {pros} & {retr} \\\\\n"
        )
        text_dump.append(
            {
                "mark": a.mark,
                "owner": a.owner,
                "year": a.year,
                "goods_services": a.goods_services,
                "pros_kl": a.pros_kl,
                "retr_kl": a.retr_kl,
                "dkl": a.dkl,
                "top_pros": a.top_pros_terms,
                "top_retr": a.top_retr_terms,
            }
        )
    tex += body + "\\bottomrule\n\\end{tabular}\n"
    (RESULTS / "token_attribution_table.tex").write_text(tex)

    # Also a long-form text dump for the paper appendix
    long = "\\begin{description}\n"
    for d in text_dump:
        gs = (d["goods_services"][:520] + ("…" if len(d["goods_services"]) > 520 else "")).replace("&", r"\&").replace("_", r"\_").replace("$", r"\$").replace("%", r"\%").replace("#", r"\#")
        owner = (d["owner"] or "").replace("&", r"\&").replace("_", r"\_")
        mark = (d["mark"] or "").replace("&", r"\&").replace("_", r"\_")
        long += (
            f"\\item[\\textbf{{{mark} ({d['year']})}}] "
            f"Owner at filing: \\emph{{{owner}}}. "
            f"$KL_{{\\text{{pros}}}}={d['pros_kl']:.2f}$, $KL_{{\\text{{retr}}}}={d['retr_kl']:.2f}$, "
            f"$\\Delta KL={d['dkl']:+.2f}$.\\\\\n"
            f"Goods/services text: \\emph{{{gs}}}\\\\\n\n"
        )
    long += "\\end{description}\n"
    (RESULTS / "filings_long.tex").write_text(long)


def main() -> int:
    print("==> time series", flush=True)
    fig_industry_time_series()
    print("==> burn-in", flush=True)
    fig_burnin()
    print("==> quadrant", flush=True)
    fig_quadrant()
    print("==> outcome curves", flush=True)
    out_means = fig_outcome_curves()
    print("==> regressions", flush=True)
    rows = regressions()
    print("==> tables", flush=True)
    write_summary(out_means)
    write_regression_table(rows)
    print("==> token attribution", flush=True)
    write_token_attribution_table()

    (RESULTS / "_metrics.json").write_text(
        json.dumps({"out_means": out_means, "regressions": rows}, indent=2, default=str)
    )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
