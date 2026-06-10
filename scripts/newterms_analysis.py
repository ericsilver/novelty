"""New cross-industry terms: which vocabulary tokens were introduced after the
burn-in, where did they originate, and how did they spread?

Uses the pre-aggregated panel from the MacBook tm-laptop syndication:
    _inbound\\tm-laptop\\data-extras\\processed\\_cct_year_class_token_counts.parquet
schema (year:Int32, token:String, count:UInt64, klass:Int32), 1990-2021, 45 classes.

Definitions
-----------
- Burn-in: 1990..1994 (5 years).
- Study:   1995..2021 (27 years).
- "New term": token whose summed count over burn-in years is exactly 0 AND whose
  summed count over study years is >= MIN_STUDY_COUNT AND appears in at least
  MIN_YEARS_PRESENT distinct study years.
- Origin industry of a term: NICE class with the largest count in the term's first
  two appearance years (smooths a 1-year fluke).
- 5y origin count: total count of token in origin class over the first 5 years
  from first appearance.
- y1..y10: total count across all classes, indexed by years-since-first-appearance.
- HHI (other industries): Herfindahl on count shares restricted to non-origin classes.
- Other-industry total: total count summed across classes != origin.

Outputs
-------
- paper/results/newterms_top100.csv  (full top-100 table)
- paper/results/newterms_top100.tex  (LaTeX-ready table fragment for the PDF)
- paper/results/newterms_top10/*.png (10 line charts)
- paper/results/newterms_overall.png (overall utilization chart)
- paper/results/newterms_meta.json   (run metadata)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(r"C:\shared-secure\research\novelty")
SRC_PARQ = Path(r"C:\shared-secure\_inbound\tm-laptop\data-extras\processed\_cct_year_class_token_counts.parquet")
RESULTS = REPO / "paper" / "results"
OUT_CHART_DIR = RESULTS / "newterms_top10"

BURN_LO, BURN_HI = 1990, 1994
STUDY_LO, STUDY_HI = 1995, 2021
MIN_STUDY_COUNT = 200
MIN_YEARS_PRESENT = 5
ORIGIN_WINDOW_YEARS = 2
TOP_N_TABLE = 100
TOP_N_CHARTS = 10
LINES_PER_CHART = 5  # top 5 industries per line-chart


def industry_name_map() -> dict[int, str]:
    sys.path.insert(0, str(REPO / "src"))
    from novelty.industries import INDUSTRY_NAME
    return {int(k): v for k, v in INDUSTRY_NAME.items()}


def main() -> int:
    t0 = time.time()
    OUT_CHART_DIR.mkdir(parents=True, exist_ok=True)
    ind_name = industry_name_map()

    print(f"[load] {SRC_PARQ}", file=sys.stderr)
    df = pl.read_parquet(SRC_PARQ)
    print(f"       {df.height:,} rows; years {df['year'].min()}..{df['year'].max()}; "
          f"{df['token'].n_unique():,} tokens; {df['klass'].n_unique()} classes  "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)

    burn = df.filter((pl.col("year") >= BURN_LO) & (pl.col("year") <= BURN_HI))
    study = df.filter((pl.col("year") >= STUDY_LO) & (pl.col("year") <= STUDY_HI))

    burn_tok = burn.group_by("token").agg(pl.col("count").sum().alias("burn_cnt"))
    study_tok = study.group_by("token").agg(
        pl.col("count").sum().alias("study_cnt"),
        pl.col("year").n_unique().alias("n_years"),
    )

    joined = study_tok.join(burn_tok, on="token", how="left").with_columns(
        pl.col("burn_cnt").fill_null(0)
    )
    new_tok = joined.filter(
        (pl.col("burn_cnt") == 0)
        & (pl.col("study_cnt") >= MIN_STUDY_COUNT)
        & (pl.col("n_years") >= MIN_YEARS_PRESENT)
    ).sort("study_cnt", descending=True)
    print(f"[new ] {new_tok.height:,} tokens meeting new-term criteria  "
          f"({time.time()-t0:.1f}s)", file=sys.stderr)

    top = new_tok.head(TOP_N_TABLE)
    tokens = top["token"].to_list()

    # per (token, year, klass) for top tokens
    sub = study.filter(pl.col("token").is_in(tokens))

    # First appearance year per token
    first_year = sub.group_by("token").agg(pl.col("year").min().alias("first_year"))
    sub2 = sub.join(first_year, on="token")

    # Origin: argmax-count klass in first ORIGIN_WINDOW_YEARS
    orig_window = sub2.filter(pl.col("year") <= pl.col("first_year") + (ORIGIN_WINDOW_YEARS - 1))
    origin = (orig_window.group_by(["token", "klass"]).agg(pl.col("count").sum().alias("cnt"))
                          .sort(["token", "cnt"], descending=[False, True])
                          .group_by("token", maintain_order=True).first()
                          .select(pl.col("token"), pl.col("klass").alias("origin_klass")))
    sub3 = sub2.join(origin, on="token")

    # 5-year origin count
    origin5 = (sub3.filter((pl.col("year") <= pl.col("first_year") + 4)
                            & (pl.col("klass") == pl.col("origin_klass")))
                    .group_by("token").agg(pl.col("count").sum().alias("origin_5y")))

    # y1..y10 (total counts across all classes, indexed by years-since-first)
    y_series = (sub3.with_columns((pl.col("year") - pl.col("first_year")).alias("y_idx"))
                     .filter((pl.col("y_idx") >= 0) & (pl.col("y_idx") < 10))
                     .group_by(["token", "y_idx"]).agg(pl.col("count").sum().alias("cnt")))
    # pivot to wide
    y_wide = y_series.pivot(values="cnt", index="token", on="y_idx", aggregate_function="sum")
    rename_map = {str(i): f"y{i+1}" for i in range(10)}
    y_wide = y_wide.rename({k: v for k, v in rename_map.items() if k in y_wide.columns})
    # ensure all y1..y10 columns exist
    for i in range(1, 11):
        col = f"y{i}"
        if col not in y_wide.columns:
            y_wide = y_wide.with_columns(pl.lit(0).alias(col))
    y_wide = y_wide.select(["token"] + [f"y{i}" for i in range(1, 11)]).fill_null(0)

    # Other-industries (klass != origin): HHI and total
    other = sub3.filter(pl.col("klass") != pl.col("origin_klass"))
    other_by_k = other.group_by(["token", "klass"]).agg(pl.col("count").sum().alias("cnt"))
    other_totals = other_by_k.group_by("token").agg(pl.col("cnt").sum().alias("other_total"))
    hhi = (other_by_k.join(other_totals, on="token")
                     .with_columns((pl.col("cnt") / pl.col("other_total")).alias("share"))
                     .group_by("token").agg((pl.col("share") ** 2).sum().alias("other_hhi")))
    other_totals = other_totals.join(hhi, on="token", how="left").with_columns(
        pl.col("other_hhi").fill_null(0.0)
    )

    # Compose the top-100 table
    tbl = (top.select(pl.col("token").alias("theme"),
                       pl.col("study_cnt").alias("total_study_cnt"))
               .join(first_year, left_on="theme", right_on="token")
               .join(origin, left_on="theme", right_on="token")
               .join(origin5, left_on="theme", right_on="token", how="left")
               .join(y_wide, left_on="theme", right_on="token", how="left")
               .join(other_totals, left_on="theme", right_on="token", how="left"))
    tbl = tbl.with_columns(
        pl.col("origin_klass").map_elements(
            lambda k: ind_name.get(int(k), f"class {k:03d}"), return_dtype=pl.Utf8
        ).alias("origin_industry"),
        pl.col("origin_5y").fill_null(0),
        pl.col("other_total").fill_null(0),
        pl.col("other_hhi").fill_null(0.0),
    )
    final = tbl.select(
        "theme", "first_year", "origin_klass", "origin_industry",
        "origin_5y", *(f"y{i}" for i in range(1, 11)),
        "other_hhi", "other_total", "total_study_cnt",
    ).sort("total_study_cnt", descending=True)

    final.write_csv(RESULTS / "newterms_top100.csv")
    print(f"[tbl ] wrote {final.height} rows -> newterms_top100.csv", file=sys.stderr)

    # ---- LaTeX table (compact) ------------------------------------------------
    def latex_escape(s: str) -> str:
        return (str(s).replace("&", "\\&").replace("_", "\\_").replace("#", "\\#")
                       .replace("$", "\\$").replace("%", "\\%"))

    rows_tex = []
    for r in final.iter_rows(named=True):
        ys = "/".join(f"{int(r[f'y{i}']):,}" for i in range(1, 11))
        rows_tex.append(
            f"{latex_escape(r['theme'])} & {r['first_year']} & "
            f"{int(r['origin_klass']):03d} {latex_escape(r['origin_industry'])[:22]} & "
            f"{int(r['origin_5y']):,} & {ys} & "
            f"{r['other_hhi']:.2f} & {int(r['other_total']):,} & "
            f"{int(r['total_study_cnt']):,} \\\\"
        )
    tex_table = (
        "{\\scriptsize\n"
        "\\begin{tabular}{llp{3.2cm}rrp{3.4cm}rrr}\n\\toprule\n"
        "theme & born & origin industry & 5y orig & y1/.../y10 & HHI\\textsubscript{other} & other total & total \\\\\n"
        "\\midrule\n" + "\n".join(rows_tex) + "\n\\bottomrule\n\\end{tabular}}\n"
    )
    (RESULTS / "newterms_top100.tex").write_text(tex_table, encoding="utf-8")

    # ---- Top-10 per-industry line charts -------------------------------------
    top10 = final.head(TOP_N_CHARTS)["theme"].to_list()
    for theme in top10:
        ts = (study.filter(pl.col("token") == theme)
                     .group_by(["year", "klass"]).agg(pl.col("count").sum().alias("cnt")))
        if ts.is_empty():
            continue
        # find top LINES_PER_CHART industries by total
        top_klasses = (ts.group_by("klass").agg(pl.col("cnt").sum().alias("tot"))
                          .sort("tot", descending=True).head(LINES_PER_CHART)["klass"].to_list())
        years_all = sorted(set(ts["year"].to_list()))
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        cmap = matplotlib.colormaps.get_cmap("tab10")
        for i, k in enumerate(top_klasses):
            sk = ts.filter(pl.col("klass") == k).sort("year").to_pandas()
            sk = sk.set_index("year").reindex(years_all, fill_value=0)
            ax.plot(sk.index, sk["cnt"], marker="o", markersize=2, lw=1.2,
                    color=cmap(i % 10),
                    label=f"{int(k):03d} {ind_name.get(int(k), '')[:18]}")
        ax.set_title(f"\"{theme}\" -- yearly registrations by industry (top {LINES_PER_CHART})")
        ax.set_xlabel("year"); ax.set_ylabel("filings with token")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=.3)
        fig.tight_layout()
        safe = theme.replace("/", "_").replace(" ", "_")[:40]
        out = OUT_CHART_DIR / f"{safe}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[plot] wrote {TOP_N_CHARTS} per-theme charts -> {OUT_CHART_DIR}", file=sys.stderr)

    # ---- Overall chart (top-10 total utilization, color = origin industry) ---
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    rows = final.head(TOP_N_CHARTS).to_pandas()
    xs = np.arange(len(rows))
    origin_classes = rows["origin_klass"].astype(int).to_numpy()
    uniq_klass = sorted(set(origin_classes.tolist()))
    color_for_klass = {k: matplotlib.colormaps.get_cmap("tab10")(i % 10)
                       for i, k in enumerate(uniq_klass)}
    bars = ax.bar(xs, rows["total_study_cnt"],
                  color=[color_for_klass[k] for k in origin_classes])
    ax.set_xticks(xs); ax.set_xticklabels(rows["theme"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("total registrations (1995--2021)")
    ax.set_title("Top-10 new themes: total cross-industry utilization (color = origin industry)")
    # legend
    handles = [matplotlib.patches.Patch(color=color_for_klass[k],
               label=f"{int(k):03d} {ind_name.get(int(k), '')[:22]}") for k in uniq_klass]
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=1)
    ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(RESULTS / "newterms_overall.png", dpi=150)
    plt.close(fig)

    # ---- meta ----
    meta = {
        "src_parquet": str(SRC_PARQ),
        "burn_years": [BURN_LO, BURN_HI], "study_years": [STUDY_LO, STUDY_HI],
        "min_study_count": MIN_STUDY_COUNT, "min_years_present": MIN_YEARS_PRESENT,
        "origin_window_years": ORIGIN_WINDOW_YEARS,
        "n_candidate_new_terms": int(new_tok.height),
        "n_in_table": int(final.height),
        "top10": top10,
        "elapsed_s": time.time() - t0,
    }
    (RESULTS / "newterms_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] ({time.time()-t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
