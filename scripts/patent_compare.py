"""Build per-firm-year patent counts from PatentsView and compare with
our trademark-novelty firm-year measures.

Steps:
  1. Read g_assignee_disambiguated.tsv  (patent_id, assignee_id, name).
  2. Read g_patent.tsv                  (patent_id, patent_date).
  3. Join, count patents per (assignee, year).
  4. Crosswalk PatentsView assignee names to USPTO trademark owner
     names via the same normalization we already use in sec_link.py;
     where an exact normalized match exists, link the two.
  5. Aggregate USPTO trademark dKL to (firm, year) and join.
  6. Pearson correlation across firm-years and across firms.

Outputs:
  data/processed/patent_firm_year.parquet
  data/processed/firm_dkl_vs_patents.parquet
  data_publish/firm_year_patents_and_dkl.csv  (the public-facing artifact)
  paper/results/patent_compare_metrics.json
  paper/results/patent_compare_table.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PV_DIR = REPO_ROOT / "data" / "raw" / "patentsview"
PROC = REPO_ROOT / "data" / "processed"
PUBLISH = REPO_ROOT / "data_publish"
RESULTS = REPO_ROOT / "paper" / "results"


# Reuse the normalization that scripts/sec_link.py defines
def _import_normalize():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from sec_link import normalize  # noqa
    return normalize


def main() -> int:
    normalize = _import_normalize()

    out_pv_fy = PROC / "patent_firm_year.parquet"
    if not out_pv_fy.exists():
        print("[load] g_patent.tsv (1 GB)…", file=sys.stderr, flush=True)
        pat = pl.scan_csv(
            PV_DIR / "g_patent.tsv", separator="\t",
            schema_overrides={"patent_id": pl.Utf8, "patent_date": pl.Utf8},
            ignore_errors=True,
        ).select("patent_id", "patent_date").collect()
        print(f"       {pat.height:,} patents", file=sys.stderr)
        pat = pat.with_columns(pl.col("patent_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year")).drop_nulls("year")

        print("[load] g_assignee_disambiguated.tsv (664 MB)…", file=sys.stderr, flush=True)
        # Schema: patent_id, assignee_sequence, assignee_id, disambig_assignee_individual_name_first, ...,
        # disambig_assignee_organization, location_id
        ag = pl.scan_csv(
            PV_DIR / "g_assignee_disambiguated.tsv", separator="\t",
            schema_overrides={"patent_id": pl.Utf8, "assignee_id": pl.Utf8},
            ignore_errors=True,
        ).select("patent_id", "assignee_id", "disambig_assignee_organization").collect()
        ag = ag.filter(
            pl.col("disambig_assignee_organization").is_not_null()
            & (pl.col("disambig_assignee_organization").str.len_chars() > 0)
        )
        print(f"       {ag.height:,} assignee-patent rows", file=sys.stderr)

        joined = ag.join(pat, on="patent_id", how="inner")
        print(f"       {joined.height:,} joined", file=sys.stderr)
        firm_year = (
            joined.group_by(["disambig_assignee_organization", "year"])
            .agg(pl.col("patent_id").n_unique().alias("n_patents"))
            .rename({"disambig_assignee_organization": "patentsview_name"})
        )
        firm_year = firm_year.with_columns(
            pl.col("patentsview_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm")
        )
        firm_year.write_parquet(out_pv_fy)
        print(f"[done] patent firm-years: {firm_year.height:,} rows -> {out_pv_fy}", file=sys.stderr)
    else:
        firm_year = pl.read_parquet(out_pv_fy)
        print(f"[skip] using cached {out_pv_fy} ({firm_year.height:,} rows)", file=sys.stderr)

    # Build trademark dKL firm-year (across all NICE classes, novel definition)
    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        parts.append(pl.read_parquet(path).filter(
            (pl.col("n_ref_prospective") >= 1000)
            & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3)
            & (pl.col("year") >= 1995) & (pl.col("year") <= 2020)
            & pl.col("owner_name").is_not_null()
        ))
    tm = pl.concat(parts)
    tm_fy = tm.group_by(["owner_name", "year"]).agg(
        pl.col("dkl").mean().alias("mean_dkl"),
        pl.col("prospective_kl").mean().alias("mean_pros"),
        pl.col("retrospective_kl").mean().alias("mean_retr"),
        pl.len().alias("n_filings"),
    ).with_columns(
        pl.col("owner_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm")
    )
    print(f"       trademark firm-years: {tm_fy.height:,}", file=sys.stderr)

    # Join on normalized name
    joined = tm_fy.join(
        firm_year.select("norm", "year", "n_patents", "patentsview_name"),
        on=["norm", "year"], how="inner",
    )
    print(f"       joined firm-years: {joined.height:,} (unique norms: {joined['norm'].n_unique():,})", file=sys.stderr)

    joined.write_parquet(PROC / "firm_dkl_vs_patents.parquet")

    # Pearson correlations
    pdf = joined.to_pandas()
    rho_fy = float(pdf["mean_dkl"].corr(pdf["n_patents"]))
    rho_fy_log = float(pdf["mean_dkl"].corr(np.log1p(pdf["n_patents"])))
    by_firm = pdf.groupby("norm").agg(mean_dkl=("mean_dkl", "mean"), total_patents=("n_patents", "sum"), n_years=("year", "nunique")).reset_index()
    by_firm = by_firm[by_firm["n_years"] >= 3]
    rho_firm = float(by_firm["mean_dkl"].corr(by_firm["total_patents"]))
    rho_firm_log = float(by_firm["mean_dkl"].corr(np.log1p(by_firm["total_patents"])))
    print(f"\n[correlation] firm-year level (n={len(pdf):,}): r(dKL, n_patents) = {rho_fy:+.3f};  r(dKL, log1p_patents) = {rho_fy_log:+.3f}")
    print(f"[correlation] firm    level (n={len(by_firm):,}): r(mean_dKL, sum_patents) = {rho_firm:+.3f};  r(mean_dKL, log1p sum) = {rho_firm_log:+.3f}")

    # Top-10 firms by patent count among matched
    print("\n[top-by-patents] (matched firms)")
    print(by_firm.sort_values("total_patents", ascending=False).head(10).to_string(index=False))

    # Public artifact: per-firm-year normalized name + dKL + patent count
    PUBLISH.mkdir(parents=True, exist_ok=True)
    public = joined.select(
        "norm", "owner_name", "patentsview_name",
        "year", "mean_dkl", "mean_pros", "mean_retr",
        "n_filings", "n_patents",
    ).rename({"norm": "normalized_name", "owner_name": "uspto_owner_name"}).sort(["normalized_name", "year"])
    public.write_csv(PUBLISH / "firm_year_patents_and_dkl.csv")
    print(f"\n[publish] wrote {public.height:,} rows -> {PUBLISH/'firm_year_patents_and_dkl.csv'}", file=sys.stderr)

    # LaTeX table
    tex = (
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Aggregation & N & Pearson $r$ \\\\\n\\midrule\n"
        f"Firm-year, $n_{{\\text{{patents}}}}$ & {len(pdf):,} & {rho_fy:+.3f} \\\\\n"
        f"Firm-year, $\\log(1+n_{{\\text{{patents}}}})$ & {len(pdf):,} & {rho_fy_log:+.3f} \\\\\n"
        f"Firm total, sum patents & {len(by_firm):,} & {rho_firm:+.3f} \\\\\n"
        f"Firm total, $\\log(1+\\text{{sum patents}})$ & {len(by_firm):,} & {rho_firm_log:+.3f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "patent_compare_table.tex").write_text(tex)
    (RESULTS / "patent_compare_metrics.json").write_text(json.dumps({
        "joined_firm_years": int(len(pdf)),
        "joined_firms": int(len(by_firm)),
        "r_fy_raw": rho_fy, "r_fy_log": rho_fy_log,
        "r_firm_raw": rho_firm, "r_firm_log": rho_firm_log,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
