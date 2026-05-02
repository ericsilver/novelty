"""Compute correlations between trademark $\\Delta KL$ and four
external public innovation comparators:
  (1) R&D-to-revenue, from SEC FSDS data we already have.
  (2) BCG Most Innovative Companies (50 firms/year, 2021-2023).
  (3) MIT Technology Review 50 Smartest Companies (2015 list).
  (4) Crunchbase total-funding (notpeter/crunchbase-data historical mirror).

For each comparator we report the correlation (or, for binary list
membership, the standardised difference) between matched firms'
mean dKL and the comparator value.

Outputs:
  paper/results/external_compare.tex
  paper/results/external_compare_metrics.json
  data_publish/comparators/external_crosswalk.csv  -- best-guess match
    of every comparator entry to a SEC CIK + matched-USPTO-owner
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
PUBLISH = REPO_ROOT / "data_publish" / "comparators"
RESULTS = REPO_ROOT / "paper" / "results"


def _import_normalize():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from sec_link import normalize  # noqa
    return normalize


def main() -> int:
    normalize = _import_normalize()
    print("[load] our trademark dKL firm-year aggregate (matched to SEC)…", flush=True)
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet").select("owner_name", "norm", "cik", "sec_name")

    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        parts.append(pl.read_parquet(path).filter(
            (pl.col("n_ref_prospective") >= 1000) & (pl.col("n_ref_retrospective") >= 1000)
            & (pl.col("n_terms") >= 3) & (pl.col("year") >= 1995) & (pl.col("year") <= 2020)
            & pl.col("owner_name").is_not_null()
        ))
    tm = pl.concat(parts).join(cw, on="owner_name", how="inner")
    firm_dkl = tm.group_by("cik").agg(
        pl.col("dkl").mean().alias("firm_mean_dkl"),
        pl.col("prospective_kl").mean().alias("firm_mean_pros"),
        pl.len().alias("firm_n_filings"),
        pl.col("sec_name").first(),
    )
    firm_dkl = firm_dkl.with_columns(
        pl.col("sec_name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm")
    )
    print(f"  {firm_dkl.height:,} matched firms with dKL", flush=True)

    metrics: dict = {}
    rows_for_table: list[dict] = []
    crosswalk_rows: list[dict] = []

    # (1) R&D-to-revenue
    print("\n[1] R&D-to-revenue (SEC FSDS)", flush=True)
    sec = pl.read_parquet(PROC / "sec_firm_year.parquet").filter(
        (pl.col("revenue").is_not_null()) & (pl.col("revenue") > 0)
        & pl.col("rd_expense").is_not_null()
    ).with_columns((pl.col("rd_expense") / pl.col("revenue")).alias("rd_intensity"))
    sec = sec.filter((pl.col("rd_intensity") >= 0) & (pl.col("rd_intensity") < 2.0))
    firm_rd = sec.group_by("cik").agg(
        pl.col("rd_intensity").mean().alias("mean_rd_intensity"),
        pl.col("fy").n_unique().alias("n_years"),
    ).filter(pl.col("n_years") >= 3)
    panel_rd = firm_dkl.join(firm_rd, on="cik", how="inner")
    pdf = panel_rd.to_pandas()
    r_dkl = float(pdf["firm_mean_dkl"].corr(pdf["mean_rd_intensity"]))
    r_pros = float(pdf["firm_mean_pros"].corr(pdf["mean_rd_intensity"]))
    print(f"  n={len(pdf):,}  r(dkl, R&D-intensity) = {r_dkl:+.4f}  r(pros, R&D) = {r_pros:+.4f}")
    metrics["rd_vs_dkl"] = {"n": int(len(pdf)), "r_dkl": r_dkl, "r_pros": r_pros}
    rows_for_table.append(("R\\&D-to-revenue (firm-mean)", "Pearson $r$", len(pdf), r_dkl))

    # (2) BCG list — list-membership comparison
    bcg = pl.concat([pl.read_csv(PUBLISH / f"bcg_{y}.csv") for y in (2021, 2022, 2023)])
    # Deduplicate to one row per firm with avg rank
    bcg_uniq = bcg.group_by("company").agg(
        pl.col("rank").mean().alias("avg_rank"),
        pl.col("year").n_unique().alias("n_years_listed"),
    ).with_columns(pl.col("company").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))
    print(f"\n[2] BCG: {bcg_uniq.height:,} unique companies across 2021-2023", flush=True)

    def list_membership(name_list: pl.DataFrame, label: str):
        # Mark firm_dkl rows by whether their normalized name matches a list entry
        matched = firm_dkl.with_columns(
            pl.col("norm").is_in(name_list["norm"].to_list()).alias("on_list")
        )
        on = matched.filter(pl.col("on_list"))
        off = matched.filter(~pl.col("on_list"))
        n_match = on.height
        if n_match == 0:
            print(f"  {label}: 0 matches"); return None
        mean_on = float(on["firm_mean_dkl"].mean())
        mean_off = float(off["firm_mean_dkl"].mean())
        std_off = float(off["firm_mean_dkl"].std())
        z = (mean_on - mean_off) / (std_off / np.sqrt(n_match)) if std_off > 0 else float("nan")
        # Crosswalk: which list entries DID and DIDN'T match
        merged = name_list.join(firm_dkl.select("norm", "sec_name", "cik", "firm_mean_dkl"), on="norm", how="left")
        for r in merged.iter_rows(named=True):
            crosswalk_rows.append({
                "comparator": label,
                "list_company": r.get("company") or "",
                "normalized_name": r["norm"],
                "matched_sec_name": r.get("sec_name") or "",
                "matched_cik": r.get("cik") if r.get("cik") is not None else "",
                "matched_firm_mean_dkl": r.get("firm_mean_dkl") if r.get("firm_mean_dkl") is not None else "",
            })
        match_rate = (merged.filter(pl.col("cik").is_not_null()).height) / merged.height
        print(f"  {label}: list-size {merged.height}  matched to SEC firm dKL: {n_match}  ({match_rate*100:.0f}% of list)")
        print(f"    on-list mean dKL = {mean_on:+.3f},  off-list mean dKL = {mean_off:+.3f}  ({mean_on-mean_off:+.3f}); z = {z:+.2f}")
        return {"label": label, "n_match": n_match, "n_off": int(off.height),
                "mean_on": mean_on, "mean_off": mean_off, "z": z, "list_size": int(merged.height)}

    bcg_res = list_membership(bcg_uniq, "BCG Most Innovative (2021-23 union)")
    if bcg_res:
        metrics["bcg"] = bcg_res
        rows_for_table.append((
            "BCG Most Innovative (2021--23, union)",
            "On-list vs off-list $\\Delta\\overline{\\Delta KL}$",
            bcg_res["n_match"],
            bcg_res["mean_on"] - bcg_res["mean_off"],
        ))

    # (3) MIT TR list
    mit = pl.read_csv(PUBLISH / "mit_tr_2015.csv")
    mit_uniq = mit.group_by("company").agg(pl.col("rank").mean().alias("avg_rank")).with_columns(
        pl.col("company").map_elements(normalize, return_dtype=pl.Utf8).alias("norm")
    )
    mit_res = list_membership(mit_uniq, "MIT TR 50 Smartest Companies (2015)")
    if mit_res:
        metrics["mit"] = mit_res
        rows_for_table.append((
            "MIT TR 50 Smartest (2015)",
            "On-list vs off-list $\\Delta\\overline{\\Delta KL}$",
            mit_res["n_match"],
            mit_res["mean_on"] - mit_res["mean_off"],
        ))

    # (4) Crunchbase total funding
    print("\n[4] Crunchbase historical mirror (notpeter/crunchbase-data)", flush=True)
    cb = pl.read_csv(PUBLISH / "crunchbase_companies.csv", ignore_errors=True).select(
        pl.col("name").cast(pl.Utf8),
        pl.col("funding_total_usd").cast(pl.Utf8).alias("funding_total_usd_raw"),
    ).with_columns(
        pl.col("funding_total_usd_raw").str.replace_all("[^0-9.]", "").cast(pl.Float64, strict=False).alias("funding_total_usd")
    ).drop_nulls("funding_total_usd").filter(pl.col("funding_total_usd") > 0)
    cb = cb.with_columns(pl.col("name").map_elements(normalize, return_dtype=pl.Utf8).alias("norm"))
    cb = cb.group_by("norm").agg(pl.col("funding_total_usd").max().alias("funding_total_usd"))
    panel_cb = firm_dkl.join(cb, on="norm", how="inner")
    pdf_cb = panel_cb.to_pandas()
    pdf_cb["log_funding"] = np.log1p(pdf_cb["funding_total_usd"])
    r_cb_log = float(pdf_cb["firm_mean_dkl"].corr(pdf_cb["log_funding"]))
    r_cb_raw = float(pdf_cb["firm_mean_dkl"].corr(pdf_cb["funding_total_usd"]))
    print(f"  n={len(pdf_cb):,} matched firms")
    print(f"  r(dKL, log(1+funding))    = {r_cb_log:+.4f}")
    print(f"  r(dKL, raw funding total) = {r_cb_raw:+.4f}")
    metrics["crunchbase"] = {"n": len(pdf_cb), "r_log": r_cb_log, "r_raw": r_cb_raw}
    rows_for_table.append(("Crunchbase total funding (log)", "Pearson $r$", len(pdf_cb), r_cb_log))

    # ---- LaTeX summary table ----
    body = ""
    for label, stat, n, val in rows_for_table:
        body += f"{label} & {stat} & {n:,} & {val:+.4f} \\\\\n"
    tex = (
        "\\begin{tabular}{llrr}\n\\toprule\n"
        "Comparator & Statistic & N matched & Value \\\\\n\\midrule\n"
        + body + "\\bottomrule\n\\end{tabular}\n"
    )
    (RESULTS / "external_compare.tex").write_text(tex)
    (RESULTS / "external_compare_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # Crosswalk CSV
    if crosswalk_rows:
        cw_df = pl.DataFrame(crosswalk_rows)
        cw_df.write_csv(PUBLISH / "external_crosswalk.csv")
        print(f"\n[crosswalk] wrote {cw_df.height:,} rows to data_publish/comparators/external_crosswalk.csv")

    print(f"\n[done] external_compare.tex / external_compare_metrics.json / external_crosswalk.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
