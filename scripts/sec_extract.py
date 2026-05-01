"""Parse SEC FSDS quarterly ZIPs (sub.txt + num.txt) into a slim parquet of
firm-period financials. Emits one row per (cik, fy) with revenue, COGS, gross
profit, R&D, net income, and computed gross margin.

Usage:
    python scripts/sec_extract.py
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "raw" / "sec"
OUT_FIRMS = REPO_ROOT / "data" / "processed" / "sec_firms.parquet"
OUT_FY = REPO_ROOT / "data" / "processed" / "sec_firm_year.parquet"

REVENUE_TAGS = {
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
}
COGS_TAGS = {"CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"}
GP_TAG = "GrossProfit"
RD_TAG = "ResearchAndDevelopmentExpense"
NI_TAG = "NetIncomeLoss"
OI_TAG = "OperatingIncomeLoss"
ASSETS_TAG = "Assets"

NEED = REVENUE_TAGS | COGS_TAGS | {GP_TAG, RD_TAG, NI_TAG, OI_TAG, ASSETS_TAG}


def _read_lines(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
            yield line.rstrip("\n")


def _parse_quarter(zip_path: Path) -> tuple[list[dict], list[dict]]:
    subs = {}
    nums: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        # sub.txt: adsh, cik, name, sic, ..., form, period, fy, fp, filed
        sub_iter = _read_lines(zf, "sub.txt")
        sub_header = next(sub_iter).split("\t")
        ix = {h: i for i, h in enumerate(sub_header)}
        for line in sub_iter:
            cols = line.split("\t")
            if len(cols) < len(sub_header):
                continue
            form = cols[ix["form"]]
            if not form.startswith("10-K") and not form.startswith("10-Q"):
                continue
            try:
                cik = int(cols[ix["cik"]])
                fy = int(cols[ix["fy"]]) if cols[ix["fy"]] else None
            except ValueError:
                continue
            subs[cols[ix["adsh"]]] = {
                "cik": cik,
                "name": cols[ix["name"]],
                "sic": cols[ix["sic"]],
                "form": form,
                "fy": fy,
                "fp": cols[ix["fp"]],
                "period": cols[ix["period"]],
                "filed": cols[ix["filed"]],
            }

        num_iter = _read_lines(zf, "num.txt")
        num_header = next(num_iter).split("\t")
        nx = {h: i for i, h in enumerate(num_header)}
        for line in num_iter:
            cols = line.split("\t")
            if len(cols) < len(num_header):
                continue
            tag = cols[nx["tag"]]
            if tag not in NEED:
                continue
            adsh = cols[nx["adsh"]]
            if adsh not in subs:
                continue
            qtrs = cols[nx["qtrs"]]
            try:
                value = float(cols[nx["value"]] or "nan")
            except ValueError:
                continue
            seg = cols[nx["segments"]]
            if seg:
                continue  # only consolidated/total
            sub = subs[adsh]
            nums.append(
                {
                    "cik": sub["cik"],
                    "name": sub["name"],
                    "sic": sub["sic"],
                    "form": sub["form"],
                    "fy": sub["fy"],
                    "fp": sub["fp"],
                    "period": sub["period"],
                    "ddate": cols[nx["ddate"]],
                    "qtrs": qtrs,
                    "tag": tag,
                    "value": value,
                }
            )
    return list(subs.values()), nums


def main() -> int:
    zips = sorted(SRC.glob("*.zip"))
    print(f"[plan] {len(zips)} ZIPs to parse", file=sys.stderr)
    all_subs: list[dict] = []
    all_nums: list[dict] = []
    for zp in zips:
        t0 = time.time()
        subs, nums = _parse_quarter(zp)
        all_subs.extend(subs)
        all_nums.extend(nums)
        print(
            f"[parse] {zp.name}: subs={len(subs):,} nums={len(nums):,} ({time.time()-t0:.1f}s)",
            file=sys.stderr,
        )

    nums_df = pl.DataFrame(all_nums)
    print(f"[total] nums rows: {nums_df.height:,}", file=sys.stderr)

    # Annual reporting only: 10-K / 10-K/A and qtrs = 4 (full-year flow) for income statement,
    # qtrs = 0 for balance sheet (point-in-time).
    is_tags = REVENUE_TAGS | COGS_TAGS | {GP_TAG, RD_TAG, NI_TAG, OI_TAG}
    fy_is = (
        nums_df.filter(
            pl.col("form").str.starts_with("10-K")
            & (pl.col("qtrs") == "4")
            & pl.col("tag").is_in(list(is_tags))
            & pl.col("fy").is_not_null()
        )
        .group_by(["cik", "name", "sic", "fy", "tag"])
        .agg(pl.col("value").last())
        .pivot(values="value", index=["cik", "name", "sic", "fy"], on="tag", aggregate_function="last")
    )

    # Pick best revenue tag per row
    rev_cols = [c for c in REVENUE_TAGS if c in fy_is.columns]
    cogs_cols = [c for c in COGS_TAGS if c in fy_is.columns]
    fy_is = fy_is.with_columns(
        pl.coalesce(rev_cols).alias("revenue") if rev_cols else pl.lit(None).alias("revenue"),
        pl.coalesce(cogs_cols).alias("cogs") if cogs_cols else pl.lit(None).alias("cogs"),
    ).with_columns(
        pl.coalesce(["GrossProfit", (pl.col("revenue") - pl.col("cogs"))]).alias("gross_profit") if GP_TAG in fy_is.columns else (pl.col("revenue") - pl.col("cogs")).alias("gross_profit"),
    ).with_columns(
        (pl.col("gross_profit") / pl.col("revenue")).alias("gross_margin"),
    )

    keep_cols = ["cik", "name", "sic", "fy", "revenue", "cogs", "gross_profit", "gross_margin"]
    for t in (RD_TAG, NI_TAG, OI_TAG):
        if t in fy_is.columns:
            keep_cols.append(t)
    fy_out = fy_is.select([c for c in keep_cols if c in fy_is.columns])
    fy_out = fy_out.rename(
        {
            RD_TAG: "rd_expense" if RD_TAG in fy_out.columns else RD_TAG,
            NI_TAG: "net_income" if NI_TAG in fy_out.columns else NI_TAG,
            OI_TAG: "operating_income" if OI_TAG in fy_out.columns else OI_TAG,
        },
        strict=False,
    )

    OUT_FY.parent.mkdir(parents=True, exist_ok=True)
    fy_out.write_parquet(OUT_FY)
    print(
        f"[done] {fy_out.height:,} firm-year rows -> {OUT_FY}  unique CIKs: {fy_out['cik'].n_unique():,}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
