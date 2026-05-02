"""Pull adjusted-close monthly history for every CIK in the USPTO-SEC
crosswalk that maps to a known ticker, plus the S&P 500 benchmark
(^GSPC). Compute trailing-N-year total returns relative to each
filing-year so we can regress returns on $\\Delta KL$.

Outputs:
  data/processed/stock_prices.parquet  (cik, ticker, year_month, adj_close)
  data/processed/stock_returns.parquet (cik, ticker, base_year, ret_1y..ret_5y, sp500_ret_1y..ret_5y)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"


def main() -> int:
    cw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")
    tickers_json = json.load((REPO_ROOT / "data/raw/sec/company_tickers.json").open())
    cik_to_ticker: dict[int, str] = {int(v["cik_str"]): v["ticker"] for v in tickers_json.values()}
    cik_universe = sorted({int(c) for c in cw["cik"].to_list() if c in cik_to_ticker})
    pairs = [(cik, cik_to_ticker[cik]) for cik in cik_universe]
    print(f"[plan] {len(pairs):,} matched CIKs with tickers", file=sys.stderr)

    rows: list[dict] = []
    benchmark_done = False
    bench_rows: list[dict] = []

    BATCH = 80
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i : i + BATCH]
        symbols = [t for _, t in batch] + (["^GSPC"] if not benchmark_done else [])
        try:
            data = yf.download(
                tickers=" ".join(symbols),
                period="max",
                interval="1mo",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as e:
            print(f"[warn] batch {i}: {e}", file=sys.stderr)
            time.sleep(5)
            continue

        for cik, tkr in batch:
            try:
                df = data[tkr] if tkr in data.columns.get_level_values(0) else None
                if df is None or "Close" not in df.columns:
                    continue
                close = df["Close"].dropna()
                for date, price in close.items():
                    rows.append({
                        "cik": cik,
                        "ticker": tkr,
                        "year_month": str(date)[:7],
                        "adj_close": float(price),
                    })
            except Exception:
                continue
        if not benchmark_done and "^GSPC" in data.columns.get_level_values(0):
            df = data["^GSPC"]
            close = df["Close"].dropna()
            for date, price in close.items():
                bench_rows.append({"year_month": str(date)[:7], "sp500_close": float(price)})
            benchmark_done = True
        print(f"[batch {i//BATCH+1}/{(len(pairs)+BATCH-1)//BATCH}] cumulative price rows: {len(rows):,}", file=sys.stderr, flush=True)
        time.sleep(0.5)

    prices = pl.DataFrame(rows)
    bench = pl.DataFrame(bench_rows)
    prices.write_parquet(PROC / "stock_prices.parquet")
    bench.write_parquet(PROC / "sp500_prices.parquet")
    print(f"[done] {prices.height:,} price rows from {prices['cik'].n_unique():,} CIKs", file=sys.stderr)
    print(f"        S&P 500: {bench.height} months", file=sys.stderr)

    # Build returns table: for each (cik, base_year) compute total return
    # from end of base_year to end of base_year+k for k=1..5
    pdf = prices.with_columns(
        pl.col("year_month").str.slice(0, 4).cast(pl.Int32).alias("year"),
    )
    yearend = (
        pdf.sort("year_month")
        .group_by(["cik", "year"])
        .agg(pl.col("adj_close").last().alias("yearend_close"))
    )
    bench_y = (
        bench.with_columns(pl.col("year_month").str.slice(0, 4).cast(pl.Int32).alias("year"))
        .sort("year_month")
        .group_by("year")
        .agg(pl.col("sp500_close").last().alias("sp500_yearend"))
    )

    returns_rows: list[dict] = []
    by_firm = yearend.to_pandas().sort_values(["cik", "year"])
    bench_d = dict(zip(bench_y["year"].to_list(), bench_y["sp500_yearend"].to_list()))
    for cik, group in by_firm.groupby("cik"):
        g = group.set_index("year")["yearend_close"].to_dict()
        for base_year, base_price in g.items():
            row = {"cik": int(cik), "base_year": int(base_year)}
            for k in (1, 2, 3, 4, 5):
                future_year = base_year + k
                if future_year in g and g[future_year] > 0 and base_price > 0:
                    row[f"ret_{k}y"] = float(g[future_year] / base_price - 1)
                else:
                    row[f"ret_{k}y"] = None
                if future_year in bench_d and base_year in bench_d:
                    row[f"sp500_ret_{k}y"] = float(bench_d[future_year] / bench_d[base_year] - 1)
                else:
                    row[f"sp500_ret_{k}y"] = None
            returns_rows.append(row)

    returns = pl.DataFrame(returns_rows)
    returns.write_parquet(PROC / "stock_returns.parquet")
    print(f"[done] {returns.height:,} (cik, year) return rows -> {PROC}/stock_returns.parquet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
