"""Compute conditional renewal rate (status bucket 6 | reached registration)
for the example tokens cited in the paper's token-level claim, and compare to
the unconditional rate that the legacy main.pdf reported.
"""
from __future__ import annotations

import re
import sys

import polars as pl

REPO = r"C:\shared-secure\research\novelty"
TM = rf"{REPO}\data\processed\tm_class009.parquet"

# Example tokens cited in the paper
NEGATIVE_TAIL = ["typesetting", "dial-up", "dial up", "e-liquid",
                 "annuity", "casualty", "underwriting", "shandy", "porter"]
POSITIVE_TAIL = ["cannabis", "cannabidiol", "cbd", "tetrahydrocannabinol",
                 "blockchain", "non-fungible", "nft", "nfts",
                 "cryptocurrency", "decentralized", "vape", "cartomizers"]
MIDDLE = ["organisational", "conformity", "freighting", "intermediation",
          "producer", "outsource", "systematization"]


def survival_for_token(df: pl.DataFrame, token: str) -> tuple[int, int, int, int]:
    """Return (n_total, n_registered, n_alive_unconditional, n_alive_conditional)."""
    pattern = rf"(?i)\b{re.escape(token)}\b"
    sub = df.filter(
        pl.col("description_of_mark").str.contains(pattern)
        | pl.col("first_use_anywhere_text").str.contains(pattern)
    )
    if sub.height == 0:
        # Fallback: try a less-restrictive field if not found
        return 0, 0, 0, 0

    n_total = sub.height
    n_reached = sub.filter(pl.col("reached_registration")).height
    # unconditional: alive AND age >= 5
    n_alive_unconditional = sub.filter(
        (pl.col("status_bucket") == "6") & ((2026 - pl.col("year")) >= 5)
    ).height
    # conditional: alive | reached registration
    n_alive_conditional = sub.filter(
        pl.col("reached_registration") & (pl.col("status_bucket") == "6")
    ).height
    return n_total, n_reached, n_alive_unconditional, n_alive_conditional


def main() -> int:
    print("loading tm_class009.parquet ...", file=sys.stderr)
    df = pl.read_parquet(TM)
    # Find a text column
    cols = df.columns
    print(f"columns: {[c for c in cols if 'text' in c.lower() or 'descr' in c.lower() or 'goods' in c.lower() or 'mark' in c.lower()][:10]}",
          file=sys.stderr)

    # Use whichever text column matches
    text_cols = [c for c in cols
                 if any(k in c.lower() for k in ("description", "goods", "services"))]
    if not text_cols:
        print("no goods/description column found; columns are:", file=sys.stderr)
        print(cols, file=sys.stderr)
        return 1
    text_col = text_cols[0]
    print(f"using text column: {text_col}", file=sys.stderr)

    df = df.with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"),
        pl.col("status_code").str.slice(0, 1).alias("status_bucket"),
    ).with_columns(
        ((pl.col("status_bucket") == "6") | (pl.col("status_bucket") == "8")).alias("reached_registration"),
    ).filter(
        (pl.col("year") >= 1990) & (pl.col("year") <= 2020)
    )

    rows = []
    for label, tokens in [("NEGATIVE TAIL", NEGATIVE_TAIL),
                         ("POSITIVE TAIL", POSITIVE_TAIL),
                         ("MIDDLE", MIDDLE)]:
        print(f"\n=== {label} ===", file=sys.stderr)
        print(f"{'token':<25}{'n':>9}{'n_reg':>9}{'uncond%':>10}{'cond%':>10}")
        for tok in tokens:
            pattern = rf"(?i)\b{re.escape(tok)}\b"
            sub = df.filter(pl.col(text_col).str.contains(pattern))
            n = sub.height
            if n == 0:
                continue
            n_reg = sub.filter(pl.col("reached_registration")).height
            n_alive_unc = sub.filter(
                (pl.col("status_bucket") == "6") & ((2026 - pl.col("year")) >= 5)
            ).height
            uncond_pct = 100.0 * n_alive_unc / n if n else 0
            cond_pct = 100.0 * n_alive_unc / n_reg if n_reg else 0
            print(f"{tok:<25}{n:>9,}{n_reg:>9,}{uncond_pct:>9.1f}%{cond_pct:>9.1f}%")
            rows.append({"label": label, "token": tok, "n": n, "n_reg": n_reg,
                         "uncond_pct": uncond_pct, "cond_pct": cond_pct})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
