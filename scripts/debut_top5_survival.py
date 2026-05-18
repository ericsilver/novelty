"""5y survival rate by year for debut filings (owner's first-ever filing),
restricted to filings whose g/s text mentions each of the top-5 most
frequent unigrams across the corpus.

Owner identity = raw owner_name (matches firm_year.py / first_filing.py).
A filing is a debut if filing_date == owner's earliest filing_date across
all classes.

Memory-frugal: pass 1 computes owner->min(filing_date) by streaming each
class's (owner, date) columns; pass 2 streams each class again, marks
debuts via the table from pass 1, and accumulates per-token per-year
sums.

Outputs:
    paper/results/debut_top5_survival.png
    paper/results/debut_top5_survival.csv
"""

from __future__ import annotations

from pathlib import Path
import gc
import polars as pl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"

TOP_TOKENS = ["field", "computer", "featuring", "information", "software"]


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def compute_owner_debut() -> pl.DataFrame:
    """Pass 1: owner_name -> earliest filing_date across all classes."""
    parts = []
    for cls in class_list():
        tm_p = PROC / f"tm_class{cls}.parquet"
        d = pl.read_parquet(tm_p, columns=["owner_name", "filing_date"]).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
        )
        # local min per owner; cheap (8-byte string + 8-char date)
        d = d.group_by("owner_name").agg(pl.col("filing_date").min().alias("debut_date"))
        parts.append(d)
        del d
        gc.collect()
    debut = pl.concat(parts).group_by("owner_name").agg(
        pl.col("debut_date").min().alias("debut_date")
    )
    print(f"[debut] {debut.height:,} unique owners", flush=True)
    return debut


def accumulate(debut: pl.DataFrame) -> pl.DataFrame:
    """Pass 2: per-token, per-year accumulate (n, n_pass5) of debut filings."""
    # rows: (token, year, n, n_pass5)
    accum = {tok: {} for tok in TOP_TOKENS}  # token -> year -> [n, n_pass5]
    for cls in class_list():
        tm_p  = PROC / f"tm_class{cls}.parquet"
        out_p = PROC / f"outcomes_class{cls}.parquet"
        if not out_p.exists():
            continue
        tm = pl.read_parquet(
            tm_p,
            columns=["serial_number", "filing_date", "owner_name", "goods_services"],
        ).filter(
            pl.col("owner_name").is_not_null()
            & pl.col("filing_date").is_not_null()
            & (pl.col("filing_date").str.len_chars() == 8)
            & pl.col("goods_services").is_not_null()
        ).join(debut, on="owner_name", how="left").filter(
            pl.col("filing_date") == pl.col("debut_date")
        ).with_columns(
            pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("year"),
        ).select("serial_number", "year", "goods_services")
        out = pl.read_parquet(
            out_p,
            columns=["serial_number", "reached_registration", "currently_live"],
        )
        j = tm.join(out, on="serial_number", how="inner").with_columns(
            (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
        ).select("year", "goods_services", "passed_5y")
        print(f"  class {cls}: {j.height:>9,} debut filings w/ outcome", flush=True)
        for tok in TOP_TOKENS:
            pat = rf"(?i)\b{tok}\b"
            sub = j.filter(pl.col("goods_services").str.contains(pat)).select("year", "passed_5y")
            if sub.height == 0:
                continue
            ag = sub.group_by("year").agg([
                pl.len().alias("n"),
                pl.col("passed_5y").cast(pl.Int32).sum().alias("n_pass5"),
            ])
            for y, n, np5 in ag.iter_rows():
                d = accum[tok].setdefault(int(y), [0, 0])
                d[0] += int(n); d[1] += int(np5)
        del tm, out, j
        gc.collect()

    rows = []
    for tok, yrs in accum.items():
        for y, (n, np5) in sorted(yrs.items()):
            rows.append({"token": tok, "year": y, "n": n, "n_pass5": np5,
                         "pass5": np5 / n if n else None})
    return pl.from_dicts(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    debut = compute_owner_debut()
    panel = accumulate(debut)
    panel.write_csv(OUT / "debut_top5_survival.csv")
    print(f"[done] wrote {OUT/'debut_top5_survival.csv'}  ({panel.height} rows)", flush=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {"field":"#2b6cb0","computer":"#cc4444","featuring":"#888",
              "information":"#229922","software":"#aa55cc"}
    for tok in TOP_TOKENS:
        p = (panel
             .filter((pl.col("token") == tok)
                     & pl.col("year").is_between(1985, 2021)
                     & (pl.col("n") >= 50))
             .to_pandas())
        if p.empty:
            continue
        ax.plot(p.year, p.pass5, "-o", color=colors[tok], lw=2, ms=4,
                label=f"{tok}  (peak n={int(p.n.max()):,})")
    ax.set_xlabel("Filing year  (debut filings only; cohorts with 5y forward window)")
    ax.set_ylabel("5-year survival rate  (reached registration AND currently live)")
    ax.set_title("Debut-filer 5y survival rate by year — top-5 most frequent unigrams")
    ax.set_ylim(0, 0.65)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.92)
    fig.tight_layout()
    out_png = OUT / "debut_top5_survival.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
