"""Memory-streaming version of templated_survival.

Reuses data/processed/gs_template_counts.parquet (computed once by the
v1 script).  Walks the corpus class-by-class, joining each per-class
goods_services frame with that count table and the corrected survival
indicator, then accumulates per-bin counts.

Output:
  paper/results/templated_survival.png
  paper/results/templated_survival.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

CLEAN = (
    (pl.col("n_ref_past") >= 1000)
    & (pl.col("n_ref_future") >= 1000)
    & (pl.col("n_terms") >= 3)
    & (pl.col("year") >= 1990) & (pl.col("year") <= 2018)
)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return p, p
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def _normalise(col: str = "goods_services") -> pl.Expr:
    return (pl.col(col).str.to_lowercase()
            .str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars())


def label_for_count(c: int) -> str:
    if c == 1: return "1 (unique)"
    if c == 2: return "2"
    if c <= 5: return "3-5"
    if c <= 10: return "6-10"
    if c <= 50: return "11-50"
    if c <= 200: return "51-200"
    if c <= 1000: return "201-1000"
    if c <= 5000: return "1001-5000"
    return "5000+"


BIN_ORDER = ["1 (unique)", "2", "3-5", "6-10", "11-50", "51-200",
             "201-1000", "1001-5000", "5000+"]


def main() -> int:
    print("[load] gs_template_counts.parquet (cached)…", flush=True)
    counts = pl.read_parquet(PROC / "gs_template_counts.parquet")
    print(f"  {counts.height:,} unique normalised strings", flush=True)

    # Streaming aggregation: per (count_bin) → n, n_reach, n_passed5
    agg = {b: {"n": 0, "reach": 0, "passed5": 0} for b in BIN_ORDER}
    n_total = 0

    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue

        out = (pl.read_parquet(path)
               .filter(CLEAN)
               .with_columns(
                   (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
               )
               .select("serial_number", "reached_registration", "passed_5y"))

        gs_path = PROC / f"tm_class{cls}.parquet"
        gs = (pl.read_parquet(gs_path)
              .select("serial_number", "goods_services")
              .filter(pl.col("goods_services").is_not_null())
              .with_columns(_normalise().alias("gs_norm"))
              .filter(pl.col("gs_norm").str.len_chars() >= 5)
              .select("serial_number", "gs_norm"))

        joined = (gs.join(out, on="serial_number", how="inner")
                    .join(counts, on="gs_norm", how="left")
                    .filter(pl.col("count").is_not_null())
                    .select("count", "reached_registration", "passed_5y"))

        # Bin breaks aligned with label_for_count
        breaks = [(1, 1, "1 (unique)"),
                  (2, 2, "2"),
                  (3, 5, "3-5"),
                  (6, 10, "6-10"),
                  (11, 50, "11-50"),
                  (51, 200, "51-200"),
                  (201, 1000, "201-1000"),
                  (1001, 5000, "1001-5000"),
                  (5001, 10**9, "5000+")]
        for lo, hi, lbl in breaks:
            sub = joined.filter((pl.col("count") >= lo) & (pl.col("count") <= hi))
            n = sub.height
            if n == 0:
                continue
            agg[lbl]["n"]       += n
            agg[lbl]["reach"]   += int(sub["reached_registration"].sum())
            agg[lbl]["passed5"] += int(sub["passed_5y"].sum())

        n_class = joined.height
        n_total += n_class
        print(f"  class {cls}: {n_class:>9,} clean filings tagged", flush=True)

        # free large objects
        del joined, gs, out

    print(f"\n  total clean filings tagged: {n_total:,}")

    # Build rows + Wilson CIs
    print("\nFiling-level outcomes by template-multiplicity bin:")
    rows = []
    for lbl in BIN_ORDER:
        a = agg[lbl]
        if a["n"] == 0:
            continue
        n = a["n"]
        reach = a["reach"] / n
        surv5 = a["passed5"] / n
        rl, rh = wilson(reach, n)
        sl, sh = wilson(surv5, n)
        rows.append(dict(label=lbl, n=n, reach=reach, surv5=surv5,
                         reach_lo=rl, reach_hi=rh, surv5_lo=sl, surv5_hi=sh))
        print(f"  {lbl:<14} n={n:>10,}  reg={reach*100:5.1f}%  surv5={surv5*100:5.1f}%")

    # Top-50 templates: just print + grab their per-template aggregate by
    # streaming class-by-class once more for that small subset
    print("\n[top50] computing outcomes for the top-50 templates…", flush=True)
    top50_norms = counts.sort("count", descending=True).head(50)
    top50_set = set(top50_norms["gs_norm"].to_list())

    top_acc = {s: {"n": 0, "reach": 0, "passed5": 0} for s in top50_set}

    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit():
            continue
        out = (pl.read_parquet(path).filter(CLEAN)
               .with_columns(
                   (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
               )
               .select("serial_number", "reached_registration", "passed_5y"))
        gs = (pl.read_parquet(PROC / f"tm_class{cls}.parquet")
              .select("serial_number", "goods_services")
              .filter(pl.col("goods_services").is_not_null())
              .with_columns(_normalise().alias("gs_norm"))
              .filter(pl.col("gs_norm").is_in(list(top50_set)))
              .select("serial_number", "gs_norm"))
        if gs.height == 0:
            continue
        merged = gs.join(out, on="serial_number", how="inner").to_pandas()
        for s, sub in merged.groupby("gs_norm"):
            top_acc[s]["n"]       += len(sub)
            top_acc[s]["reach"]   += int(sub["reached_registration"].sum())
            top_acc[s]["passed5"] += int(sub["passed_5y"].sum())

    top50_rows = []
    for s in top50_set:
        a = top_acc[s]
        c = counts.filter(pl.col("gs_norm") == s)["count"].item()
        if a["n"] > 0:
            top50_rows.append(dict(
                gs_norm=s, count_total=c, n_clean=a["n"],
                reach=a["reach"]/a["n"], surv5=a["passed5"]/a["n"]))
    top50_rows.sort(key=lambda r: -r["count_total"])

    print("\nTop-50 most-templated normalised G/S strings (clean cohort):")
    print(f"{'count':>8} {'n_clean':>8} {'reg%':>5} {'surv5%':>6}  | text")
    for r in top50_rows:
        snip = r["gs_norm"][:120].replace("\n", " ")
        print(f"  {int(r['count_total']):>7} {int(r['n_clean']):>7} {r['reach']*100:5.1f} {r['surv5']*100:5.1f}  | {snip}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = np.arange(len(rows))
    surv = np.array([r["surv5"] for r in rows]) * 100
    reach = np.array([r["reach"] for r in rows]) * 100
    surv_err_lo = surv - np.array([r["surv5_lo"] for r in rows]) * 100
    surv_err_hi = np.array([r["surv5_hi"] for r in rows]) * 100 - surv
    reach_err_lo = reach - np.array([r["reach_lo"] for r in rows]) * 100
    reach_err_hi = np.array([r["reach_hi"] for r in rows]) * 100 - reach
    width = 0.4
    ax.bar(xs - width/2, reach, width, yerr=[reach_err_lo, reach_err_hi],
           color="#d62728", alpha=0.85, label="Completed registration", capsize=3)
    ax.bar(xs + width/2, surv, width, yerr=[surv_err_lo, surv_err_hi],
           color="#1f77b4", alpha=0.85, label="Still live (registered \\& not cancelled)", capsize=3)
    for i, r in enumerate(rows):
        ax.text(i, max(r["reach"], r["surv5"]) * 100 + 1.5,
                f"n={r['n']:,}", ha="center", va="bottom", fontsize=8, rotation=20)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("Outcome rate (%)")
    ax.set_xlabel("Filings sharing the exact (normalised) G/S text")
    ax.set_title("Templated trademark descriptions: registration \\& post-registration survival\n"
                 f"by multiplicity of identical G/S text  ·  filing years 1990--2018  ·  n={n_total:,} clean filings")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png = RESULTS / "templated_survival.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {out_png}")

    out_json = RESULTS / "templated_survival.json"
    out_json.write_text(json.dumps({
        "n_clean_filings": n_total,
        "n_unique_norm_strings": int(counts.height),
        "by_count_bin": rows,
        "top50": top50_rows,
    }, indent=2, default=str))
    print(f"[done] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
