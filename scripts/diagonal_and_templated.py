"""Two diagnostics on the survival topomap:

(1) Diagonal-clustering hypothesis. The topomap's worst survival cells
    sit in the lower-left (commonplace).  Decompose each firm's position
    into:
        mean_KL   = (pros + retr) / 2     (distance up the diagonal)
        abs_dKL   = | pros - retr |        (perpendicular off-diagonal distance)
    Stratify by both and report 5y survival.  If the U/penalty is
    diagonal-driven, low-mean_KL + low-abs_dKL cells should be the worst.

(2) Templated G/S strings.  Group filings by their exact (normalised)
    goods/services text and rank by count.  Then report registration and
    5y-survival rates as a function of how many other filings share the
    exact same description.

Output:
  stdout: tables + summary stats
  paper/results/diagonal_and_templated.png: 4-panel figure
  paper/results/diagonal_and_templated.json: machine-readable summary
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC = REPO_ROOT / "data" / "processed"
RESULTS = REPO_ROOT / "paper" / "results"

# Cohort cap moved to filing_year ≤ 2018; passed_5y replaces the broken
# survived_5y column.  See diagonal_and_templated.py rev history / paper
# for the rationale (Section-8 timing).
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


# ---------------------------------------------------------------------------
# Part (1): diagonal vs off-diagonal
# ---------------------------------------------------------------------------
def part_one() -> dict:
    print("=" * 78)
    print("PART 1 — diagonal-clustering hypothesis")
    print("=" * 78, flush=True)

    parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit(): continue
        parts.append(pl.read_parquet(path).filter(
            CLEAN & pl.col("owner_name").is_not_null()
        ).with_columns(
            (pl.col("reached_registration") & pl.col("currently_live")).alias("passed_5y")
        ).select("owner_name", "kl_vs_past", "kl_vs_future",
                 "reached_registration", "passed_5y"))
    universe = pl.concat(parts)
    firm = universe.group_by("owner_name").agg(
        pl.col("kl_vs_past").mean().alias("mean_pros"),
        pl.col("kl_vs_future").mean().alias("mean_retr"),
        pl.col("reached_registration").mean().alias("mean_reach"),
        pl.col("passed_5y").mean().alias("mean_surv5"),
        pl.len().alias("n_filings"),
    ).filter(pl.col("n_filings") >= 3).to_pandas()
    print(f"  {len(firm):,} firms with >=3 clean filings", flush=True)

    firm["mean_KL"] = 0.5 * (firm["mean_pros"] + firm["mean_retr"])
    firm["abs_dKL"] = (firm["mean_pros"] - firm["mean_retr"]).abs()

    # Quintile-cross by mean_KL and abs_dKL.  Each cell is a strict slice;
    # firms inside have similar position-along-diagonal and perpendicular
    # distance from it.  If the survival penalty is concentrated in the
    # bottom-left of the topomap (low mean_KL, on-diagonal i.e. small
    # abs_dKL), the (Q1 mean_KL, Q1 abs_dKL) cell will be the worst.
    firm["mean_KL_q"]  = pd.qcut(firm["mean_KL"],  5, labels=[1, 2, 3, 4, 5]).astype(int)
    firm["abs_dKL_q"]  = pd.qcut(firm["abs_dKL"],  5, labels=[1, 2, 3, 4, 5]).astype(int)

    grid = firm.groupby(["mean_KL_q", "abs_dKL_q"]).agg(
        n=("mean_surv5", "size"),
        surv5=("mean_surv5", "mean"),
        reach=("mean_reach", "mean"),
        mean_KL=("mean_KL", "mean"),
        abs_dKL=("abs_dKL", "mean"),
    ).reset_index()
    print("\n5y survival by (mean_KL quintile rows, abs_dKL quintile cols)")
    pivot_surv = grid.pivot(index="mean_KL_q", columns="abs_dKL_q", values="surv5")
    pivot_n    = grid.pivot(index="mean_KL_q", columns="abs_dKL_q", values="n")
    print((pivot_surv * 100).round(1))
    print("\nfirm counts:")
    print(pivot_n)

    # The diagonal-only (abs_dKL Q1) and far-off-diagonal (abs_dKL Q5) slices
    diag = grid[grid["abs_dKL_q"] == 1].sort_values("mean_KL_q")
    offd = grid[grid["abs_dKL_q"] == 5].sort_values("mean_KL_q")
    print("\nOn-diagonal slice (abs_dKL Q1):")
    print(diag[["mean_KL_q", "n", "mean_KL", "abs_dKL", "surv5"]].to_string(index=False))
    print("Far-off-diagonal slice (abs_dKL Q5):")
    print(offd[["mean_KL_q", "n", "mean_KL", "abs_dKL", "surv5"]].to_string(index=False))

    # Same but ten linear bins on mean_KL across the 5-95 percentile range,
    # split into "near diagonal" (abs_dKL < median) vs "far off-diag" (>= median).
    lo, hi = firm["mean_KL"].quantile([0.05, 0.95])
    edges = np.linspace(lo, hi, 11)
    centres = 0.5 * (edges[:-1] + edges[1:])
    firm["mean_KL_bin"] = np.digitize(firm["mean_KL"], edges) - 1
    in_range = (firm["mean_KL_bin"] >= 0) & (firm["mean_KL_bin"] < 10)
    abs_dKL_med = firm["abs_dKL"].median()
    print(f"\nmedian abs_dKL = {abs_dKL_med:.3f}")

    diag_curve = []
    offd_curve = []
    for b in range(10):
        for tag, mask in (
            ("near", in_range & (firm["mean_KL_bin"] == b) & (firm["abs_dKL"] <  abs_dKL_med)),
            ("far",  in_range & (firm["mean_KL_bin"] == b) & (firm["abs_dKL"] >= abs_dKL_med)),
        ):
            sub = firm[mask]
            if len(sub) < 50:
                continue
            r = sub["mean_surv5"].mean()
            n = len(sub)
            lo_ci, hi_ci = wilson(r, n)
            (diag_curve if tag == "near" else offd_curve).append(
                (centres[b], r, lo_ci, hi_ci, n)
            )

    return {
        "firm_panel_size": int(len(firm)),
        "median_abs_dKL": float(abs_dKL_med),
        "grid_surv5_by_quintile": pivot_surv.values.tolist(),
        "grid_n_by_quintile":     pivot_n.values.tolist(),
        "diag_slice_surv5":       diag[["mean_KL_q","mean_KL","abs_dKL","n","surv5"]].to_dict(orient="records"),
        "offd_slice_surv5":       offd[["mean_KL_q","mean_KL","abs_dKL","n","surv5"]].to_dict(orient="records"),
        "near_diag_curve":        diag_curve,
        "far_off_diag_curve":     offd_curve,
        "mean_KL_centres":        centres.tolist(),
    }


# ---------------------------------------------------------------------------
# Part (2): templated G/S strings
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_NORM = re.compile(r"[^a-z0-9 ]")

def _norm(s: str) -> str:
    s = s.lower()
    s = _NORM.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def part_two() -> dict:
    print("\n" + "=" * 78)
    print("PART 2 — templated G/S strings and survival")
    print("=" * 78, flush=True)

    keep_cols = ["serial_number", "reached_registration", "survived_5y"]

    # Build the clean serial-number universe across all classes
    out_parts = []
    for path in sorted(PROC.glob("outcomes_class*.parquet")):
        cls = path.stem.replace("outcomes_class", "")
        if not cls.isdigit(): continue
        out_parts.append(pl.read_parquet(path).filter(CLEAN).select(keep_cols))
    out_all = pl.concat(out_parts).unique(subset=["serial_number"])
    print(f"  clean filings: {out_all.height:,}", flush=True)

    # Pull goods_services for these serials, normalise, group/count
    print("  loading & normalising goods_services across all 45 classes…", flush=True)
    gs_parts = []
    for path in sorted(PROC.glob("tm_class*.parquet")):
        cls = path.stem.replace("tm_class", "")
        if not cls.isdigit(): continue
        gs_parts.append(pl.read_parquet(path).select("serial_number", "goods_services"))
    gs_all = (pl.concat(gs_parts)
              .unique(subset=["serial_number"])
              .filter(pl.col("goods_services").is_not_null()))
    print(f"  goods_services rows: {gs_all.height:,}", flush=True)

    joined = (gs_all.join(out_all, on="serial_number", how="inner")
              .with_columns(
                  pl.col("goods_services").str.to_lowercase()
                  .str.replace_all(r"[^a-z0-9 ]", " ")
                  .str.replace_all(r"\s+", " ")
                  .str.strip_chars()
                  .alias("gs_norm")
              )
              .filter(pl.col("gs_norm").str.len_chars() >= 5))
    print(f"  joined and normalised: {joined.height:,}", flush=True)

    # Per-string aggregates
    by_str = joined.group_by("gs_norm").agg(
        pl.len().alias("count"),
        pl.col("reached_registration").mean().alias("reach_rate"),
        pl.col("survived_5y").mean().alias("surv5_rate"),
    )
    print(f"  {by_str.height:,} unique normalised G/S strings")

    # Top-50 most templated (printed)
    top = by_str.sort("count", descending=True).head(50).to_pandas()
    print("\nTop 50 most-templated G/S strings (count, registration%, 5y survival%):")
    for _, row in top.iterrows():
        snippet = row["gs_norm"][:120].replace("\n", " ")
        print(f"  n={int(row['count']):>7}  reg={row['reach_rate']*100:5.1f}%  surv5={row['surv5_rate']*100:5.1f}%  | {snippet}")

    # Filing-level: attach 'count of identical filings' (its template multiplicity)
    filing = joined.join(by_str.select("gs_norm", "count"), on="gs_norm", how="left").to_pandas()
    print(f"\n  {len(filing):,} filings tagged with template multiplicity")

    # Bin by log10(count). count==1 means the description is unique.
    bins = [0.5, 1.5, 2.5, 5.5, 10.5, 50.5, 200.5, 1000.5, 5000.5, 1e9]
    labels = ["1 (unique)", "2", "3-5", "6-10", "11-50", "51-200", "201-1000", "1001-5000", "5000+"]
    filing["count_bin"] = pd.cut(filing["count"], bins=bins, labels=labels, include_lowest=True)
    grp = filing.groupby("count_bin", observed=True).agg(
        n=("survived_5y", "size"),
        reach=("reached_registration", "mean"),
        surv5=("survived_5y", "mean"),
    ).reset_index()
    grp["reach_lo"], grp["reach_hi"] = zip(*[wilson(r, n) for r, n in zip(grp["reach"], grp["n"])])
    grp["surv5_lo"], grp["surv5_hi"] = zip(*[wilson(r, n) for r, n in zip(grp["surv5"], grp["n"])])
    print("\nFiling-level outcomes by template multiplicity:")
    print(grp[["count_bin", "n", "reach", "surv5"]].to_string(index=False))

    summary = {
        "n_clean_filings": int(out_all.height),
        "n_unique_norm_strings": int(by_str.height),
        "top50_templates": top.to_dict(orient="records"),
        "by_count_bin": grp.assign(count_bin=grp["count_bin"].astype(str)).to_dict(orient="records"),
    }

    return summary, grp, top


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_all(p1: dict, p2_grp: pd.DataFrame, p2_top: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) heatmap of the 5x5 quintile-cross
    ax = axes[0, 0]
    H = np.array(p1["grid_surv5_by_quintile"]) * 100
    im = ax.imshow(H, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(["Q1\nnear\ndiag.", "Q2", "Q3", "Q4", "Q5\nfar from\ndiag."])
    ax.set_yticks(range(5)); ax.set_yticklabels(["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"])
    ax.set_xlabel(r"$|p - r|$  quintile  (off-diagonal distance)")
    ax.set_ylabel(r"$(p + r)/2$  quintile  (along the diagonal)")
    ax.set_title("Mean 5y survival rate (%) by quintile of position\n"
                 "(rows = up the diagonal, cols = away from the diagonal)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{H[i,j]:.1f}", ha="center", va="center",
                    color="white" if H[i, j] < H.mean() else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) near-diagonal vs far-off-diagonal sweep along mean_KL
    ax = axes[0, 1]
    near = np.array(p1["near_diag_curve"])
    far = np.array(p1["far_off_diag_curve"])
    if near.size:
        ax.fill_between(near[:, 0], near[:, 2] * 100, near[:, 3] * 100,
                        color="#7a1111", alpha=0.25, linewidth=0)
        ax.plot(near[:, 0], near[:, 1] * 100, "o-", color="#7a1111", linewidth=1.5,
                label=r"near diagonal ($|p-r|<\mathrm{median}$)")
    if far.size:
        ax.fill_between(far[:, 0], far[:, 2] * 100, far[:, 3] * 100,
                        color="#1f3a7a", alpha=0.25, linewidth=0)
        ax.plot(far[:, 0], far[:, 1] * 100, "o-", color="#1f3a7a", linewidth=1.5,
                label=r"far off-diagonal ($|p-r|\geq\mathrm{median}$)")
    ax.set_xlabel(r"position along the diagonal $(p+r)/2$")
    ax.set_ylabel("Mean 5y survival rate (%)")
    ax.set_title("Survival as you move along vs. away from the diagonal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # (c) survival by template multiplicity (filing-level)
    ax = axes[1, 0]
    xs = np.arange(len(p2_grp))
    ax.bar(xs, p2_grp["surv5"] * 100, color="#1f77b4", alpha=0.85, label="5y survival")
    ax.bar(xs, p2_grp["reach"] * 100, color="#d62728", alpha=0.45, label="completed registration")
    for i, (s, r, n) in enumerate(zip(p2_grp["surv5"], p2_grp["reach"], p2_grp["n"])):
        ax.text(i, max(s, r) * 100 + 0.6, f"n={n:,}", ha="center", va="bottom", fontsize=8, rotation=20)
    ax.set_xticks(xs); ax.set_xticklabels(p2_grp["count_bin"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Outcome rate (%)")
    ax.set_xlabel("Filings sharing the exact (normalised) G/S text")
    ax.set_title("Templated descriptions: survival vs. multiplicity")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (d) printed list of top templates (smaller summary)
    ax = axes[1, 1]
    ax.axis("off")
    top10 = p2_top.head(15)
    text_lines = ["Top-15 most-templated normalised G/S strings",
                  "n      reg%  surv5%  | text (truncated)"]
    for _, row in top10.iterrows():
        snip = row["gs_norm"][:60].replace("\n", " ")
        text_lines.append(f"{int(row['count']):>6}  {row['reach_rate']*100:5.1f}  {row['surv5_rate']*100:5.1f}   | {snip}")
    ax.text(0.0, 1.0, "\n".join(text_lines), family="monospace", fontsize=8, va="top")

    fig.suptitle("Diagonal-clustering hypothesis & templated-description survival", fontsize=12, y=1.01)
    fig.tight_layout()
    out = RESULTS / "diagonal_and_templated.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] wrote {out}")


def main() -> int:
    # Part 2 (templated G/S) lives in scripts/templated_survival.py now.
    # This entrypoint runs only Part 1 (diagonal hypothesis) by default.
    p1 = part_one()
    out_json = RESULTS / "diagonal_part1.json"
    out_json.write_text(json.dumps({"part1": p1}, indent=2, default=str))
    print(f"[done] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
