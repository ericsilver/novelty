"""'Novel relative to what?' -- reference-window choice and outcome
prediction, class 009.

Variants of dKL built from the W=3, W=5, W=7 surprise files (and
asymmetric mixes across files):

  sym3    pros(W3) - retr(W3)
  sym5    pros(W5) - retr(W5)   (paper baseline)
  sym7    pros(W7) - retr(W7)
  asym37  pros(W3) - retr(W7)   (short memory, long foresight)
  asym73  pros(W7) - retr(W3)   (long memory, short foresight)

For each variant: (a) registration outcome on filings 1995-2018 (dKL
quintile rates), (b) first-Section-8-gate survival on the 2016-2018
registration cohort (quintile rates + per-sigma logit-style standardized
gap). Clean filter as elsewhere; n_ref floors use each file's own
references.

Outputs: paper/results/window_choice_009.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

PASS = {"701", "702", "703", "704", "705", "800"}
FAIL = {"710"}


def load_windows() -> pl.DataFrame:
    def one(suffix: str, tag: str) -> pl.DataFrame:
        return pl.read_parquet(
            PROC / f"surprise_class009{suffix}.parquet",
            columns=["serial_number", "year", "n_terms", "prospective_kl",
                     "retrospective_kl", "n_ref_prospective", "n_ref_retrospective"],
        ).filter(
            pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
            & (pl.col("n_ref_prospective") >= 600)
            & (pl.col("n_ref_retrospective") >= 600)
        ).rename({
            "prospective_kl": f"pros{tag}", "retrospective_kl": f"retr{tag}",
        }).select("serial_number",
                  *( ["year", "n_terms"] if tag == "5" else [] ),
                  f"pros{tag}", f"retr{tag}")
    base = one("", "5")
    w3 = one("_W3", "3")
    w7 = one("_W7", "7")
    j = base.join(w3, on="serial_number", how="inner").join(
        w7, on="serial_number", how="inner")
    return j.with_columns([
        (pl.col("pros3") - pl.col("retr3")).alias("sym3"),
        (pl.col("pros5") - pl.col("retr5")).alias("sym5"),
        (pl.col("pros7") - pl.col("retr7")).alias("sym7"),
        (pl.col("pros3") - pl.col("retr7")).alias("asym37"),
        (pl.col("pros7") - pl.col("retr3")).alias("asym73"),
    ]).filter(pl.col("n_terms") >= 3)


def quintile_rates(df: pl.DataFrame, var: str, outcome: str) -> dict:
    arr = df[var].to_numpy()
    cuts = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(cuts, arr)
    g = df.with_columns(pl.Series("q", qi)).group_by("q").agg(
        pl.len().alias("n"),
        pl.col(outcome).cast(pl.Float64).mean().alias("rate")).sort("q")
    d = {f"q{int(r['q'])+1}": r["rate"] for r in g.iter_rows(named=True)}
    d["lift_q5_q1"] = d["q5"] - d["q1"]
    d["inverseU_depth"] = d["q3"] - 0.5 * (d["q1"] + d["q5"])
    return d


def std_gap(df: pl.DataFrame, var: str, outcome: str) -> float:
    """Mean of standardized var among successes minus failures
    (a scale-free separation measure, comparable across variants)."""
    x = df[var].to_numpy()
    y = df[outcome].cast(pl.Float64).to_numpy()
    xz = (x - x.mean()) / x.std()
    return float(xz[y == 1].mean() - xz[y == 0].mean())


def main() -> int:
    j = load_windows()
    print(f"[load] {j.height:,} filings with all three windows",
          file=sys.stderr, flush=True)

    tm = pl.read_parquet(
        PROC / "tm_class009.parquet",
        columns=["serial_number", "registration_date", "status_code"],
    ).with_columns(
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8
         ).alias("registered"),
        pl.col("registration_date").str.slice(0, 4)
          .cast(pl.Int32, strict=False).alias("reg_year"),
        pl.col("status_code").is_in(list(PASS)).alias("st_pass"),
        pl.col("status_code").is_in(list(FAIL)).alias("st_fail"),
    )
    d = j.join(tm, on="serial_number", how="inner")

    reg_panel = d.filter(pl.col("year").is_between(1995, 2018))
    gate_panel = d.filter(pl.col("reg_year").is_between(2016, 2018)
                          & (pl.col("st_pass") | pl.col("st_fail")))
    print(f"[panel] registration n={reg_panel.height:,}; "
          f"gate n={gate_panel.height:,}", file=sys.stderr, flush=True)

    variants = ["sym3", "sym5", "sym7", "asym37", "asym73"]
    out = {"n_reg_panel": reg_panel.height, "n_gate_panel": gate_panel.height,
           "variants": {}}
    for v in variants:
        out["variants"][v] = {
            "registration": quintile_rates(reg_panel, v, "registered"),
            "gate": quintile_rates(gate_panel, v, "st_pass"),
            "gate_std_gap": std_gap(gate_panel, v, "st_pass"),
        }
        g = out["variants"][v]
        print(f"  {v:>7}: gate lift {g['gate']['lift_q5_q1']:+.4f}  "
              f"std-gap {g['gate_std_gap']:+.4f}  "
              f"reg invU {g['registration']['inverseU_depth']:+.4f}",
              file=sys.stderr, flush=True)

    # correlations between variants (per filing)
    corr = {}
    pd_df = j.select(variants).to_pandas()
    for a in variants:
        for b in variants:
            if a < b:
                corr[f"{a}~{b}"] = float(pd_df[a].corr(pd_df[b]))
    out["variant_correlations"] = corr

    (RES / "window_choice_009.json").write_text(json.dumps(out, indent=1, default=float))

    # figure: gate survival quintile curves per variant
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, 6)
    for v in variants:
        g = out["variants"][v]["gate"]
        ax.plot(xs, [g[f"q{i}"] for i in xs], "o-",
                label=f"{v} (lift {g['lift_q5_q1']*100:+.1f}pp)")
    ax.set_xticks(xs)
    ax.set_xlabel("dKL-variant quintile")
    ax.set_ylabel("P(Section 8 accepted | faced first gate)")
    ax.set_title("First-gate survival by reference-window variant, class 009\n"
                 "registrations 2016-2018")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "window_choice_009.png", dpi=140, bbox_inches="tight")
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
