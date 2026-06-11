"""Reference-window decomposition, all classes.

Variants per filing (requires W3, W5, W7 surprise files per class):
  sym3, sym5, sym7, asym37 = pros(W3)-retr(W7), asym73 = pros(W7)-retr(W3)

Outcomes:
  - first Section 8 gate survival (registrations 2016-2018,
    PASS=701/702/703/704/705/800, FAIL=710): pooled quintile rates per
    variant + per-class q5-q1 lifts for sym5/asym37/asym73
  - registration (filings 1995-2018): pooled inverse-U depth per variant

Outputs: paper/results/window_choice_all.{json,png}
"""
from __future__ import annotations

import gc
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
VARIANTS = ["sym3", "sym5", "sym7", "asym37", "asym73"]


def load_class(cls: str) -> pl.DataFrame | None:
    paths = {t: PROC / f"surprise_class{cls}{s}.parquet"
             for t, s in (("5", ""), ("3", "_W3"), ("7", "_W7"))}
    if not all(p.exists() for p in paths.values()):
        return None

    def one(tag: str) -> pl.DataFrame:
        cols = ["serial_number", "prospective_kl", "retrospective_kl",
                "n_ref_prospective", "n_ref_retrospective"]
        if tag == "5":
            cols += ["year", "n_terms"]
        return pl.read_parquet(paths[tag], columns=cols).filter(
            pl.col("prospective_kl").is_finite()
            & pl.col("retrospective_kl").is_finite()
            & (pl.col("n_ref_prospective") >= 600)
            & (pl.col("n_ref_retrospective") >= 600)
        ).rename({"prospective_kl": f"pros{tag}",
                  "retrospective_kl": f"retr{tag}"}).drop(
            "n_ref_prospective", "n_ref_retrospective")

    j = one("5").join(one("3"), on="serial_number", how="inner").join(
        one("7"), on="serial_number", how="inner").filter(pl.col("n_terms") >= 3)
    tm = pl.read_parquet(
        PROC / f"tm_class{cls}.parquet",
        columns=["serial_number", "registration_date", "status_code"],
    ).with_columns(
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8
         ).alias("registered"),
        pl.col("registration_date").str.slice(0, 4)
          .cast(pl.Int32, strict=False).alias("reg_year"),
        pl.col("status_code").is_in(list(PASS)).alias("st_pass"),
        pl.col("status_code").is_in(list(FAIL)).alias("st_fail"),
    )
    return j.join(tm, on="serial_number", how="inner").with_columns([
        (pl.col("pros3") - pl.col("retr3")).alias("sym3"),
        (pl.col("pros5") - pl.col("retr5")).alias("sym5"),
        (pl.col("pros7") - pl.col("retr7")).alias("sym7"),
        (pl.col("pros3") - pl.col("retr7")).alias("asym37"),
        (pl.col("pros7") - pl.col("retr3")).alias("asym73"),
    ]).select("serial_number", "year", "registered", "reg_year",
              "st_pass", "st_fail", *VARIANTS)


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


def main() -> int:
    classes = sorted(p.stem.replace("tm_class", "")
                     for p in PROC.glob("tm_class*.parquet")
                     if p.stem.replace("tm_class", "").isdigit())
    gate_parts, reg_parts = [], []
    per_class = []
    for cls in classes:
        d = load_class(cls)
        if d is None or d.height == 0:
            continue
        gate = d.filter(pl.col("reg_year").is_between(2016, 2018)
                        & (pl.col("st_pass") | pl.col("st_fail")))
        regp = d.filter(pl.col("year").is_between(1995, 2018))
        if gate.height >= 2000:
            row = {"cls": cls, "n_gated": gate.height}
            for v in ("sym5", "asym37", "asym73"):
                row[f"gate_lift_{v}"] = quintile_rates(gate, v, "st_pass")["lift_q5_q1"]
            per_class.append(row)
        gate_parts.append(gate.select("st_pass", *VARIANTS))
        reg_parts.append(regp.select("registered", *VARIANTS))
        print(f"  {cls}: gate {gate.height:,}, reg {regp.height:,}",
              file=sys.stderr, flush=True)
        del d, gate, regp
        gc.collect()

    gate_all = pl.concat(gate_parts); del gate_parts; gc.collect()
    reg_all = pl.concat(reg_parts); del reg_parts; gc.collect()
    out = {"n_gate_pooled": gate_all.height, "n_reg_pooled": reg_all.height,
           "pooled": {}, "per_class": per_class}
    for v in VARIANTS:
        out["pooled"][v] = {
            "gate": quintile_rates(gate_all, v, "st_pass"),
            "registration": quintile_rates(reg_all, v, "registered"),
        }
        g = out["pooled"][v]
        print(f"  {v:>7}: gate lift {g['gate']['lift_q5_q1']:+.4f}  "
              f"reg invU {g['registration']['inverseU_depth']:+.4f}",
              file=sys.stderr, flush=True)

    neg37 = sum(1 for r in per_class if r["gate_lift_asym37"] < 0)
    neg73 = sum(1 for r in per_class if r["gate_lift_asym73"] < 0)
    neg5 = sum(1 for r in per_class if r["gate_lift_sym5"] < 0)
    out["per_class_negative_counts"] = {
        "sym5": neg5, "asym37": neg37, "asym73": neg73, "total": len(per_class)}
    print(f"per-class negative gate lifts: sym5 {neg5}/{len(per_class)}, "
          f"asym37 {neg37}/{len(per_class)}, asym73 {neg73}/{len(per_class)}",
          file=sys.stderr, flush=True)

    (RES / "window_choice_all.json").write_text(json.dumps(out, indent=1, default=float))

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, 6)
    for v in VARIANTS:
        g = out["pooled"][v]["gate"]
        ax.plot(xs, [g[f"q{i}"] for i in xs], "o-",
                label=f"{v} (lift {g['lift_q5_q1']*100:+.1f}pp)")
    ax.set_xticks(xs)
    ax.set_xlabel("variant quintile")
    ax.set_ylabel("P(Section 8 accepted | faced first gate)")
    ax.set_title("First-gate survival by reference-window variant, all classes\n"
                 "registrations 2016-2018")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "window_choice_all.png", dpi=140, bbox_inches="tight")
    print("[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
