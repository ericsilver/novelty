"""Per-industry appendix generator.

For each NICE class with sufficient clean-panel data, emit one short
appendix section into paper/results/per_industry_appendix.tex containing:
  - title (NICE number + abbreviated industry label)
  - filing count, distinct owners, year range
  - flux stats: mean prospective, retrospective, ΔKL on the clean panel
  - 5y survival rate and SEC-listed share
  - top 3 firms by firm-mean ΔKL (≥10 filings)
  - 3 worked examples: highest +ΔKL filing, near-zero filing, lowest -ΔKL filing

These appendix sections expand on the brief industry table in the main
paper (\\input{results/industry_summary.tex}) by giving deep treatment
to each NICE class, satisfying the goal that "appendices allow readers
to deeply explore different domains."

Outputs:
  paper/results/per_industry_appendix.tex   (one file, many \\subsection*)
"""

from __future__ import annotations

import gc
from pathlib import Path

import polars as pl

from novelty.industries import name as industry_name

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def texescape(s: str | None) -> str:
    if s is None:
        return ""
    s = s.replace("\\", r"\textbackslash{}")
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return s


def render_one(cls: str, sec_owners: set[str]) -> str | None:
    sp = PROC / f"surprise_class{cls}.parquet"
    tm = PROC / f"tm_class{cls}.parquet"
    op = PROC / f"outcomes_class{cls}.parquet"
    if not (sp.exists() and tm.exists() and op.exists()):
        return None

    s = pl.read_parquet(
        sp,
        columns=["serial_number", "year", "owner_name",
                 "kl_vs_past", "kl_vs_future",
                 "n_ref_past", "n_ref_future", "n_terms"],
    ).filter(
        (pl.col("n_ref_past") >= 1000)
        & (pl.col("n_ref_future") >= 1000)
        & (pl.col("n_terms") >= 3)
        & pl.col("kl_vs_past").is_finite()
        & pl.col("kl_vs_future").is_finite()
        & pl.col("year").is_between(1990, 2021)
    ).with_columns(
        (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
    )
    if s.height < 5_000:
        return None
    o = pl.read_parquet(
        op, columns=["serial_number", "survived_5y", "reached_registration"]
    )
    t = pl.read_parquet(tm, columns=["serial_number", "mark_identification", "goods_services"])
    j = s.join(o, on="serial_number", how="inner").join(t, on="serial_number", how="inner")
    if j.height == 0:
        return None

    n_filings  = j.height
    n_owners   = j["owner_name"].n_unique()
    y_lo, y_hi = j["year"].min(), j["year"].max()
    mean_pros  = float(j["kl_vs_past"].mean())
    mean_retr  = float(j["kl_vs_future"].mean())
    mean_dkl   = float(j["dkl"].mean())
    med_dkl    = float(j["dkl"].median())
    surv_rate  = float(j["survived_5y"].cast(pl.Float64).mean())
    sec_share  = float(j["owner_name"].is_in(sec_owners).cast(pl.Float64).mean())

    # Top 3 firms by firm-mean ΔKL with at least 10 filings
    firm = (j.group_by("owner_name")
              .agg([pl.len().alias("nf"), pl.col("dkl").mean().alias("mdkl")])
              .filter(pl.col("nf") >= 10)
              .sort("mdkl", descending=True)
              .head(3))
    top_firms = "; ".join(
        f"{texescape((r['owner_name'] or '')[:50])} (${r['mdkl']:+.2f}$)"
        for r in firm.iter_rows(named=True)
    )

    # Three worked examples: high +ΔKL, near-zero, low -ΔKL
    def example(df: pl.DataFrame, n: int = 1) -> list[dict]:
        return df.filter(pl.col("n_terms").is_between(8, 30)).head(n).to_dicts()

    pos = example(j.sort("dkl", descending=True))
    neg = example(j.sort("dkl"))
    near_zero = example(j.with_columns(pl.col("dkl").abs().alias("absdkl"))
                          .sort("absdkl"))

    def fmt(row: dict | None, lbl: str) -> str:
        if not row:
            return ""
        mark = texescape((row.get("mark_identification") or "")[:40])
        gs = texescape((row.get("goods_services") or "").replace("\n", " ")[:200])
        return (f"  \\textbf{{{lbl}}} \\textsc{{{mark}}} ({row['year']}, "
                f"$\\Delta KL={row['dkl']:+.2f}$):"
                f"\\\\\n  \\textit{{{gs}}}\\\\")

    name = texescape(industry_name(cls))
    lines = [
        f"\\subsection*{{Class {cls}: {name}}}",
        f"\\label{{app:industry-{cls}}}",
        "",
        f"\\begin{{tabular}}{{ll}}",
        f"Clean filings: & {n_filings:,} \\\\",
        f"Unique owners: & {n_owners:,} \\\\",
        f"Year range: & {y_lo}--{y_hi} \\\\",
        f"Mean $KL_\\text{{pros}}$: & {mean_pros:.2f} nats \\\\",
        f"Mean $KL_\\text{{retr}}$: & {mean_retr:.2f} nats \\\\",
        f"Mean / median $\\Delta KL$: & {mean_dkl:+.3f} / {med_dkl:+.3f} \\\\",
        f"5y survival rate: & {surv_rate*100:.1f}\\% \\\\",
        f"Share with SEC-listed owner: & {sec_share*100:.2f}\\% \\\\",
        f"\\end{{tabular}}",
        "",
        f"\\textbf{{Top 3 firms by firm-mean $\\Delta KL$ ($\\geq 10$ filings):}} "
        f"{top_firms if top_firms else 'none'}",
        "",
        f"\\textbf{{Worked examples}}:",
        "",
        fmt(pos[0] if pos else None, "Positive flux:"),
        fmt(near_zero[0] if near_zero else None, "Flux-neutral:"),
        fmt(neg[0] if neg else None, "Negative flux:"),
        "",
    ]
    del s, o, t, j
    gc.collect()
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sec_xw = pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")
    sec_owners = set(sec_xw["owner_name"].to_list())

    sections = []
    for cls in class_list():
        print(f"[render] class {cls}…", flush=True)
        sec = render_one(cls, sec_owners)
        if sec:
            sections.append(sec)

    body = "\n".join(sections)
    (OUT / "per_industry_appendix.tex").write_text(body)
    print(f"[done] wrote {OUT/'per_industry_appendix.tex'}  ({len(sections)} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
