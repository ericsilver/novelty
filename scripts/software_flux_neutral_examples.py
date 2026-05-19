"""Worked examples in Software (Class 9) of failed filings with
ΔKL ≈ 0, stratified by prospective KL.

Two flavours of flux-neutral failure to characterise:
  (a) ΔKL ≈ 0 AND high $KL^\\text{pros}$ — the filing is distinctive
      against the recent past AND against the near future (an isolated
      outlier; nobody adopts).
  (b) ΔKL ≈ 0 AND low $KL^\\text{pros}$  — pure boilerplate language
      neither novel nor receding (the 'me-too' middle).

For each stratum we pull 12 examples of FAILED filings (didn't survive
5y) with their owner, mark, and full goods/services text.

Outputs:
  paper/results/software_flux_neutral_examples.txt
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cls = "009"
    s = pl.read_parquet(
        PROC / f"surprise_class{cls}.parquet",
        columns=["serial_number", "year", "owner_name", "mark_identification",
                 "prospective_kl", "retrospective_kl",
                 "n_ref_prospective", "n_ref_retrospective", "n_terms"],
    ).filter(
        (pl.col("n_ref_prospective") >= 1000)
        & (pl.col("n_ref_retrospective") >= 1000)
        & (pl.col("n_terms").is_between(8, 50))   # normal-size filings
        & pl.col("prospective_kl").is_finite()
        & pl.col("retrospective_kl").is_finite()
        & pl.col("year").is_between(1995, 2019)
    ).with_columns(
        (pl.col("prospective_kl") - pl.col("retrospective_kl")).alias("dkl"),
    )
    o = pl.read_parquet(PROC / f"outcomes_class{cls}.parquet",
                         columns=["serial_number", "survived_5y", "reached_registration"])
    t = pl.read_parquet(PROC / f"tm_class{cls}.parquet",
                         columns=["serial_number", "goods_services"])
    j = s.join(o, on="serial_number", how="inner").join(t, on="serial_number", how="inner")
    print(f"[panel] Class 9 clean, 1995-2019, n_terms>=8: n={j.height:,}", flush=True)

    pros = j["prospective_kl"].to_numpy()
    import numpy as np
    p10 = float(np.quantile(pros, 0.10))
    p90 = float(np.quantile(pros, 0.90))
    print(f"        pros KL p10 = {p10:.2f}  p90 = {p90:.2f}", flush=True)

    # FLUX-NEUTRAL = |dkl| <= 0.05 (5pp window around zero)
    flux_neutral = j.filter(pl.col("dkl").abs() <= 0.05)
    print(f"        flux-neutral (|ΔKL|<=0.05): n={flux_neutral.height:,}", flush=True)
    failed = flux_neutral.filter(~pl.col("survived_5y"))
    print(f"        flux-neutral failed: n={failed.height:,}", flush=True)

    # (a) high-pros stratum (top 10%)
    high = failed.filter(pl.col("prospective_kl") >= p90).sort("prospective_kl", descending=True).head(12)
    # (b) low-pros stratum (bottom 10%)
    low  = failed.filter(pl.col("prospective_kl") <= p10).sort("prospective_kl").head(12)

    def render(rows: pl.DataFrame, label: str) -> list[str]:
        out = [f"\n=== {label} ===\n"]
        for r in rows.iter_rows(named=True):
            gs = (r["goods_services"] or "").replace("\n", " ").strip()
            out.append(
                f"  year {r['year']}  pros={r['prospective_kl']:.2f}  "
                f"retr={r['retrospective_kl']:.2f}  ΔKL={r['dkl']:+.3f}  "
                f"n_terms={r['n_terms']}"
            )
            out.append(f"    owner: {(r['owner_name'] or '')[:80]}")
            out.append(f"    mark:  {(r['mark_identification'] or '')[:80]}")
            out.append(f"    g/s:   {gs[:400]}")
            out.append("")
        return out

    lines = [
        "Software (Class 9) flux-neutral failures (|ΔKL| <= 0.05), 1995-2019, n_terms >= 8.",
        "These are the failures in the squeeze: filings whose language",
        "is neither moving toward nor away from where the field is going.\n",
        f"Class-9 flux-neutral cohort: {flux_neutral.height:,} clean filings; "
        f"{failed.height:,} failed (survived_5y = False).",
        f"Prospective-KL p10 = {p10:.2f} nats, p90 = {p90:.2f} nats.\n",
    ]
    lines += render(high, "FLAVOUR (a): ΔKL≈0 AND high prospective KL — isolated outliers "
                          "(distinctive against past AND future; no echo)")
    lines += render(low,  "FLAVOUR (b): ΔKL≈0 AND low prospective KL — pure boilerplate "
                          "(mainstream language, neither leading nor trailing)")

    text = "\n".join(lines)
    (OUT / "software_flux_neutral_examples.txt").write_text(text)
    print(text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
