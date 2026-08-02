"""Pull representative filings with very negative ΔKL — firms whose
goods/services vocabulary is closer to the past than to the future, i.e.
exploiting a vanishing trend.

For each era (pre-2000, 2000-2009, 2010-2021), report:
  - the 20 most-negative-ΔKL filings with substantive g/s text
  - the tokens that distinguish negative-ΔKL filings (over-represented
    vs. corpus baseline)

Filters:
  - CLEAN H=2y cut (n_ref >=1000, n_terms>=3, finite KLs)
  - n_terms >= 10 for the example filings (avoid single-word boilerplate)
  - kl_vs_past in (3.5, 7) so we're not just pulling extreme-distinctive outliers

Outputs:
  paper/results/vanishing_trend_examples.txt
"""

from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT  = REPO / "paper" / "results"


def class_list() -> list[str]:
    return sorted(
        p.stem.replace("tm_class", "")
        for p in PROC.glob("tm_class*.parquet")
        if p.stem.replace("tm_class", "").isdigit()
    )


def collect() -> pl.DataFrame:
    parts = []
    for cls in class_list():
        sp = PROC / f"surprise_class{cls}.parquet"
        tm = PROC / f"tm_class{cls}.parquet"
        if not (sp.exists() and tm.exists()): continue
        s = pl.read_parquet(
            sp,
            columns=["serial_number","year","kl_vs_past","kl_vs_future",
                     "n_ref_past","n_ref_future","n_terms",
                     "owner_name","mark_identification"],
        ).filter(
            (pl.col("n_ref_past") >= 1000)
            & (pl.col("n_ref_future") >= 1000)
            & (pl.col("n_terms") >= 10)
            & pl.col("kl_vs_past").is_finite()
            & pl.col("kl_vs_future").is_finite()
            & pl.col("kl_vs_past").is_between(3.5, 7.0)
            & pl.col("year").is_between(1985, 2021)
        ).with_columns(
            (pl.col("kl_vs_past") - pl.col("kl_vs_future")).alias("dkl"),
        )
        t = pl.read_parquet(tm, columns=["serial_number","goods_services"])
        j = s.join(t, on="serial_number", how="inner").with_columns(
            pl.lit(cls).alias("cls"))
        if j.height:
            parts.append(j)
        del s, t, j
        gc.collect()
    return pl.concat(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[load] streaming all classes …", flush=True)
    df = collect()
    print(f"       n={df.height:,}  ΔKL min={df['dkl'].min():.3f}  median={df['dkl'].median():.3f}", flush=True)

    eras = [("pre-2000", 1985, 1999),
            ("2000s",    2000, 2009),
            ("2010-21",  2010, 2021)]
    lines = ["Most-negative-ΔKL filings (n_terms>=10, pros in [3.5,7]).\n"
             "These are filings whose goods/services vocabulary is more typical of\n"
             "the past than of the future — firms exploiting a vanishing trend.\n"]
    for era, y0, y1 in eras:
        sub = df.filter(pl.col("year").is_between(y0, y1))
        if sub.height == 0: continue
        top = sub.sort("dkl").head(20)
        lines.append(f"\n=== {era}  (n={sub.height:,};  bottom-1% ΔKL cutoff = {sub['dkl'].quantile(0.01):.3f}) ===\n")
        for r in top.iter_rows(named=True):
            gs = (r["goods_services"] or "").replace("\n"," ")[:200]
            lines.append(
                f"  ΔKL={r['dkl']:+.2f}  pros={r['kl_vs_past']:.2f} retr={r['kl_vs_future']:.2f}  "
                f"{r['year']}  cls{r['cls']}\n"
                f"    owner: {(r['owner_name'] or '')[:70]}\n"
                f"    mark:  {(r['mark_identification'] or '')[:60]}\n"
                f"    g/s:   {gs}\n"
            )

    # Token-level: what unigrams are over-represented in bottom-decile ΔKL?
    print("[tokens] computing over-representation in bottom-decile ΔKL …", flush=True)
    threshold = df["dkl"].quantile(0.10)
    is_bottom = (df["dkl"] <= threshold).to_numpy()
    docs = df["goods_services"].to_list()
    bot_tok = Counter(); all_tok = Counter()
    for i, d in enumerate(docs):
        if d is None: continue
        toks = set(t for t in d.lower().split() if t.isalpha() and len(t) > 2)
        for t in toks:
            all_tok[t] += 1
            if is_bottom[i]:
                bot_tok[t] += 1
    n_bottom = int(is_bottom.sum())
    n_all = len(docs)
    pop_rate = {t: c/n_all for t, c in all_tok.items() if c >= 1000}
    bot_rate = {t: bot_tok.get(t,0)/n_bottom for t in pop_rate}
    enrich = sorted(((t, bot_rate[t]/pop_rate[t], bot_rate[t], pop_rate[t]) for t in pop_rate),
                    key=lambda x: -x[1])
    lines.append("\n\n=== Tokens over-represented in bottom-decile-ΔKL filings ===")
    lines.append(f"   (rate in bottom decile  /  rate in corpus; tokens with corpus freq >= 1000)")
    lines.append(f"   {'token':<28s}  {'enrich':>6s}  {'p_bot':>6s}  {'p_pop':>6s}")
    for t, e, pb, pp in enrich[:40]:
        lines.append(f"   {t:<28s}  {e:>6.2f}x  {pb:>6.3f}  {pp:>6.3f}")

    txt = "\n".join(lines)
    (OUT / "vanishing_trend_examples.txt").write_text(txt)
    print(f"\n[done] wrote {OUT/'vanishing_trend_examples.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
