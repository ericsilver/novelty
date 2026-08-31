"""Curated-vocabulary rungs on the rebuilt Form D match.

Recomputes the internet / AI / blockchain debut ladder rates the paper
quotes (funded, reporting, IPO per curated group, against all debuts) on
the 2009-2018 registered-debut frame and the current
funding_owner_match.parquet, and persists them.

Output: paper/results/curated_ladder.json
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"
CLASSES = [f"{i:03d}" for i in range(1, 46)]

PATTERNS = {
    "internet": r"\binternet\b|\bonline\b|\bon-line\b|\bweb ?sites?\b|\bweb pages?\b|\bwebsites?\b|\bworld wide web\b|\be-?commerce\b|\belectronic commerce\b|\bweb portals?\b",
    "ai": r"artificial intelligence|machine learning|deep learning|neural network|natural language processing|natural language understanding|computer vision|predictive analytics|chatbots?|generative ai|large language model",
    "blockchain": r"blockchain|cryptocurrenc|crypto asset|crypto token|\bbitcoin\b|non-fungible token|distributed ledger|digital currency|virtual currency|smart contract",
}


def log(m): print(m, file=sys.stderr, flush=True)


def main() -> int:
    fm = pl.read_parquet(PROC / "funding_owner_match.parquet",
                         columns=["owner_name", "first_formd_date", "in_sec", "in_fsds",
                                  "in_8a", "ipo_date"])
    parts = []
    for c in CLASSES:
        tp = PROC / f"tm_class{c}.parquet"
        sp = PROC / f"rolling_surprise_class{c}.parquet"
        if not (tp.exists() and sp.exists()):
            continue
        tm = pl.read_parquet(tp, columns=["serial_number", "owner_name", "filing_date",
                                          "registration_date", "goods_services"]).filter(
            pl.col("owner_name").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
            & (pl.col("registration_date").fill_null("").str.len_chars() >= 8))
        sc = pl.read_parquet(sp, columns=["serial_number", "topic_dkl"]).filter(
            pl.col("topic_dkl").is_finite())
        tm = tm.join(sc, on="serial_number", how="inner").with_columns(
            pl.col("goods_services").fill_null("").str.to_lowercase().alias("gs"))
        for name, pat in PATTERNS.items():
            tm = tm.with_columns(pl.col("gs").str.contains(pat).alias(name))
        parts.append(tm.select("serial_number", "owner_name", "filing_date",
                               "internet", "ai", "blockchain"))
        del tm, sc
        gc.collect()
    d = pl.concat(parts); del parts; gc.collect()
    debut = d.group_by("owner_name").agg(pl.col("filing_date").min().alias("dd"))
    first = d.join(debut, on="owner_name", how="inner").filter(
        pl.col("filing_date") == pl.col("dd")).with_columns(
        pl.col("filing_date").str.slice(0, 4).cast(pl.Int32).alias("fy"))
    # collapse multi-class debuts: owner counts once, flags OR'd
    first = first.group_by("owner_name").agg(
        pl.col("fy").min(), pl.col("internet").any(), pl.col("ai").any(),
        pl.col("blockchain").any(), pl.col("filing_date").min().alias("fd"))
    first = first.filter(pl.col("fy").is_between(2009, 2018))
    x = first.join(fm, on="owner_name", how="left").with_columns(
        (pl.col("first_formd_date").is_not_null()
         & (pl.col("first_formd_date") >= pl.col("fd").str.strptime(pl.Date, "%Y%m%d", strict=False))).alias("funded"),
        (pl.col("in_sec").fill_null(0) + pl.col("in_fsds").fill_null(0) >= 1).alias("reporting"),
        ((pl.col("in_8a").fill_null(0) == 1) | pl.col("ipo_date").is_not_null()).alias("ipo"))
    out = {"n_debut_owners": x.height}
    groups = {"all": pl.lit(True), "internet": pl.col("internet"),
              "ai": pl.col("ai"), "blockchain": pl.col("blockchain")}
    for name, cond in groups.items():
        g = x.filter(cond)
        out[name] = {"n": g.height,
                     "funded": float(g["funded"].mean()),
                     "reporting": float(g["reporting"].mean()),
                     "ipo": float(g["ipo"].mean())}
        log(f"[{name}] n={g.height:,} funded={100*out[name]['funded']:.2f}% "
            f"reporting={100*out[name]['reporting']:.2f}% ipo={100*out[name]['ipo']:.2f}%")
    (RES / "curated_ladder.json").write_text(json.dumps(out, indent=1))
    log("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
