"""Build keyword cohorts -- Internet, AI, blockchain -- and score their lifecycles.

Theme bundles were the first attempt at this and they only half work. Running
scripts/theme_bundles.py shows why: at T=500 there are real themes for the
internet, for mobile hardware, for biotechnology and for clean energy, but none
at all for artificial intelligence, blockchain or cloud software. The apparent
AI theme is additive manufacturing, matched on the word "generative"; the
apparent blockchain theme is the generic downloadable-software theme, which
holds 97% of blockchain vocabulary while devoting under 2% of itself to it.

That gap is a property of how the model was fitted, not of the corpus. The LDA
draws at most 10,000 filings per Nice class, so class 009 enters at 0.63% and
class 023 at 100% -- a 158-fold spread in sampling rate, running inversely with
class size. Themes are therefore allocated per class rather than per filing, and
the vocabulary that concentrates in the largest classes is thinned before the
model ever sees it: blockchain appears in 0.62% of the corpus but 0.23% of the
fit sample, AI in 0.95% against 0.39%, cloud in 2.52% against 0.39%. Solar, the
one concept here that spreads across many small classes, enters the sample at
its corpus rate -- and it is the one with clean themes.

A cohort avoids the problem by not going through the model. Membership is a
match against the filing's own goods-and-services text over the whole corpus,
with no sampling, so a concept present in a hundred thousand filings is found in
a hundred thousand filings. What is lost is the model's tolerance for paraphrase:
a cohort finds filings that use the word and misses those that describe the same
thing differently. Cohorts and bundles are therefore reported as what they are,
two different objects, rather than merged.

Patterns are written to avoid the traps the bundle work turned up: "block chain"
matches chains of pulley blocks, "token" matches coin-operated machines, "wallet"
matches leather goods, "generative" matches generative manufacturing, and
"carbon" matches carbon fiber, carbon steel and carbonated drinks.

Usage:  python scripts/theme_cohorts.py
Output: paper/results/theme_cohorts.json
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
GATE_LO, GATE_HI = 4.0, 8.5

COHORTS = {
    "internet": {
        "label": "Internet and the web",
        "note": "Online commerce and communication.",
        "pat": r"\binternet\b|\bonline\b|\bon-line\b|\bweb ?sites?\b|\bweb pages?\b|"
               r"\bwebsites?\b|\bworld wide web\b|\be-?commerce\b|"
               r"\belectronic commerce\b|\bweb portals?\b|\bweb browsers?\b",
    },
    "ai": {
        "label": "Artificial intelligence",
        "note": "Explicit AI vocabulary. Excludes bare 'generative', which in "
                "this corpus is usually generative manufacturing.",
        "pat": r"\bartificial intelligence\b|\bmachine learning\b|"
               r"\bdeep learning\b|\bneural networks?\b|\bnatural language "
               r"(processing|understanding)\b|\bcomputer vision\b|"
               r"\bpredictive analytics\b|\bchat ?bots?\b|"
               r"\bgenerative (ai|artificial)\b|\blarge language model",
    },
    "blockchain": {
        "label": "Blockchain and cryptocurrency",
        "note": "Distributed-ledger vocabulary. 'block chain' is required to "
                "carry a crypto context, since it also names chains of blocks.",
        "pat": r"\bblockchains?\b|\bcrypto ?currenc|\bcrypto ?assets?\b|"
               r"\bcrypto ?tokens?\b|\bbitcoin\b|\bnon-?fungible token|\bnfts?\b|"
               r"\bdistributed ledger|\bdigital currenc|\bvirtual currenc|"
               r"\bsmart contracts?\b",
    },
    "mobile": {
        "label": "Mobile and wireless",
        "note": "Handsets and the applications written for them.",
        "pat": r"\bmobile (app|application|phone|device|telephone|computing)|"
               r"\bsmart ?phones?\b|\bcellular (phone|telephone|communication)|"
               r"\bhand-?held (computer|device|electronic)|"
               r"\bwireless (communication|device|network|internet|telephone)|"
               r"\btablet computers?\b",
    },
    "cloud": {
        "label": "Cloud and hosted software",
        "note": "Software delivered as a service rather than shipped.",
        "pat": r"\bcloud comput|\bcloud-based\b|\bsoftware as a service\b|"
               r"\bsaas\b|\bplatform as a service\b|\bhosting services?\b|"
               r"\bweb hosting\b|\bdata cent(er|re)s?\b|\bvirtualization\b",
    },
    "biotech": {
        "label": "Biotechnology",
        "note": "Molecular and genomic vocabulary; a pre-internet comparison.",
        "pat": r"\bbiotechnolog|\bgenomics?\b|\bgene therapy\b|"
               r"\bmolecular biology\b|\brecombinant\b|\bstem cells?\b|"
               r"\bmonoclonal\b|\bnucleic acid|\bdna\b|\brna\b|"
               r"\bclinical trials?\b",
    },
    "green": {
        "label": "Clean energy and climate",
        "note": "Renewables and emissions. 'carbon' is admitted only in its "
                "climate senses, never carbon fiber, steel or carbonation.",
        "pat": r"\bsolar (cell|energy|panel|power|module)|\bphotovoltaic\b|"
               r"\brenewable energ|\bwind (turbine|energy|power)\b|"
               r"\bbio-?fuels?\b|\bbio-?diesel\b|\bcarbon (capture|credits|"
               r"emissions|footprint|offset|sequestration|dioxide)\b|"
               r"\bgreenhouse gas|\belectric vehicles?\b|\bfuel cells?\b|"
               r"\bgeothermal\b|\benergy efficien",
    },
}


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def main() -> int:
    ev = pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("gd")
    ).drop_nulls("gd").group_by("serial_number").agg(pl.col("gd").min())
    cw = set(pl.read_parquet(PROC / "uspto_sec_crosswalk.parquet")["owner_name"]
             .to_list())
    log(f"[setup] {ev.height:,} gate events, {len(cw):,} SEC-matched owners")

    agg = {k: {"n": 0, "reg": 0, "sec": 0, "failed": 0, "debut_n": 0,
               "debut_sec": 0, "by_class": {}, "quarters": {}}
           for k in COHORTS}
    corpus_n = 0

    for c in CLASSES:
        f = PROC / f"tm_class{c}.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f, columns=["serial_number", "filing_date",
                                        "registration_date", "owner_name",
                                        "goods_services"]).filter(
            pl.col("goods_services").is_not_null()
            & (pl.col("filing_date").fill_null("").str.len_chars() >= 8))
        corpus_n += d.height
        d = d.with_columns(
            pl.col("goods_services").str.to_lowercase().alias("gs"),
            (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
            .cast(pl.Int64).alias("reg"),
            pl.col("owner_name").is_in(list(cw)).cast(pl.Int64).alias("sec"),
            (pl.col("filing_date").str.slice(0, 4) + "Q"
             + ((pl.col("filing_date").str.slice(4, 2).cast(pl.Int32,
                                                            strict=False) - 1)
                // 3 + 1).cast(pl.Utf8)).alias("q"),
            pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d",
                                                     strict=False).alias("rd"),
        ).join(ev, on="serial_number", how="left").with_columns(
            (((pl.col("gd") - pl.col("rd")).dt.total_days() / 365.25)
             .is_between(GATE_LO, GATE_HI, closed="left")).fill_null(False)
            .cast(pl.Int64).alias("failed"),
            (pl.col("filing_date")
             == pl.col("filing_date").min().over("owner_name")).alias("is_debut"),
        )

        for k, spec in COHORTS.items():
            s = d.filter(pl.col("gs").str.contains("(?i)" + spec["pat"]))
            if not s.height:
                continue
            a = agg[k]
            a["n"] += s.height
            a["reg"] += int(s["reg"].sum())
            a["sec"] += int(s["sec"].sum())
            a["failed"] += int(s["failed"].sum())
            a["by_class"][c] = s.height
            # SEC reporting is a property of the owner, so a firm with 300 marks
            # would otherwise cast 300 votes. The debut split counts each owner
            # once, which is the denominator the paper's firm-level results use.
            db = s.filter(pl.col("is_debut"))
            a["debut_n"] += db.height
            a["debut_sec"] += int(db["sec"].sum())
            for q, n in s.group_by("q").len().iter_rows():
                a["quarters"].setdefault(c, {})
                a["quarters"][c][q] = a["quarters"][c].get(q, 0) + int(n)
        log(f"  [{c}] {d.height:,} filings scanned")
        del d
        gc.collect()

    out = {"corpus_n": corpus_n, "cohorts": {}}
    for k, spec in COHORTS.items():
        a = agg[k]
        n = max(a["n"], 1)
        out["cohorts"][k] = {
            "label": spec["label"], "note": spec["note"],
            "pattern": spec["pat"], **a,
            "share_of_corpus": a["n"] / corpus_n,
            "reg_pct": 100.0 * a["reg"] / n,
            "failed_pct": 100.0 * a["failed"] / max(a["reg"], 1),
            "sec_pct": 100.0 * a["sec"] / n,
            "debut_sec_pct": 100.0 * a["debut_sec"] / max(a["debut_n"], 1),
        }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "theme_cohorts.json").write_text(json.dumps(out, indent=1))

    print("")
    print(f"corpus: {corpus_n:,} filings with goods/services text")
    print("")
    print(f"{'cohort':<14}{'filings':>11}{'share':>8}{'reg%':>8}"
          f"{'gatefail%':>11}{'SEC% (all)':>12}{'SEC% (debut)':>14}")
    for k in COHORTS:
        o = out["cohorts"][k]
        print(f"{k:<14}{o['n']:>11,}{o['share_of_corpus']:>7.2%}"
              f"{o['reg_pct']:>8.1f}{o['failed_pct']:>11.1f}"
              f"{o['sec_pct']:>12.2f}{o['debut_sec_pct']:>14.2f}")
    print("")
    print("[done] theme_cohorts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
