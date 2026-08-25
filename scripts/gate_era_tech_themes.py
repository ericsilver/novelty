"""The technology swing, class by class and theme by theme.

Section "The penalty is not modern, and it is not general" splits the corpus
into four technology classes (009, 035, 038, 042) and forty-one others, and
finds the whole era swing in the gate penalty -- strongly positive for filings
made into the late 1990s, negative for those made 2000-2004, positive again from
2008 -- inside the four. That leaves two questions the split cannot answer: is
it all four classes or one of them, and is it a property of particular kinds of
language within them, or of technology filings generally.

This script answers both on the production scoring. Each settled registration
in the four classes is given its dominant theme under the production T=50
model, and the top-minus-bottom lead-quintile contrast in first-gate failure is
computed per class and per theme, by filing era and by registration cohort.
Three versions of the pooled technology contrast are reported for each era:
raw (quintiles within the era's technology filings, as in the paper's table),
within class x cohort cells, and within theme x class x cohort cells. If the
swing shrinks when quintiles are assigned inside themes, the leading filings of
each era were concentrated in particular themes and the swing is partly a
story about which themes were filed into; if it does not shrink, being leading
is punished or rewarded inside every theme alike.

Outputs:
  paper/results/gate_era_tech_themes.json
  paper/results/gate_era_tech_themes.png       per-class cohort series
  paper/results/gate_era_tech_classes.tex      class x era table
  paper/results/gate_era_tech_themes.tex       theme x era table
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

SRC = os.environ.get("SURPRISE_SRC", "rolling")
TECH = ["009", "035", "038", "042"]
NAMES = {"009": "Electrical, software", "035": "Business services",
         "038": "Telecommunications", "042": "Scientific, computer services"}
EDGE = "2026-04-02"
RESOLVED_AGE = 9.0
WIDE_LO, WIDE_HI = 4.0, 8.5
ERAS = [("1995-1999", 1995, 1999), ("2000-2004", 2000, 2004),
        ("2005-2007", 2005, 2007), ("2008-2014", 2008, 2014)]
T = 50
TOKEN = r"(?u)\b[a-z][a-z\-]{2,}\b"
MIN_THEME = 20_000      # settled registrations a theme needs to be tabled
MIN_CELL = 3_000
EPS = 1e-12


def log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def contrast_arrays(q: np.ndarray, y: np.ndarray) -> dict | None:
    """Top-minus-bottom quintile contrast in the failure rate, with its SE."""
    m1, m5 = q == 0, q == 4
    n1, n5 = int(m1.sum()), int(m5.sum())
    if n1 < 200 or n5 < 200:
        return None
    p1, p5 = float(y[m1].mean()), float(y[m5].mean())
    se = ((p1 * (1 - p1) / n1) + (p5 * (1 - p5) / n5)) ** 0.5
    return {"n": int(len(y)), "base": float(y.mean()), "lift": p5 - p1, "se": se,
            "p1": p1, "p5": p5}


def contrast(df: pl.DataFrame, min_n: int = MIN_CELL) -> dict | None:
    """Raw contrast: quintiles assigned within the frame passed in."""
    if df.height < min_n:
        return None
    s = df.sort(["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
    return contrast_arrays(s["q"].to_numpy(), s["failed"].to_numpy())


def contrast_within(df: pl.DataFrame, cells: list[str], min_n: int = MIN_CELL) -> dict | None:
    """Contrast with quintiles assigned inside each cell, then pooled."""
    if df.height < min_n:
        return None
    s = df.sort(cells + ["topic_dkl", "serial_number"]).with_columns(
        ((pl.col("topic_dkl").rank("ordinal").over(cells) - 1) * 5
         // pl.len().over(cells)).cast(pl.Int8).alias("q"))
    return contrast_arrays(s["q"].to_numpy(), s["failed"].to_numpy())


def load_events() -> pl.DataFrame:
    return pl.scan_parquet(PROC / "case_events.parquet").filter(
        pl.col("code").is_in(["C8..", "C71T"]) & (pl.col("date") > 19000000)
    ).select("serial_number", "date").collect().with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("d")
    ).drop_nulls("d").group_by("serial_number").agg(pl.col("d").min())


def load_class(c: str, ev: pl.DataFrame, lda, vec) -> pl.DataFrame:
    tm = pl.read_parquet(
        PROC / f"tm_class{c}.parquet",
        columns=["serial_number", "filing_date", "registration_date", "goods_services"]).filter(
        (pl.col("registration_date").fill_null("").str.len_chars() >= 8)
        & (pl.col("filing_date").fill_null("").str.len_chars() >= 8)
        & pl.col("goods_services").is_not_null())
    sc = pl.read_parquet(PROC / f"{SRC}_surprise_class{c}.parquet",
                         columns=["serial_number", "topic_dkl"]).filter(
        pl.col("topic_dkl").is_finite())
    d = tm.join(sc, on="serial_number", how="inner").unique(subset="serial_number")
    del tm, sc
    edge = pl.lit(EDGE).str.strptime(pl.Date, "%Y-%m-%d")
    d = d.with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("rd"),
        pl.col("filing_date").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("fd"),
    ).drop_nulls(["rd", "fd"]).filter(
        pl.col("rd").dt.offset_by(f"{int(RESOLVED_AGE * 365.25)}d") <= edge)
    d = d.join(ev, on="serial_number", how="left").with_columns(
        ((pl.col("d") - pl.col("rd")).dt.total_days() / 365.25).alias("age"),
        pl.col("rd").dt.year().alias("ry"),
        pl.col("fd").dt.year().alias("fy"),
        pl.lit(c).alias("cls"),
    ).with_columns(
        ((pl.col("age") >= WIDE_LO) & (pl.col("age") < WIDE_HI))
        .fill_null(False).cast(pl.Float64).alias("failed"))
    # Dominant theme under the production model, on the settled sample only.
    texts = d["goods_services"].to_list()
    dom = np.empty(len(texts), dtype=np.int16)
    top = np.empty(len(texts), dtype=np.float32)
    for s in range(0, len(texts), 200_000):
        th = lda.transform(vec.transform(texts[s:s + 200_000]))
        dom[s:s + 200_000] = th.argmax(axis=1)
        top[s:s + 200_000] = th.max(axis=1)
    d = d.with_columns(pl.Series("theme", dom), pl.Series("theme_share", top)).drop(
        "goods_services", "filing_date", "registration_date")
    log(f"  [{c}] {d.height:,} settled registrations themed")
    return d


def fmt(r: dict | None) -> str:
    return f"${100*r['lift']:+.2f}$ ({100*r['se']:.2f})" if r else "---"


def main() -> int:
    m = joblib.load(PROC / "topic_model.joblib")
    lda, vocab = m["lda"], m["vocabulary"]
    vec = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern=TOKEN,
                          ngram_range=(1, 2))
    words = json.loads((PROC / "topic_lda_meta.json").read_text())["top_words"]
    ev = load_events()
    log(f"[events] {ev.height:,} gate events")

    d = pl.concat([load_class(c, ev, lda, vec) for c in TECH])
    del ev
    gc.collect()
    log(f"[frame] {d.height:,} settled technology registrations")

    out = {"scoring": SRC, "T": T, "classes": TECH, "n": int(d.height),
           "by_class_era": {}, "by_class_cohort": {}, "pooled_era": {},
           "by_theme_era": {}, "theme_words": {}, "composition": {}}

    # 1. Class x era, and class x cohort.
    for c in TECH:
        dc = d.filter(pl.col("cls") == c)
        out["by_class_era"][c] = {lab: contrast(dc.filter(pl.col("fy").is_between(lo, hi)))
                                  for lab, lo, hi in ERAS}
        out["by_class_cohort"][c] = {str(ry): contrast(dc.filter(pl.col("ry") == ry))
                                     for ry in sorted(dc["ry"].unique().to_list())}
        log(f"  class {c}: " + "  ".join(
            f"{lab} {fmt(out['by_class_era'][c][lab])}" for lab, _, _ in ERAS))

    # 2. Pooled technology contrast per era: raw, within class x cohort,
    #    within theme x class x cohort.
    for lab, lo, hi in ERAS:
        de = d.filter(pl.col("fy").is_between(lo, hi))
        out["pooled_era"][lab] = {
            "raw": contrast(de),
            "within_class_cohort": contrast_within(de, ["cls", "ry"]),
            "within_theme_class_cohort": contrast_within(de, ["theme", "cls", "ry"]),
        }
        r = out["pooled_era"][lab]
        log(f"  era {lab}: raw {fmt(r['raw'])}  cls x coh {fmt(r['within_class_cohort'])}"
            f"  theme x cls x coh {fmt(r['within_theme_class_cohort'])}")

    # 3. Theme x era, for themes large enough to table.
    counts = d.group_by("theme").agg(pl.len().alias("n")).sort("n", descending=True)
    themes = [int(t) for t, n in counts.iter_rows() if n >= MIN_THEME]
    for t in themes:
        dt = d.filter(pl.col("theme") == t)
        out["theme_words"][str(t)] = words[str(t)][:6]
        out["by_theme_era"][str(t)] = {
            "n": int(dt.height),
            "base": float(dt["failed"].mean()),
            "eras": {lab: contrast(dt.filter(pl.col("fy").is_between(lo, hi)))
                     for lab, lo, hi in ERAS},
            "all": contrast_within(dt, ["cls", "ry"]),
        }

    # 4. Composition: where the leading fifth of each era sits, by theme,
    #    against where all of that era's filings sit.
    for lab, lo, hi in ERAS:
        de = d.filter(pl.col("fy").is_between(lo, hi)).sort(
            ["topic_dkl", "serial_number"]).with_columns(
            ((pl.col("topic_dkl").rank("ordinal") - 1) * 5 // pl.len()).cast(pl.Int8).alias("q"))
        tot = de.group_by("theme").agg(pl.len().alias("n_all"))
        top = de.filter(pl.col("q") == 4).group_by("theme").agg(pl.len().alias("n_top"))
        n_top_all = de.filter(pl.col("q") == 4).height
        j = tot.join(top, on="theme", how="left").fill_null(0).with_columns(
            (pl.col("n_all") / de.height).alias("share_all"),
            (pl.col("n_top") / n_top_all).alias("share_top"))
        out["composition"][lab] = {
            str(int(r["theme"])): {"share_all": r["share_all"], "share_top": r["share_top"]}
            for r in j.iter_rows(named=True) if int(r["theme"]) in themes}

    # Figure: per-class cohort series.
    colors = {"009": "#2b6cb0", "035": "#c05621", "038": "#2f855a", "042": "#6b46c1"}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0, color="#4a5568", lw=1.0, zorder=1)
    for c in TECH:
        ys = out["by_class_cohort"][c]
        xs = [int(y) for y, r in ys.items() if r]
        ls = [100 * ys[str(x)]["lift"] for x in xs]
        es = [196 * ys[str(x)]["se"] for x in xs]
        ax.errorbar(xs, ls, yerr=es, fmt="o-", color=colors[c], lw=1.8, ms=4,
                    elinewidth=0.9, capsize=0, zorder=3, label=f"{c} {NAMES[c]}")
    ax.set_xlabel("Registration cohort")
    ax.set_ylabel("Gate failure, top minus bottom\nlead quintile (pp)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", color="#e2e8f0", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    RES.mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / "gate_era_tech_themes.png", dpi=200)
    plt.close(fig)

    # Tex fragments.
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            "Class & " + " & ".join(lab for lab, _, _ in ERAS) + r" \\", r"\midrule"]
    for c in TECH:
        rows.append(f"{c} {NAMES[c]} & " + " & ".join(
            fmt(out["by_class_era"][c][lab]) for lab, _, _ in ERAS) + r" \\")
    rows.append(r"\midrule")
    for key, name in (("raw", "Four classes pooled, raw"),
                      ("within_class_cohort", r"\quad within class$\times$cohort"),
                      ("within_theme_class_cohort", r"\quad within theme$\times$class$\times$cohort")):
        rows.append(f"{name} & " + " & ".join(
            fmt(out["pooled_era"][lab][key]) for lab, _, _ in ERAS) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (RES / "gate_era_tech_classes.tex").write_text("\n".join(rows) + "\n")

    rows = [r"\begin{tabular}{p{5.2cm}rrrrr}", r"\toprule",
            "Theme (top words) & $n$ & " + " & ".join(lab for lab, _, _ in ERAS) + r" \\",
            r"\midrule"]
    for t in themes:
        e = out["by_theme_era"][str(t)]
        rows.append(", ".join(out["theme_words"][str(t)][:5]) + f" & {e['n']:,} & "
                    + " & ".join(fmt(e["eras"][lab]) for lab, _, _ in ERAS) + r" \\")
    rows.append(r"\midrule")
    pooled_n = sum(out["pooled_era"][lab]["raw"]["n"] for lab, _, _ in ERAS)
    rows.append(f"All themes, within theme$\\times$class$\\times$cohort & {pooled_n:,} & "
                + " & ".join(fmt(out["pooled_era"][lab]["within_theme_class_cohort"])
                             for lab, _, _ in ERAS) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (RES / "gate_era_tech_themes.tex").write_text("\n".join(rows) + "\n")

    (RES / "gate_era_tech_themes.json").write_text(json.dumps(out, indent=1))
    log("[done] gate_era_tech_themes.{json,png,tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
