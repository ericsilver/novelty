"""Generate a face-validation document for the 50 LDA themes (PROPOSAL_diffusion §2.4).

For each topic the rater sees:
  - Top 10 words
  - 8 representative trademark marks (highest topic-weight, post-burn-in, min length)
  - Per-class peak prevalence + year, and where (1-class niche / multi-class / general-purpose)
  - Rating box: [ ] coherent / [ ] partial / [ ] noise + optional label

The user marks each topic. Themes promoted to headline require >= 0.8 agreement
across raters (per §2.4); here we provide the single-rater form ready to fill in.

Output: paper/face_validation.md
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT = REPO / "paper" / "face_validation.md"
CLASSES = {"009": "software", "042": "tech-svcs", "039": "transport", "035": "advert-retail"}
N_EXAMPLES = 8
TAU = 0.30
MIN_TERMS = 12


def main() -> int:
    res = json.loads((REPO / "paper" / "results" / "diff3_themes.json").read_text())
    top_words = {int(k): v for k, v in res["top_words"].items()}
    T = res["T"]

    filings = pl.read_parquet(DATA / "filing_topics.parquet")
    panel = pl.read_parquet(DATA / "theme_panel.parquet")

    # mark_identification: join with tm_class records (need mark+n_terms-ish; we used n_terms via X earlier but it's not in filing_topics; approximate via raw word count of goods_services)
    rec_parts = []
    for c in CLASSES:
        d = pl.read_parquet(DATA / f"tm_class{c}.parquet",
                            columns=["serial_number", "mark_identification", "goods_services"])
        rec_parts.append(d)
    recs = pl.concat(rec_parts).with_columns(
        pl.col("goods_services").str.split(" ").list.len().alias("n_words"),
    ).select("serial_number", "mark_identification", "n_words")

    joined = filings.join(recs, on="serial_number", how="inner").filter(
        (pl.col("top_weight") >= TAU) & (pl.col("n_words") >= MIN_TERMS)
        & (pl.col("year") >= 1996) & (pl.col("year") <= 2020)  # post-burn-in, full-window era
    )

    # per-class panel summary per theme
    panel_pd = panel.to_pandas()

    out_lines = []
    out_lines.append("# Face-validation form for the 50 LDA themes\n")
    out_lines.append("Per PROPOSAL_diffusion §2.4: each rater marks every theme as\n"
                     "`coherent` (the cluster names a recognizable business-model\n"
                     "component), `partial` (it captures something but is mixed/noisy),\n"
                     "or `noise` (no recognizable theme). A theme is promoted to the\n"
                     "headline set if rater agreement on `coherent` is >= 0.8.\n\n"
                     "Source: 200k stratified granted-filing sample from classes\n"
                     "009/042/039/035, LDA T=50, min_df=200 (V=69,361). For each\n"
                     "topic, eight representative marks are shown (top-topic weight\n"
                     "≥ 0.30, post-burn-in 1996–2020, ≥ 12 words).\n\n")
    out_lines.append("---\n\n")

    for k in range(T):
        out_lines.append(f"## Theme T{k:02d}\n\n")
        out_lines.append(f"**Top words:** {', '.join(top_words[k][:10])}\n\n")

        # per-class summary
        sub = panel_pd[panel_pd.theme == k]
        if len(sub):
            by_class = []
            for c, label in CLASSES.items():
                cs = sub[sub.nice == c]
                if len(cs):
                    peak_row = cs.loc[cs.share_top.idxmax()]
                    by_class.append(f"{label}: peak {peak_row.share_top:.2%} ({int(peak_row.year)})")
                else:
                    by_class.append(f"{label}: --")
            out_lines.append(f"**Per-class peak prevalence:** {' | '.join(by_class)}\n\n")
        else:
            out_lines.append("**Per-class peak prevalence:** (theme never exceeds floor)\n\n")

        examples = (joined.filter(pl.col("top_topic") == k)
                          .sort("top_weight", descending=True)
                          .head(N_EXAMPLES))
        if examples.height:
            out_lines.append("**Representative marks:**\n\n")
            for r in examples.iter_rows(named=True):
                mark = (r["mark_identification"] or "").strip() or "(no mark text)"
                out_lines.append(
                    f"- {r['year']} `{CLASSES[r['nice']]}` w={r['top_weight']:.2f}  "
                    f"**{mark[:60]}**\n"
                )
            out_lines.append("\n")
        else:
            out_lines.append("**Representative marks:** (no filings meet the threshold)\n\n")

        out_lines.append("**Rating:**\n\n")
        out_lines.append("- [ ] coherent\n- [ ] partial\n- [ ] noise\n\n")
        out_lines.append("**If coherent, short label (e.g., 'on-demand transportation', 'subscription replenishment'):** ____________________________\n\n")
        out_lines.append("---\n\n")

    OUT.write_text("".join(out_lines), encoding="utf-8")
    print(f"[done] {T} topics -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
