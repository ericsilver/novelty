"""Worked exhibit: how the same filing is encoded under term-level
(dynamic unigram+bigram) scoring and under topic scoring at each T.

Examples: AMAZON.COM (1997, class 035) and KOMBUCHA (1997, class 032),
plus ETHEREUM (2018, class 009) for the emergent-category case.

For each filing:
  - the goods/services text
  - its term encoding: tokens with within-class document frequency and
    a flag for absence from the topic-model vocabulary
  - for every persisted topic model (topic_model{,_T200,_T500}.joblib):
    the top-5 topic loadings with each topic's top words

Output: paper/results/exhibit_encodings.json (+ readable stderr dump)
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
RES = REPO / "paper" / "results"

EXAMPLES = [
    ("035", "75277670", "AMAZON.COM (1997)"),
    ("032", None, "KOMBUCHA (1997)"),      # resolve serial by mark/year
    ("009", None, "ETHEREUM (2018)"),
]


def resolve(cls: str, serial: str | None, label: str) -> dict:
    tm = pl.read_parquet(
        PROC / f"tm_class{cls}.parquet",
        columns=["serial_number", "filing_date", "mark_identification",
                 "goods_services"])
    if serial:
        row = tm.filter(pl.col("serial_number") == serial).row(0, named=True)
    else:
        mark, year = label.split(" (")[0], label.split("(")[1][:4]
        row = tm.filter(
            pl.col("mark_identification").str.to_uppercase().str.contains(f"^{mark}$|^{mark} ")
            & pl.col("filing_date").str.starts_with(year)
        ).sort("filing_date").row(0, named=True)
    return {"cls": cls, "label": label, "serial": row["serial_number"],
            "filing_date": row["filing_date"], "text": row["goods_services"]}


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from novelty.dictionary import STOPWORDS, _make_analyzer
    analyzer = _make_analyzer(frozenset(STOPWORDS), (1, 2))

    models = {}
    for suffix, name in (("", "T=50"), ("_T200", "T=200"), ("_T500", "T=500")):
        p = PROC / f"topic_model{suffix}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
            print(f"[model] {name} loaded", file=sys.stderr, flush=True)

    out = []
    for cls, serial, label in EXAMPLES:
        info = resolve(cls, serial, label)
        text = info["text"]
        tokens = analyzer(text)
        counts = collections.Counter(tokens)

        vocab = pl.read_parquet(PROC / f"vocab_class{cls}.parquet").filter(
            pl.col("doc_freq") >= 50)
        df_map = dict(zip(vocab["token"].to_list(), vocab["doc_freq"].to_list()))

        token_rows = []
        for tok, c in counts.most_common():
            token_rows.append({
                "token": tok, "count": c,
                "class_df": df_map.get(tok),
                "in_term_vocab": tok in df_map,
            })

        enc = {"info": info, "n_tokens": len(tokens),
               "tokens": token_rows, "topics": {}}
        for name, m in models.items():
            vec, lda = m["vectorizer"], m["lda"]
            feats = vec.get_feature_names_out()
            fset = set(feats)
            X = vec.transform([text])
            theta = lda.transform(X)[0]
            top = np.argsort(theta)[::-1][:5]
            comp = lda.components_
            enc["topics"][name] = {
                "oov_tokens": [t for t in counts if t not in fset],
                "top_topics": [
                    {"topic": int(k), "weight": float(theta[k]),
                     "top_words": [feats[i] for i in comp[k].argsort()[-8:][::-1]]}
                    for k in top],
            }
        out.append(enc)

        # readable dump
        print(f"\n===== {label}  [{info['serial']}, filed {info['filing_date']}]",
              file=sys.stderr, flush=True)
        print(f"TEXT: {text[:300]}", file=sys.stderr, flush=True)
        print("TERM ENCODING (token, count, class df, in term vocab):",
              file=sys.stderr, flush=True)
        for r in token_rows[:18]:
            print(f"   {r['token'][:34]:<36} x{r['count']} df={r['class_df']} "
                  f"{'OOV-class!' if not r['in_term_vocab'] else ''}",
                  file=sys.stderr, flush=True)
        for name, tp in enc["topics"].items():
            print(f"-- {name}: OOV for topic model: {tp['oov_tokens'][:8]}",
                  file=sys.stderr, flush=True)
            for t in tp["top_topics"][:3]:
                print(f"   topic {t['topic']:>3} w={t['weight']:.3f}  "
                      f"{', '.join(t['top_words'][:6])}",
                      file=sys.stderr, flush=True)

    (RES / "exhibit_encodings.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("\n[done]", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
