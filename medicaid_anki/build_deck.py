#!/usr/bin/env python3
"""Build the Medicaid Anki deck from the YAML card sources in ``cards/``.

Outputs:
  dist/Medicaid.apkg        importable Anki package (three subdecks)
  dist/*.tsv                plain-text fallback, one file per subdeck

Usage:
    pip install genanki pyyaml
    python build_deck.py
"""

from __future__ import annotations

import csv
import html
import pathlib
import re
import sys

import yaml

try:
    import genanki
except ImportError:  # pragma: no cover - the TSV path still works without it
    genanki = None

HERE = pathlib.Path(__file__).resolve().parent
CARD_DIR = HERE / "cards"
DIST_DIR = HERE / "dist"

# Stable IDs. Never change these: Anki uses them to recognise an updated deck
# or note type on re-import instead of creating duplicates.
MODEL_BASIC_ID = 1607392311
MODEL_CLOZE_ID = 1607392312
DECK_ROOT_ID = 2059400110

CSS = """
.card {
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 19px;
  line-height: 1.55;
  text-align: left;
  color: #1a1a1a;
  background-color: #fbfbfa;
  padding: 14px 18px;
}
.nightMode.card, .night_mode .card { color: #e8e8e8; background-color: #2c2c2e; }
.q { font-weight: 600; }
.a { }
hr#answer { border: none; border-top: 1px solid #d8d8d4; margin: 14px 0; }
.nightMode hr#answer, .night_mode hr#answer { border-top-color: #4a4a4c; }
ul { margin: 6px 0 6px 0; padding-left: 22px; }
li { margin: 3px 0; }
.extra {
  margin-top: 14px;
  font-size: 15px;
  line-height: 1.45;
  color: #5a5a5a;
  border-left: 3px solid #c9c9c4;
  padding-left: 10px;
}
.nightMode .extra, .night_mode .extra { color: #a8a8a8; border-left-color: #555; }
.cloze { font-weight: 700; color: #0b6fa4; }
.nightMode .cloze, .night_mode .cloze { color: #6fb8e0; }
"""

BASIC_TEMPLATE = [
    {
        "name": "Recall",
        "qfmt": '<div class="q">{{Front}}</div>',
        "afmt": '<div class="q">{{Front}}</div><hr id="answer">'
        '<div class="a">{{Back}}</div>'
        '{{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}',
    }
]

CLOZE_TEMPLATE = [
    {
        "name": "Cloze",
        "qfmt": '<div class="q">{{cloze:Text}}</div>',
        "afmt": '<div class="q">{{cloze:Text}}</div>'
        '{{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}',
    }
]


def make_models():
    basic = genanki.Model(
        MODEL_BASIC_ID,
        "Medicaid Basic",
        fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Extra"}],
        templates=BASIC_TEMPLATE,
        css=CSS,
    )
    cloze = genanki.Model(
        MODEL_CLOZE_ID,
        "Medicaid Cloze",
        fields=[{"name": "Text"}, {"name": "Extra"}],
        templates=CLOZE_TEMPLATE,
        css=CSS,
        model_type=genanki.Model.CLOZE,
    )
    return basic, cloze


CLOZE_RE = re.compile(r"\{\{c\d+::.*?\}\}", re.DOTALL)


def render(text: str) -> str:
    """Turn the plain-text card source into the small HTML subset the cards use.

    Lines starting with "- " become an unordered list; every other newline
    becomes a <br>. Literal HTML is passed through, so a card can drop in a
    <b> or a <i> when it needs one; everything else is escaped.
    """
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    out: list[str] = []
    bullets: list[str] = []

    def flush() -> None:
        if bullets:
            out.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for line in text.split("\n"):
        line = line.rstrip()
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        else:
            flush()
            if line:
                out.append(line)

    flush()

    body = ""
    for i, chunk in enumerate(out):
        if chunk.startswith("<ul>"):
            body += chunk
        else:
            if body and not body.endswith(("</ul>", ">")):
                body += "<br>"
            elif body and body.endswith("</ul>"):
                pass
            elif body:
                body += "<br>"
            body += chunk
    return body


def load_files() -> list[dict]:
    files = sorted(CARD_DIR.glob("*.yaml"))
    if not files:
        sys.exit(f"no card files found in {CARD_DIR}")
    decks = []
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not data or "cards" not in data:
            sys.exit(f"{path.name}: expected top-level keys 'deck' and 'cards'")
        data["_path"] = path
        decks.append(data)
    return decks


def validate(decks: list[dict]) -> None:
    seen: set[str] = set()
    problems: list[str] = []
    for deck in decks:
        for card in deck["cards"]:
            cid = card.get("id")
            if not cid:
                problems.append(f"{deck['_path'].name}: card without an id")
                continue
            if cid in seen:
                problems.append(f"duplicate card id {cid}")
            seen.add(cid)
            ctype = card.get("type", "basic")
            if ctype == "basic":
                if not card.get("front") or not card.get("back"):
                    problems.append(f"{cid}: basic card needs both front and back")
            elif ctype == "cloze":
                if not card.get("text"):
                    problems.append(f"{cid}: cloze card needs text")
                elif not CLOZE_RE.search(str(card["text"])):
                    problems.append(f"{cid}: cloze card has no {{{{c1::...}}}} deletion")
            else:
                problems.append(f"{cid}: unknown type {ctype!r}")
    if problems:
        sys.exit("card validation failed:\n  " + "\n  ".join(problems))


def build_apkg(decks: list[dict]) -> pathlib.Path:
    basic_model, cloze_model = make_models()
    anki_decks = []

    for offset, deck in enumerate(decks):
        anki_deck = genanki.Deck(DECK_ROOT_ID + offset + 1, deck["deck"])
        base_tags = [t.replace(" ", "-") for t in deck.get("tags", [])]
        for card in deck["cards"]:
            tags = base_tags + [
                str(t).replace(" ", "-") for t in card.get("tags", []) or []
            ]
            extra = render(card.get("extra"))
            if card.get("type", "basic") == "cloze":
                note = genanki.Note(
                    model=cloze_model,
                    fields=[render(card["text"]), extra],
                    tags=tags,
                    guid=genanki.guid_for(card["id"]),
                )
            else:
                note = genanki.Note(
                    model=basic_model,
                    fields=[render(card["front"]), render(card["back"]), extra],
                    tags=tags,
                    guid=genanki.guid_for(card["id"]),
                )
            anki_deck.add_note(note)
        anki_decks.append(anki_deck)

    DIST_DIR.mkdir(exist_ok=True)
    out = DIST_DIR / "Medicaid.apkg"
    genanki.Package(anki_decks).write_to_file(out)
    return out


def strip_html(text: str) -> str:
    text = re.sub(r"<li>", "• ", text)
    text = re.sub(r"</li>", "; ", text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip(" ;")


def build_tsv(decks: list[dict]) -> list[pathlib.Path]:
    DIST_DIR.mkdir(exist_ok=True)
    written = []
    for deck in decks:
        name = deck["_path"].stem + ".tsv"
        path = DIST_DIR / name
        base_tags = [t.replace(" ", "-") for t in deck.get("tags", [])]
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("#separator:tab\n#html:true\n#notetype column:1\n#deck column:2\n")
            fh.write("#tags column:6\n")
            writer = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for card in deck["cards"]:
                tags = " ".join(
                    base_tags
                    + [str(t).replace(" ", "-") for t in card.get("tags", []) or []]
                )
                extra = render(card.get("extra"))
                if card.get("type", "basic") == "cloze":
                    writer.writerow(
                        ["Cloze", deck["deck"], render(card["text"]), extra, "", tags]
                    )
                else:
                    writer.writerow(
                        [
                            "Basic",
                            deck["deck"],
                            render(card["front"]),
                            render(card["back"]) + (f"<br><br><i>{extra}</i>" if extra else ""),
                            "",
                            tags,
                        ]
                    )
        written.append(path)
    return written


def main() -> None:
    decks = load_files()
    validate(decks)

    total = sum(len(d["cards"]) for d in decks)
    for deck in decks:
        print(f"  {len(deck['cards']):>4} cards  {deck['deck']}")
    print(f"  {total:>4} cards  total")

    tsvs = build_tsv(decks)
    for path in tsvs:
        print(f"wrote {path.relative_to(HERE)}")

    if genanki is None:
        print("genanki not installed - skipped .apkg (pip install genanki)")
        return
    apkg = build_apkg(decks)
    print(f"wrote {apkg.relative_to(HERE)}")


if __name__ == "__main__":
    main()
