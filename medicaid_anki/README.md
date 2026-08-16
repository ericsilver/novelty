# Medicaid Anki deck

A 264-card spaced-repetition deck covering Medicaid fundamentals, the program
design space and its alternatives, and the published research on efficacy and
reform.

```
dist/Medicaid.apkg      import this into Anki
cards/*.yaml            the source text — edit here, not in Anki
build_deck.py           regenerates dist/ from cards/
dist/*.tsv              plain-text fallback if .apkg import fails
```

## Import

Anki → **File → Import** → select `dist/Medicaid.apkg`. Three subdecks appear
under a parent deck `Medicaid`. Nothing else in your collection is touched: the
deck ships its own two note types (`Medicaid Basic`, `Medicaid Cloze`) with
their own IDs.

Re-importing an updated `.apkg` **updates** existing notes rather than
duplicating them — note GUIDs are derived from the stable card IDs in the YAML,
and scheduling history is preserved.

The `.tsv` files are a fallback for AnkiDroid or any client that chokes on the
package. They carry `#deck` and `#notetype` headers, so File → Import handles
them without column mapping; the source note goes into the Back field there
rather than its own field.

## Contents

| Deck | Cards | Covers |
|---|---:|---|
| 1. Fundamentals | 120 | Statute and structure, eligibility, benefits, financing and FMAP, delivery systems, LTSS, waivers, key litigation, the current policy environment |
| 2. Program Design & Alternatives | 68 | Block grants and per capita caps, expansion alternatives, work requirement design, cost sharing, managed care and payment design, drug purchasing, LTSS and dual integration, evaluation design |
| 3. Research: Efficacy & Reform | 76 | Identification strategies, the Oregon experiment, mortality and long-run childhood effects, financial protection, labor supply, crowd-out, managed care, access and payment, unwinding, synthesis and open questions |

Every card is tagged. Useful ones to filter on:

```
tag:financing   tag:eligibility   tag:ltss     tag:managed-care   tag:waivers
tag:oregon      tag:mortality     tag:methods  tag:work-requirements
tag:obbba       tag:synthesis     tag:framework
```

To study one thread, make a filtered deck: `deck:Medicaid tag:ltss`.

## Two things to know before you study

**Card structure.** Each card has a question, an answer, and a *source note* in
smaller type below the divider. The source note carries the citation, the
caveat, or the reason the fact matters. It is not tested — read it, don't drill
it. Most of the actual understanding is there rather than in the answer.

**Date-sensitive figures.** Enrollment, spending, expansion state counts, and
everything touching the 2025 reconciliation law (H.R. 1) move. Those cards say
so in the source note and give an "as of" date. The law's major provisions
phase in between 2026 and 2032, so several cards describe rules that are
scheduled but not yet operative — verify status before relying on them in
writing. Structural material (statutory architecture, waiver authorities,
financing mechanics, research findings) is stable.

## Suggested study order

1. **Deck 1 first, to card F065 or so** — statute, eligibility, benefits, and
   financing. The financing cards (FMAP, non-federal share, provider taxes,
   DSH/UPL/directed payments) are the ones everything else depends on, and they
   are the hardest to pick up by osmosis later.
2. **Deck 3's methods cards (R001-R010)** early, out of order. Ten cards, and
   they change how you read everything else.
3. **Deck 2 and the rest of deck 3 in parallel.** They interlock — a design
   card names a trade-off, and a research card gives the evidence on it.

Twenty new cards a day finishes the deck in about two weeks with reviews
settling around 40-60 cards a day.

## Editing and rebuilding

Cards live in `cards/*.yaml`:

```yaml
- id: F001              # stable — changing it orphans review history
  type: basic           # or: cloze
  tags: [statute]
  front: |-
    What statute creates Medicaid, and when was it enacted?
  back: |-
    Title XIX of the Social Security Act ...
  extra: |-
    The same 1965 law created Medicare as Title XVIII ...
```

Cloze cards use `text:` with `{{c1::...}}` deletions instead of front/back.
Lines starting with `- ` become bullets; inline `<b>` and `<i>` pass through.

```bash
pip install genanki pyyaml
python build_deck.py
```

The build validates as it goes: duplicate IDs, missing fields, and cloze cards
with no deletion all fail the build rather than producing a broken deck.

## Sources

Facts were drawn from the statute and regulations, CMS and MACPAC material, CRS
reports, KFF, and the primary research literature, with current-figure and
2025-2026 policy items checked against published sources in August 2026.

Standing references worth keeping open:

- **MACPAC** — MACStats data book; issue briefs; the best single source for
  program statistics
- **KFF** — State Health Facts; the annual Medicaid Budget Survey
- **CMS / Medicaid.gov** — waiver and SPA documents, expenditure reports
- **CRS** — e.g. R42640 (Medicaid financing and expenditures), R48569 (health
  coverage provisions of H.R. 1)
- **CBO** — baselines and cost estimates for proposed changes
- **42 U.S.C. ch. 7 subch. XIX** and **42 C.F.R. parts 430-460** for anything
  legal; secondary summaries of Medicaid age quickly

The research deck cites papers by author, year, and journal so each one can be
pulled directly. Where a figure is given approximately, that is deliberate —
go to the paper before quoting a number in writing.
