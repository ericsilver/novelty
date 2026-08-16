# Medicaid Anki decks

A 472-note / 620-card spaced-repetition series covering Medicaid fundamentals,
program design and its alternatives, the published research on efficacy and
reform, the people and institutions producing that research, and the craft of
evaluating benefit programs.

```
dist/Medicaid.apkg      import this into Anki
cards/*.yaml            the source text — edit here, not in Anki
build_deck.py           regenerates dist/ from cards/
dist/*.tsv              plain-text fallback, one per deck
```

## The series

Deck numbers are the suggested study order.

| Deck | Notes | Purpose |
|---|---:|---|
| **1. Core Facts (Cloze)** | 95 | Cloze drills for facts that should be automatic — thresholds, formulas, rates, statutory cites, dates, headline findings |
| **2. Fundamentals** | 120 | Statute and structure, eligibility, benefits, financing, delivery systems, LTSS, waivers, litigation, current policy |
| **3. Program Design & Alternatives** | 68 | Capped financing, expansion alternatives, work requirement design, managed care and payment reform, drug purchasing, rebalancing, dual integration |
| **4. Research: Efficacy & Reform** | 76 | Identification strategies, Oregon, mortality, long-run childhood effects, financial protection, labor supply, crowd-out, managed care, synthesis |
| **5. People & Institutions** | 43 | Who publishes what — researchers, think tanks, agencies, journals, and how to weigh each |
| **6. Benefit Evaluation** | 70 | Framing, designs, measurement, inference, welfare and cost, take-up and burden, evaluation practice |

Deck 1 is cloze throughout; the rest are question-and-answer with a source note.
Decks 1–4 overlap deliberately: deck 1 drills a number cold, deck 2 explains the
mechanism, deck 4 gives the evidence. That is redundancy by design, not
duplication.

## Prioritizing

Every note carries a priority tag, orthogonal to its deck:

| Tag | Meaning | Notes |
|---|---|---:|
| `prio::1-core` | Know cold. Study first. | 248 |
| `prio::2-working` | Working knowledge. | 149 |
| `prio::3-depth` | Reference-level; learn when you need it. | 75 |

Three ways to use them, in increasing order of control:

**Filtered deck (simplest).** Tools → Create Filtered Deck, search
`deck:Medicaid tag:prio::1-core`. That gives you a 248-note fast track across
all six decks. Swap the tag as you finish a tier, or narrow further:
`deck:Medicaid tag:prio::1-core tag:financing`.

**Per-deck new-card limits.** Each deck has its own options preset. Setting
deck 6 to 15 new cards a day and deck 3 to 5 makes the series advance at
different rates without any filtering — the right approach if the evaluation
deck matters most for your work right now.

**Suspend the tail.** Browse → `tag:prio::3-depth` → suspend. Those 75 notes
stay in the collection, searchable, and unsuspend when you want them. This is
the cleanest way to shrink the daily load without deleting anything.

Topic tags work the same way. The useful ones:

```
financing  eligibility  benefits  ltss  managed-care  waivers  duals  drugs
oregon  mortality  children  labor  crowd-out  work-requirements  obbba
methods  synthesis  framework  economists  institutions  journals
```

## Suggested path

1. **Deck 1 straight through.** Cloze drills are fast and they make everything
   else cheaper to learn. Two weeks at 20 new a day.
2. **Deck 2, `prio::1-core` first.** The financing cards — FMAP, non-federal
   share, provider taxes, DSH/UPL/directed payments — carry the most weight and
   are the hardest to absorb by osmosis later.
3. **Deck 6 in parallel from day one**, since it is closest to working needs.
   Its default priority is 1 for that reason.
4. **Decks 3 and 4 together.** They interlock: a design card names a trade-off,
   a research card supplies the evidence on it.
5. **Deck 5 whenever.** It is recognition material and does not depend on the
   rest. Low cognitive load, useful early.

At 25 new cards a day the whole series takes about a month, with reviews
settling around 80–120 a day.

## Two things to know before you study

**Card structure.** Each card has a question, an answer, and a *source note* in
smaller type below the divider — the citation, the caveat, or the reason the
fact matters. It is not tested. Read it, don't drill it; much of the actual
understanding lives there.

**Date-sensitive figures.** Enrollment, spending, expansion state counts, and
everything touching the 2025 reconciliation law (H.R. 1) move. Those cards say
so and give an "as of" date. The law's provisions phase in between 2026 and
2032, so several cards describe rules that are scheduled but not yet operative.
Structural material — statutory architecture, waiver authorities, financing
mechanics, research findings, evaluation methods — is stable.

## Scope note on deck 6

Deck 6 reads "benefit evaluation" as **evaluating public benefit programs**:
impact and implementation evaluation of transfer and coverage programs. Medicaid
supplies most examples, but the material transfers to SNAP, TANF, and housing.
If your work is actually a narrower slice — 1115 demonstration evaluation
specifically, or state agency program evaluation, or academic health economics —
the deck can be re-cut to match; say which and it is a straightforward edit.

## Import

Anki → **File → Import** → `dist/Medicaid.apkg`. Six subdecks appear under a
parent `Medicaid` deck. Nothing else in your collection is touched: the series
ships its own note types (`Medicaid Basic`, `Medicaid Cloze`) with their own IDs.

Re-importing an updated `.apkg` **updates** existing notes rather than
duplicating them — note GUIDs derive from stable card IDs in the YAML, so
scheduling history survives a rebuild.

One caveat if you imported the earlier three-deck version: deck names and
numbering changed (`1. Fundamentals` → `2. Fundamentals`, and so on). Anki
matches decks by name, so the old decks will remain alongside the new ones.
Simplest fix is to delete the old `Medicaid` deck tree before importing —
review history for those 264 notes is lost, but nothing else is.

The `.tsv` files are a fallback for clients that choke on the package. They
carry `#deck`, `#notetype`, and `#tags` headers, so import handles them without
column mapping; the source note goes into the Back field there rather than its
own field.

## Editing and rebuilding

```yaml
- id: F001              # stable — changing it orphans review history
  priority: 1           # 1, 2, or 3; omit to use the file's default_priority
  type: basic           # or: cloze
  tags: [statute]
  front: |-
    What statute creates Medicaid, and when was it enacted?
  back: |-
    Title XIX of the Social Security Act ...
  extra: |-
    The same 1965 law created Medicare as Title XVIII ...
```

Cloze notes use `text:` with `{{c1::...}}` deletions instead of front/back.
Lines starting with `- ` become bullets; inline `<b>` and `<i>` pass through.
Each file's header sets its `deck`, a stable `deck_id`, and a
`default_priority`.

```bash
pip install genanki pyyaml
python build_deck.py
```

The build validates first: duplicate note IDs, duplicate or missing deck IDs,
missing fields, bad priority values, and cloze notes with no deletion all fail
the build rather than producing a broken package. It prints a per-deck priority
breakdown so you can see the tiers stay balanced as you add cards.

## Sources

Drawn from the statute and regulations, CMS and MACPAC material, CRS reports,
KFF, and the primary research literature, with current-figure and 2025–26 policy
items checked against published sources in August 2026.

Standing references:

- **MACPAC** — MACStats data book; the most citable single source for statistics
- **KFF** — State Health Facts; the annual Medicaid Budget Survey
- **CMS / Medicaid.gov** — waiver and SPA documents, expenditure reports
- **CRS** — R42640 (financing and expenditures), R48569 (H.R. 1 coverage provisions)
- **CBO** — baselines and cost estimates
- **42 U.S.C. ch. 7 subch. XIX** and **42 C.F.R. parts 430–460** for legal
  questions; secondary summaries of Medicaid age quickly

Deck 4 cites papers by author, year, and journal so each can be pulled directly.
Where a figure is approximate, that is deliberate — go to the paper before
quoting a number in writing.
