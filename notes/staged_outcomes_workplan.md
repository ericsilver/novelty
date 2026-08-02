# Staged outcomes table + column rename — work in flight (2026-08-02)

## The finding that motivated all of this

The two axes do different work at different stages, and the *gap between stages* is
the interesting part:

| Stage | Atypicality | Forward lean |
|---|---|---|
| Raising a Reg D round | **null / mildly negative** (Q5 −0.042pp, t = −1.1) | **−0.526pp, t = −12.7** (−20.7% relative) |
| Listing, given funding | **+0.716pp, t = +7.1** (top quintile +67% relative) | −0.284, t = −3.4 |

Eric's read, which the data supports: venture investors do **not** price atypicality at
entry, but atypical firms go public more often once funded. That is a candidate
**venture-market inefficiency** — under-valuation of atypical language at the funding
stage — and it is the opposite of his prior (that novelty is over-rewarded).

Note his other hypothesis is NOT supported: forward lean (being early) is punished at
*both* funding and listing, so this is not "first movers get funded but don't exit."
Forward lean loses everywhere — registration, the Section 8 gate, funding, and listing.
Four independent gates, same sign.

Replication check on the paper's published spec: unconditional listing on the 2009–2018
cohort gives abs_zd +0.0217pp (t = +4.1), z_d −0.0338pp (t = −9.8) — same signs and
comparable magnitudes to published spec C (t = 3.79, t = −5.01).

## Column rename — DONE

`scripts/migrate_kl_column_names.py`, run 2026-08-02: **271 parquet files rewritten,
0 failures, 0 legacy names remaining, no temp files left.**

    prospective_kl      -> kl_vs_past
    retrospective_kl    -> kl_vs_future
    n_ref_prospective   -> n_ref_past
    n_ref_retrospective -> n_ref_future
    topic_pros          -> topic_kl_vs_past
    topic_retr          -> topic_kl_vs_future
    topic_dkl           unchanged (the paper's named construct)

**Verified against the scorer code before renaming** — this was NOT a guess:
`src/novelty/surprise.py` (~L91–107) aggregates `log_pros` over years [y−W, y−1] = PAST
and `log_retr` over [y+1, y+W] = FUTURE; `scripts/topic_p_scorer_all.py` (~L143–167)
sets `q_p` = past, `q_f` = future, `pros = KL(P‖q_p)`, `retr = KL(P‖q_f)`. So the old
names followed the Murdock/Barron convention (prospective = measured against the past),
which is correct in the source literature and matches the paper's own definitions — but
reads backwards to anyone meeting the column cold. Sign of ΔKL is UNCHANGED:
ΔKL = vs_past − vs_future, positive = the field moved toward the filing.

## In flight

1. **Verification agent** — updating every consumer of the renamed columns and doing a
   *semantic* audit (is a variable called "pros" actually holding the vs-past quantity;
   is every ΔKL constructed in the right order; do plot labels mislead). Also adding a
   README subsection documenting the columns. Told NOT to touch paper/*.tex.
2. **Funding agent (resumed)** — persisting `data/processed/funding_owner_match.parquet`
   (owner → CIK, first Form D date, n notices, first amount, in_sec/in_fsds/in_8a, IPO
   date) so the staged table can reuse its matching rather than re-deriving it. Told to
   use new column names and to write a NEW file (`scripts/persist_funding_match.py`) so
   it does not collide with the verification agent.
3. **`scripts/staged_outcomes_table.py`** — running. Skips the funding stages until (2)
   lands, then needs a re-run.

## The table being built

One specification at every stage so the coefficients read down a column:
`outcome ~ z_level + z_lean + log_len`, class × debut-year FE, HC1 robust SE,
coefficients in percentage points per SD. Unit is the **debut owner**, so one firm is
one row and the funnel is a funnel.

Stages: registered | filed; passed first S8 gate | registered; funded | filed;
funded | registered (**Eric's granted-only question**); listed | funded;
listed | registered (the published comparison).

Cohort windows necessarily differ — the S8 gate needs elapsed time, Form D is
electronic only from 2009 — so rows are NOT a nested decomposition, and the caption
says so.

## Reproducibility bug found and fixed (2026-08-02) — affects the headline

The verification agent noticed `atypicality_and_funding.py` produced different
quintile coefficients on two identical runs. Traced to `rank("ordinal").over(cell)`,
which breaks ties by frame row order — not stable across runs in polars.

**This is not cosmetic.** Ties are common: in class 009, **26.8% of scored filings
share a `topic_dkl` value with another filing**, and the largest tie groups run to
16,137 / 14,901 / 11,166 filings. Boilerplate goods/services text yields identical
topic distributions. With tie groups that size, quintile cut points move materially
between runs.

Three scripts used the pattern, and two of them feed published numbers:
- `scripts/gate_decisive_regression.py` — **the paper's headline** (+2.3pp/quintile)
- `scripts/debut_edgar_substantiate.py` — the listing/EDGAR table
- `scripts/atypicality_and_funding.py` — the new funding work

All three now sort on `[cell, score, unique_key]` before ranking, which pins the
tie-break. Re-running `gate_decisive_regression.py` to see whether the published
headline moves; prior JSON backed up to `paper/results/.gate_before.bak`.

Deeper issue worth a decision before submission: when 16,000 filings share one
score, splitting them across a quintile boundary is arbitrary regardless of
determinism. `rank("min")`/`rank("average")` would keep tied filings together at
the cost of unequal quintiles. The deterministic fix makes the current estimand
reproducible; it does not make ties principled.

## Still to do

- Re-run the staged table once `funding_owner_match.parquet` exists.
- Add the subsection + table to `paper/ssrn_diffusion_paper.tex`; rebuild; open.
- Commit (remember: `notes/` is public — this file is fine, it quotes no correspondence).
