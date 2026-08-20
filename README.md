# Vocabulary position and trademark lifecycles

This repository builds an event-dated corpus of all 13.99 million USPTO trademark case files and defines a filing-level text measure on it. Each filing's goods/services description is scored by how surprising its wording is to two readers with disjoint knowledge — one who has seen only what the filing's Nice class filed in the 1,826 days *before* it, one who has seen only what it filed in the 1,826 days *after* — and the resulting pair is rotated into an unsigned axis, **atypicality**, and a signed axis, **lead**. Because the corpus also carries every prosecution event with its date, a filing's language can be followed to what actually happened to the mark: registration, the five-to-six-year use-proof gate, the year-ten renewal, and the owner's appearance in the SEC filing universe.

The working paper is **`paper/ssrn_diffusion_paper.tex`** (PDF alongside). Read its abstract and Section 3 before the code. `paper/journal_paper.tex` is a superseded June 2026 sibling of the same project, kept for reference only — it does not describe the current measure, sample, or results.

## The measure in one screen

For filing *i* made on date *d*, with topic distribution *P<sub>i</sub>*:

| Quantity | Definition | Column | Paper's name | Murdock/Barron name |
|---|---|---|---|---|
| Surprise against the past | KL(*P<sub>i</sub>* ‖ *Q*<sub>past</sub>), *Q* pooled over [*d*−1826, *d*) | `topic_kl_vs_past` | past-facing surprise | novelty |
| Surprise against the future | KL(*P<sub>i</sub>* ‖ *Q*<sub>future</sub>), *Q* pooled over (*d*, *d*+1826] | `topic_kl_vs_future` | future-facing surprise | transience |
| Average | *A* = ½(*K*⁻ + *K*⁺) | *formed downstream* | **atypicality** | — |
| Signed difference | *L* = *K*⁻ − *K*⁺ | `topic_dkl` | **lead** (leading / lagging) | resonance |

Notes that save a re-derivation:

- **Both windows are anchored on the filing's own date**, not on its calendar year. No two filings made on different days share a reference; filings made on the same day are excluded from each other's windows.
- Within a window every day counts the same — a flat pooled aggregate, not a decayed one. The measure asks whether wording is unusual against a *period*, not whether it is ahead of a trend within that period.
- The names describe the *reader's vantage*, not the direction of the window. Prospective surprise is surprise measured against what came *before*.
- **Positive lead means the class moved toward the filing**: unusual when filed, ordinary five years later. Negative lead is the reverse.
- *A* and *L* are a 45-degree rotation of (*K*⁻, *K*⁺). The rotation matters because the raw pair correlates at 0.979, so entering both is close to entering one variable twice; *L* carries about 1.4% of their combined variance and is correspondingly fragile.
- Everything is estimated **within class and year**. Across classes, atypicality levels are not comparable — a class where the USPTO ID Manual supplies dense standard language has a compressed distribution for reasons unrelated to innovation.

## The `SURPRISE_SRC` switch

Two scorings of the whole corpus exist. An environment variable selects between them, and nine analysis scripts honour it:

| `SURPRISE_SRC` | Reference | Files | Status |
|---|---|---|---|
| `rolling` | Per-filing, ±1,826 days from the filing's own date | `data/processed/rolling_surprise_class{NNN}[_T200].parquet` | **Current.** Every estimate in the paper. |
| `topic` | One object per class-year: the class averaged over the five preceding *calendar* years | `data/processed/topic_surprise_class{NNN}[_T200].parquet` | **Retired.** Kept so the appendix can quantify what it cost. |

Both write identical column names, so switching is a path change and nothing else:

```bash
SURPRISE_SRC=rolling python scripts/gate_decisive_regression.py
```

Scripts honouring it: `gate_decisive_regression.py`, `event_gates_all.py`, `event_gates_2019_2021.py`, `two_gate_009.py`, `staged_outcomes_table.py`, `debut_edgar_substantiate.py`, `topic_debut.py`, `topic_outcomes_all.py`, `online_appendix.py`. The default is `rolling` in all of them, so the paper's numbers reproduce without setting it; `SURPRISE_SRC=topic` selects the retired scoring for the appendix comparison.

What the retired scoring cost, in short: annual bucketing imprints a spurious gradient in filing month (a December filing sits eleven months further from its past reference and eleven months nearer its future one), which inflated the gate penalty by about a third and manufactured a curvature at registration on the signed axis that does not survive. Signs and conclusions are unchanged; magnitudes and one functional form are not. Appendix "What annual reference buckets cost" has the full accounting.

## Data pipeline

Nothing below the first stage is committed; the whole chain regenerates from public sources.

**1. Bulk XML → per-class records.** `scripts/download_all_classes.py` streams the USPTO TRTYRAP backfile (83 archives, 1884–2025, ~12 GB) once and writes a slim parquet per Nice class. A filing declaring several classes is written into each of their parquets.
→ `data/processed/tm_class{NNN}.parquet`

**2. Bulk XML → prosecution events.** `scripts/events_full_build.py` re-parses the same backfile for what the published research files drop: the dated sequence of events behind each case's current status. 242 million events across 13.99 million case files.
→ `case_events.parquet` (serial, code, type, date, seq), `case_extras.parquet` (counsel of record, statutory filing basis, postregistration declaration flags, abandonment/status dates), `event_code_dict.parquet` (760 codes → modal description)

**3. Records → term-level scores.** `scripts/process_all_classes.py` drives `novelty.dictionary`, `novelty.surprise`, `novelty.firm_year`, `novelty.survival` per class. Token distributions (unigram + bigram, within-class document frequency ≥ 50) against class-year windows with Dirichlet smoothing.
→ `vocab_class{NNN}.parquet`, `surprise_class{NNN}.parquet`, `firm_year_class{NNN}.parquet`, `outcomes_class{NNN}.parquet`

**4. Records → topic-level scores.** `scripts/topic_p_scorer_all.py` fits LDA (*T* = 50 by default; set `TOPIC_T` for the 200/500 sweep) on a stratified 448,437-filing sample across all 45 classes, then transforms and scores every filing against *class-year* references. This is the retired scoring.
→ `topic_surprise_class{NNN}[_T{T}].parquet`, `topic_lda_meta*.json`, `topic_model*.joblib`

**5. Rescore on per-filing windows.** `scripts/rolling_rescore_all.py` reuses the fitted LDA model from stage 4 and rescores every class against references anchored on each filing's own date. Then `scripts/rolling_add_year.py` backfills the `year` column the rescorer does not write. **Run these in that order** — two downstream scripts filter on `year` and will raise without it.
→ `rolling_surprise_class{NNN}[_T200].parquet`

**6. External universes.** `scripts/download_sec_fsds.py` + `scripts/sec_extract.py` build the SEC financial-statement panel; `scripts/sec_link.py` resolves USPTO owner names to CIKs. `scripts/persist_funding_match.py` / `funding_lag_prototype.py` resolve Regulation D (Form D) issuers.
→ `sec_firm_year.parquet`, `uspto_sec_crosswalk.parquet`, `funding_owner_match.parquet`

**7. Analysis.** The scripts in the table below read the above and write JSON, figures, and `.tex` fragments into `paper/results/`, plus the online appendix into `docs/online-appendix/`.

## Reproducing

Requires Python 3.11, a TeX install (TeX Live or MiKTeX), and a free USPTO Open Data Portal API key (<https://data.uspto.gov>, My ODP → My API Key; or <https://account.uspto.gov/api-manager/>).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install statsmodels matplotlib rapidfuzz

cp .env.example .env          # then add USPTO_ODP_API_KEY=<your key>

make tm                       # stages 1 and 3: ~12 GB streamed once, multi-hour
make sec crosswalk            # stage 6
```

`make tm` is the corpus build and is the target to trust. **The `analysis` and `paper` targets in the `Makefile` are stale** — they build `paper/main.pdf` from a `paper/main.tex` that no longer exists, and their script list predates the current paper. Run the analysis scripts from the table below directly, and compile with:

```bash
cd paper && pdflatex ssrn_diffusion_paper.tex && pdflatex ssrn_diffusion_paper.tex
```

Cost worth knowing before you start:

- The backfile download is ~12 GB streamed once; per-class records/vocab/surprise/outcome parquets come to ~15 GB after the build; SEC FSDS ZIPs another ~7 GB.
- Stages 4 and 5 are **roughly an hour per topic resolution** on the full corpus (33 min at *T* = 50, 58 min at *T* = 200 on the reference build). A full rescore is not a casual rerun.
- Most analysis scripts run in minutes once the parquets exist; `online_appendix.py` and `staged_outcomes_table.py` are the slow ones.

## Where each paper exhibit comes from

Run everything in this table with `SURPRISE_SRC=rolling` unless the row says otherwise. Tables typed into the `.tex` by hand are marked; the rest are generated files the paper reads directly.

| Paper exhibit | Script | Artifact |
|---|---|---|
| **Table 1**, the funnel | `topic_debut.py`, `gate_decisive_regression.py`, `event_gates_all.py`, `examiner_confound.py` | `debut_outcome_topic.json`, `gate_decisive_regression.json`, `event_gates_all.json`, `examiner_confound.json` (typed) |
| **Table 2**, the five quantities and their names | — | prose only |
| **Figure 1**, debut outcomes (registration; listing given registration) | `topic_debut.py` | `debut_outcome_topic.{png,json}` |
| **Figure 2**, quintile profiles of all three outcomes | `quintile_profiles.py` | `quintile_profiles.{png,pdf}` |
| **Table 3**, first-gate failure LPM — the central result | `gate_decisive_regression.py` | `gate_decisive_regression.json` (typed) |
| **Figure 3**, per-industry gate forest | `event_gates_all.py` | `event_gate_forest.png` |
| **Figure 4**, gate penalty by cohort incl. the 2019 replication | `event_gates_2019_2021.py` | `event_gate_cohort_curve_extended.png`, `event_gates_2019_2021.json` |
| Two-gate attenuation, class 009 | `two_gate_009.py` | `two_gate_009.json` |
| **Table 4**, debut listing LPM and the *A*/*L* decomposition | `debut_edgar_substantiate.py` | `debut_edgar_substantiate.json` (typed) |
| **Table 5**, within-firm ΔKL vs patenting | `wsC_within_firm_patents.py` | `wsC_within_firm_patents.json` (typed) |
| Complement/substitute sector split of the patent null | `patent_complementarity_by_sector.py` | `patent_complementarity_by_sector.json` |
| **Table 6**, every outcome under one specification | `staged_outcomes_table.py` | `staged_outcomes_table.tex` (input directly), `staged_outcomes.json` |
| Form D / listing-after-funding owner match behind Table 6 | `persist_funding_match.py`, `atypicality_and_funding.py` | `funding_owner_match.parquet`, `atypicality_and_funding.json` |
| **Table 7**, phrase transit across classes | `diff2_theme_transit.py` | `diff2_transit.json` (typed) |
| LDA themes, Bass fits, class-flow asymmetry, entrant base rate, NMF check | `diff3_lda_themes.py`, `diff4_phase1_rqs.py`, `diff5_rq4_baseline.py`, `diff6_nmf_compare.py` | `diff3_themes.json`, `diff4_phase1.json`, `diff5_rq4_baseline.json`, `diff6_nmf.json` |
| **Figure 5**, era turbulence and era-term prevalence | `era_turbulence.py` | `era_turbulence.{png,json}` |
| **Figure 6**, burn-in density | `analysis_full.py` (the figure), `burnin_optimization.py` (the bias profile behind the 1995 cut) | `burnin.png`, `burnin_by_class.{csv,json,png}` |
| **Figure 7**, topic vs term on identical samples | `topic_outcomes_all.py` | `topic_outcomes_all.{png,json}` |
| Appendix A.5, what annual buckets cost | `rolling_window_rescore.py`, `rolling_window_gate_test.py` | `rolling_window_comparison.json`, `rolling_window_gate_test.json` |
| **Figure 8**, reference-window decomposition | `run_full_corpus_w37.py` + `window_choice_all.py` (class-009 deep dive: `window_choice_009.py`) | `window_choice_all.{png,json}` |
| Sub-year freshness variant, class 009 | `subyear_window_009.py` | `subyear_window_009.json` |
| Appendix B, the measurement hazard | `vocab_forensics.py`, `topic_outcomes_all.py`, `topic_firm_margin.py`, `financials_regression.py`, `returns_regression.py` | `vocab_forensics.json`, `topic_firm_margin{,_T200,_T500}.json`, `financials_metrics.json`, `returns_metrics.json` |
| **Figure 9**, the unconditional composite (Appendix C) | `registration_and_unconditional.py` | `appendix_unconditional.{png,json}` |
| **Figure 10**, length surface by representation (Appendix D) | `representation_appendix.py` | `representation_appendix.png` |
| **Table 8**, the three encoding exhibits (Appendix E) | `exhibit_encodings.py` | `exhibit_encodings.json` (typed) |
| Topic count is a partition dial, not a coverage dial | `topic_p_scorer_all.py` (set `TOPIC_T`) | `topic_lda_meta{,_T200,_T500}.json` |
| Snapshot single-cohort gate cross-check | `s8_survival_corrected.py` | `s8_corrected_summary.{csv,json}`, `s8_corrected_{forest,pooled}.png` |
| Online appendix, all 45 classes | `online_appendix.py` | `docs/online-appendix/` |

## Online appendix

**`docs/online-appendix/`** — published alongside the paper, and the place to look for anything class-level.

- `index.md` — a three-panel breakout for each Nice class (first-gate failure by lead quintile under cohort FE; registration completion on both axes; the class placed in the cross-industry distribution), plus the cross-industry forest and the size/base-rate scatters.
- `figures/class_{NNN}.png`, `figures/cross_{forest,scatter}.png`
- `per_class_estimates.csv` — machine-readable: scored filings, registrations, base failure rate, raw and fixed-effects gate contrasts with standard errors, completion at both tails of each axis.

Rebuild with `python scripts/online_appendix.py` (defaults to rolling scoring). It reuses the raw per-class lifts from `paper/results/event_gates_all.json` rather than recomputing them, so the appendix and the paper's forest figure cannot drift apart.

## Known limitations

Read these before reusing anything here.

**The SEC crosswalk is normalized-exact-match only.** `sec_link.py` contains a `rapidfuzz` pass, but the committed crosswalk has **19,889 owner strings and zero fuzzy matches** — every row is `match_type == "exact"`. Match rates therefore favour formally constituted entities with stable legal names, and every listing and financing estimate is conditional on matchability. The owner universe is every owner in all 45 class files (rebuilt 2026-08-07; it previously came from five `firm_year_class*.parquet` tables, which made 57% of the debut panel unmatchable and made matchability itself a function of the text being scored). Normalized names resolving to more than one registrant are dropped rather than assigned, and keys under three characters are excluded. Note also that only about half the matched CIKs carry an exchange ticker — the outcome is SEC *reporting*, not exchange listing. Listing rates in the panels are a lower bound on true listing; the estimands are within-cell contrasts, which a uniform match rate leaves unbiased, but the levels should not be read as population rates.

**Two topic-scored analyses remain on annual scoring.** The reference-window (*W* ∈ {3,5,7}) decomposition, which needs a separate scoring on each side, and the response-latency split in the software class. No per-filing-window version of either exists. By the evidence of the rolling/annual comparison their magnitudes are likely inflated by something like a third, so read them for the contrast they draw and not for their levels.

**Term-level scores have no rolling equivalent.** `surprise_class*.parquet` is scored against class-year windows and always will be under the current build; there is no `rolling_surprise` counterpart on the token side. Anything term-scored — the worked examples, phrase transit, the freshness measure, the era-turbulence series, the comparison arm of the topic/term battery — is therefore on the older reference construction by necessity, not by choice.

**Term scoring and document length are close to the same measurement.** Term-scored atypicality correlates with log filing length at −0.729 in class 009 and −0.688 in class 035; on descriptions a few dozen words long, "unusual" and "brief" are nearly interchangeable. Topic scoring takes those to −0.251 and −0.125. This is why the paper reports topic scoring for every estimate and term scoring for every worked example, and why the firm-level correlations that survive term scoring dissolve or reverse under distributions. Do not build firm-performance claims on term-level ΔKL.

**Topic scoring returns a finite value for only about 65% of filings**, so the topic and term columns are not the same sample. Comparisons across the two representations should be re-estimated on the common subset.

**Ties are rare under per-filing windows.** About 2.7% of scored filings share a ΔKL value with another filing, and the largest tie group runs to fourteen. Under class-year references, where boilerplate scored against a shared reference produced identical values in bulk, ties were an order of magnitude commoner. Quintile cuts are made deterministic by sorting on the score and then on the serial number. Splitting a tie group across a boundary is still arbitrary; `TIE_RULE=min` in `gate_decisive_regression.py` re-estimates with tied filings kept together, at the cost of unequal quintiles.

**Filings are not independent draws.** A fifth to a third share an exact position with another filing, and outcomes cluster within owner (within-owner correlation 0.42 for gate survival, mechanically 1.0 for listing). Gate regressions cluster on normalized owner; firm-level specifications carry one row per owner. Binning filings without that correction overstates how much structure the corpus contains.

**Registration is selected on language.** Correlations computed over all filings mix what examination does to a description with what the market does to the product. The main results condition on grant for that reason; the unconditional version is reported separately and diverges.

**The measure has no momentum channel.** References are flat pooled aggregates, so a term the class adopted five years ago and one it adopted last year are equally familiar to the past-facing reader. And the topic basis is fitted on the pooled corpus, so a look-ahead channel remains open; a vintage refit is owed. The months-scale freshness variant is already look-ahead-free but has only been run in one class.

**Legacy column names.** `src/novelty/surprise_decay.py` emits `kl_vs_past` / `kl_vs_future` alongside Kish effective sizes still called `n_eff_prospective` / `n_eff_retrospective`; `n_eff_prospective` pairs with `kl_vs_past`. `scripts/recompute_h2.py` maps them onto `n_ref_past` / `n_ref_future`. `scripts/migrate_kl_column_names.py` rewrites older panels (`prospective_kl`, `retrospective_kl`, `n_ref_prospective`, `n_ref_retrospective`, `topic_pros`, `topic_retr`) into the current names; it is idempotent and does not touch `topic_dkl`.

**`scripts/rolling_window_gate_test.py` reads a stale path.** It expects a `roll_dkl` column in `rolling_surprise_class009_T200.parquet`, a pairing that belonged to an earlier version of `rolling_window_rescore.py`. As the tree stands, `rolling_surprise_*` carries production column names and the `roll_*` schema lives in `rolling_diag_*`. Its committed JSON predates the split; re-running it as written will raise.

## Repository layout

```
.
├── paper/
│   ├── ssrn_diffusion_paper.{tex,pdf}   the working paper  ← start here
│   ├── section_construct.tex             Section 3, kept separate so it can be revised alone
│   ├── journal_paper.{tex,pdf}           SUPERSEDED June 2026 sibling
│   ├── newterms_report.{tex,pdf}         companion: new cross-industry vocabulary
│   ├── face_validation.md                rateable form for the 50 LDA themes
│   ├── results/                          JSON metrics, figures, .tex fragments
│   └── _legacy/                          archived earlier papers
├── docs/online-appendix/  per-class results published with the paper
├── scripts/               the analysis chain (see the exhibit table above)
│   └── blocked/           runnable scripts still gated on external data
├── src/novelty/           the Python package: dictionary, surprise, firm_year, survival
├── data_publish/
│   ├── firm_year_dkl.csv                (CIK, year) mean ΔKL, n = 59,033
│   ├── firm_year_patents_and_dkl.csv    joined to PatentsView, n = 174,569
│   └── comparators/                     BCG / MIT TR50 / Crunchbase lists + crosswalk
├── METHOD.md, PROJECT.md, PROPOSAL.md, PROPOSAL_diffusion.md, BLOCKED.md, INBOX.md
├── Makefile, Makefile.dynamism, pyproject.toml
└── LICENSE                GPL-3.0
```

Design documents: `METHOD.md` (the prospective/retrospective KL method note), `PROPOSAL.md` and `PROPOSAL_diffusion.md` (original and diffusion-program proposals), `PROJECT.md` (umbrella framing), `BLOCKED.md` (remaining workstreams with decision rules).

## Citing

> Silver, E. (2026). *Vocabulary position and trademark lifecycles: An event-dated corpus and a lead/lag text measure for the commercial economy*. SSRN Working Paper. https://ssrn.com/abstract=6954598 (DOI: 10.2139/ssrn.6954598)

## Author

Eric Silver — `epsilver@gmail.com`. Independent researcher; formerly a PhD candidate at Carnegie Mellon University. The author's current employment is unrelated to this work, and the views expressed are the author's alone.

## License

GPL-3.0 (see `LICENSE`).
