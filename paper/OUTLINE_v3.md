# v3 restructure — outline, asset map, and compute plan (2026-08-23)

Review copy frozen at tag `review-copy-2026-08-23` (`paper/frozen/`). Eric's line edits
arrive against that copy; the restructure below is drafted in `paper/v3/`.

Design rule from Eric: sections 2–4 should each stand nearly as independent papers.
Section 1 introduces context, approach and vocabulary. Section 5 is robustness. The
introduction is written last; then three abstracts (two not descended from the current
one); then a dozen candidate titles built on "linguistic tells and business themes
predict success, survival and thriving."

Writing rule: plain declarative findings, subject named, one clause per fact, no
figurative framing or asides (memory: feedback_no_figurative_framing).

## §1 Context, approach, vocabulary

Reuse: `section_construct.tex` (two readers; information content; counting words;
themes; KL; A and L; the Amazon table; neighbours subsection from main tex).
New: a paragraph distinguishing **themes** (LDA clusters of co-occurring goods/services
terms — what a filing draws on) from **topics** (the Darwin/FRev usage: the subject of
a document); we use "theme" for the fitted clusters and reserve "topic" for the model
family name only. State once: one global model, references per class.

## §2 Registration and the first use-proof gate, largest viable dataset

Dataset: all class-records 1995–2018 for registration; unique registrations 2002–2018
for the gate. Reuse: funnel table; registration inverse-U (app:registration);
gate LPM table; era/technology/theme swing (sec:eras + sec:techswing); self vs
counsel (69.6 vs 46.8; +1.8 vs +1.2); amendments (refile_text_change: 42,115 pairs,
conform toward class; app:refile); conditioning argument (old 4.1).
New: **internet as a theme** — web-based "engineering" companies vs their Nice class.
Use theme_cohorts.json internet pattern + per-class breakout; within-class contrast of
internet-bearing filings' registration and gate rates vs the class, 1995–2018, by
class; show divergence (009/035/042 vs goods classes). Script: `internet_breakout.py`.
End of section: why later sections filter to registrations that reached the first gate.

## §3 The year-five gate under three theme resolutions; new themes vs new-to-class

Compare on identical samples:
  (a) global 500 themes, references per class   — `rolling_rescore_all.py 500` RUNNING
  (b) 50 themes fitted per Nice class            — `perclass_lda_rescore.py` RUNNING
  (c) global 50, references per class (production)
Gate contrast, lead and atypicality, pooled and per class; correlation between scorings.
New themes: from T=500 assignment, a theme's corpus-wide first sustained year vs its
first sustained year in each class (theme_assign_T500 months-by-class). Classify each
registration's dominant theme as (i) wholly new to the corpus at filing, (ii) new to the
class but established elsewhere, (iii) established. Gate survival by group, within class
and year. Script: `theme_novelty_origin.py`.

## §4 SEC events: fundraising (Form D) and IPOs; value concentration; curated themes; patents

Reuse: `atypicality_and_funding.py` (Form D near-census 2009+), `funding_owner_match.parquet`,
`debut_edgar_substantiate.py` (EDGAR reporting), `dynamism_ipo_events.csv` (IPO year per
crosswalked owner), `funding_lag_prototype.py` (term → funding → IPO lags),
`theme_cohorts.py` (internet/AI/blockchain keyword cohorts), patent panel
(`patent_compare.py`, `patent_timing_by_industry.py`, `patent_complementarity_by_sector.py`).
New: (i) outcome ladder per debut owner: none → Form D round → EDGAR reporting → IPO;
lead/atypicality contrasts at each rung. (ii) "Most new business value": among
crosswalked owners, rank by market value at IPO or by revenue in sec_firm_year; share of
total listed revenue held by the top 50/100/500 trademark-debut firms; their vocabulary
position at debut. (iii) Hand-curated internet / blockchain / AI theme bundles
(theme_bundles.py) compared on registration, gate, funding, listing. (iv) Patents:
orthogonality (exists); funding→patent vs patent→funding timing on the Form D panel.
Script: `sec_event_ladder.py`, `value_concentration.py`.

## §5 Robustness

Reuse: app:uncond (no conditioning on grant); eras (time periods, 2002 cut vs carried
back; 2019 replication); app:rolling (annual vs per-filing); app:choices (weighting,
anchoring, resolution; seed); decay (`rolling_rescore_decay.py` outputs); window mix;
pair surprise; measurement hazard (term vs theme at firm level).
Rule: state the simpler formulation used in the body for legibility, then the more
complex finding beside it.

## Order of work

1. Compute (running): T=500 rescore all classes; per-class 50-theme rescore.
2. Draft §1, §2 (internet breakout script first), then §5 (mostly assembly).
3. §3 when compute lands; `theme_novelty_origin.py`.
4. §4 scripts, then draft.
5. Introduction ("pioneers are the ones with arrows in their backs"), three abstracts,
   twelve titles.
