# Vocabulary position and trademark lifecycles

Code, public data panels, and the working paper for an empirical application of the DeDeo prospective/retrospective Kullback–Leibler resonance framework (Murdock, Allen & DeDeo 2017; Barron, Huang, Spang & DeDeo 2018) to US trademark goods/services text.

The framework was developed on cognitive corpora — Darwin's reading notebooks, French Revolution parliamentary debates — where the text is authored by the agent whose novelty is being measured. This work applies the same apparatus to commercially-incentivised legally-drafted text: granted USPTO filings 1990–2024, with the deepest diagnostics on the complete software/electronics class (1.81M filings) and a four-class build (software, tech-services, transport, advertising/retail; 2.32M granted filings) for the cross-industry work.

The construct, ΔKL, is a filing-level measure of commercial novelty that is cheap to compute, available for millions of firms that never patent, and empirically distinct from patent-track invention. Three validation results anchor it:

1. **ΔKL marks commercial risk at the mark level, robustly across scoring choices.** The construct is scored on topic distributions (the source papers' own operationalization; dense regardless of document length). Forward-leaning applications complete registration slightly less often (an inverse-U; completion mostly measures follow-through to commercial use), and among 2016–2018 registrations facing their first §8 maintenance gate, survival falls −5.5pp across ΔKL quintiles (n≈770k) — the same to within 0.2pp under term scoring and at topic resolutions T=50, 200, and 500. The event-dated estimate (full prosecution re-parse, 4.12M registrations 2002–2018) gives +3.9pp first-gate failure, emerging with the 2008–2010 cohorts and buffered by legal representation and recent proof of use.
2. **A methodological caution the construct surfaces.** Under term-level scoring, forward-leaning filings appear to belong to better firms (margins +2–3pp/σ, listing +40% relative); these associations dissolve or reverse under distribution scoring at every resolution tried (margin +2.3pp/σ token → −1.3 at T=50 → −2.2 at T=200 → −0.8 at T=500, never approaching the term result) and the listing gradient dissolves under document-length controls. The firm signal is unsigned lexical specificity, not signed resonance. Term-resolved text-novelty measures can manufacture firm-performance correlations from drafting style; the paper documents the hazard rather than anchoring on the firm-level claims.
3. **ΔKL is empirically distinct from patents.** Within firm, ΔKL on log(1+patents) gives −0.048σ (t=−7.9 on 14,470 matched firms), while patent counts are uncorrelated with ΔKL on the same panel.
4. **The construct supports direct measurement of cross-industry idea diffusion.** Software- and tech-services-origin themes (cloud, AI, as-a-service, blockchain, streaming) arrive in transport and advertising/retail with 1–13 year lags; theme adoption follows Bass curves (median q/p ≈ 6.5 on 118 fits); the cross-class flow is strongly asymmetric (software originates 33 of 72 origin→arrival edges and receives only 10).

A base-rate discipline result is also reported: the raw entrant share among earliest theme carriers (73.7%) sits slightly *below* the corpus debut-rate baseline (76.4%), and the themes that lean incumbent include AI, cloud, and sustainability — the canonically "new" themes are carried by incumbents diversifying in at least as much as by entrants.

The paper defends a bounded reading: ΔKL is a measurement of lexical resonance with interpretable empirical structure on commercial text, not an innovation measure in any strong sense.

## Start here

| File | Pages | Description |
|---|---|---|
| **`paper/ssrn_diffusion_paper.pdf`** | 20 | The working paper. **Read this first.** |
| `paper/newterms_report.pdf` | 12 | Companion: cross-industry vocabulary introduced after a 1990–1994 burn-in. Top 100 themes tabulated with adoption trajectories. |
| `paper/face_validation.md` | — | Rateable form for the 50 LDA themes (top words, per-class peak, representative marks). |
| `paper/_legacy/` | — | Earlier papers, preserved for reference. See `paper/_legacy/README.md`. |

Design documents:

- `METHOD.md` — the DeDeo prospective/retrospective KL method note.
- `PROPOSAL.md` — original proposal.
- `PROPOSAL_diffusion.md` — the business-model-diffusion program proposal.
- `PROJECT.md` — umbrella project framing.
- `BLOCKED.md` — remaining workstreams with decision rules.

## What's in this repo

```
.
├── paper/                 the working paper, companion, and analysis outputs
│   ├── ssrn_diffusion_paper.{tex,pdf}    the SSRN working paper
│   ├── newterms_report.{tex,pdf}         cross-industry new vocabulary 1995–2021
│   ├── face_validation.md                 50-LDA-theme rateable form
│   ├── results/                            JSON metrics, figures, tables, per-class outputs
│   └── _legacy/                            archived earlier papers
├── scripts/               analysis chain (see "Result lookup" below)
│   └── blocked/           runnable scripts still gated on external data
├── src/novelty/           the Python package
├── data_publish/          public data panels
│   ├── firm_year_dkl.csv               (CIK, year) mean ΔKL, n = 59,033
│   ├── firm_year_patents_and_dkl.csv   panel joined to PatentsView patents, n = 174,569
│   └── comparators/                    BCG / MIT TR50 / Crunchbase external lists + crosswalk
├── PROJECT.md, INBOX.md, BLOCKED.md, METHOD.md, PROPOSAL.md, PROPOSAL_diffusion.md
├── Makefile, Makefile.dynamism, pyproject.toml
└── LICENSE                GPL-3.0
```

The raw 12 GB USPTO TRTYRAP backfile is not committed and is regenerable from the USPTO API.

## Reproducing the analysis

Requires Python 3.11, a TeX install (TeX Live or MiKTeX), and a free USPTO Open Data Portal API key from <https://data.uspto.gov> (My ODP → My API Key).

```bash
# 1. environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install statsmodels matplotlib

# 2. API key
cp .env.example .env
# edit .env and add USPTO_ODP_API_KEY=<your key>

# 3. corpus (multi-hour; streams ~12 GB and writes a parquet per NICE class)
make tm

# 4. SEC financial panel (for the panels under data_publish/)
make sec
make crosswalk

# 5. analysis (the scripts under scripts/ build paper/results/)
make analysis
```

The current paper compiles via `pdflatex` directly:

```bash
cd paper
pdflatex ssrn_diffusion_paper.tex    # twice for citations
pdflatex ssrn_diffusion_paper.tex
```

## Result lookup

The empirical claims in the paper and their script → output paths:

| Claim | Script | Output |
|---|---|---|
| Debut outcomes, term-scored: registration (inverse-U in ΔKL) + EDGAR-given-registration (term-only; see topic battery row) | `scripts/debut_outcome_by_kl_v3.py` | `paper/results/debut_outcome_by_kl.{png,csv}`, `debut_outcome_metrics.json` |
| Event-dated first-gate failure, all cohorts/classes (+3.9pp; modern emergence; counsel/basis strata) | `scripts/events_full_build.py` + `event_gates_all.py` | `paper/results/event_gates_all.json`, `event_gate_{cohort_curve,forest}.png`; `data/processed/case_{events,extras}.parquet` (not committed) |
| Two-gate decomposition + latency telemetry + funded-flailing (class 009) | `scripts/hazard_pipeline_009.py` | `paper/results/hazard_009.json`, `two_gate_009.json` |
| First §8 gate survival, snapshot single cohort (−5.5pp; scoring-robust) | `scripts/s8_survival_corrected.py` | `paper/results/s8_corrected_{summary.{csv,json},forest.png,pooled.png}` |
| Registration inverse-U per industry (37/44 positive, 33 significant) | `scripts/registration_and_unconditional.py` | `paper/results/registration_by_industry.{csv,json}`, `registration_inverseU_forest.png` |
| Burn-in optimization (1990 bias ~2.0 nats; ≤0.04 from 1993; standard 1995 cut) | `scripts/burnin_optimization.py` | `paper/results/burnin_by_class.{csv,json,png}`, `burnin_convergence_examples.png` |
| Reference-window decomposition: gate penalty loads on past-rupture (−7.3pp) vs foresight (−3.2pp) | `scripts/run_full_corpus_w37.py` + `window_choice_all.py` (+ class-009 deep dive `window_choice_009.py`) | `paper/results/window_choice_all.{json,png}`, `window_choice_009.{json,png}` |
| AI vs internet era comparison (AI at year 9 ≈ internet at year 5; no turbulence spike) | `scripts/era_turbulence.py` | `paper/results/era_turbulence.{json,png}` |
| Unconditional composite outcome (appendix; inverse-U +1.4pp depth) | `scripts/registration_and_unconditional.py` | `paper/results/appendix_unconditional.{png,json}` |
| Quadrant figure regeneration with verified post-burn-in brand labels | `scripts/quadrant_regen.py` | `paper/results/quadrant.png`, `quadrant_labeled_points.csv` |
| Full-corpus topic scoring, the method (T=50; T=200/T=500 robustness) | `scripts/topic_p_scorer_all.py` (set TOPIC_T) | `data/processed/topic_surprise_class*{,_T200,_T500}.parquet` (not committed), `topic_lda_meta*.json`, `topic_model*.joblib` |
| Topic-count is a partition dial, not a coverage dial (crypto topic at T=200, absent at T=50/T=500) | `scripts/topic_p_scorer_all.py` | `data/processed/topic_lda_meta{,_T200,_T500}.json` |
| Encoding exhibit: Amazon/Kombucha/Ethereum, term vs topic at T=50/200/500 (Appendix B) | `scripts/exhibit_encodings.py` | `paper/results/exhibit_encodings.json` |
| Patent + BCG/MIT rescore under token/T=50/T=200 | `scripts/topic_patent_expert_rescore.py` | `paper/results/topic_patent_expert_rescore.json` |
| Vocabulary forensics: invisible-share split, signed-vs-unsigned firm signal | `scripts/vocab_forensics.py` | `paper/results/vocab_forensics.json` |
| Topic-vs-token outcome battery (mark-level reproduces; firm-level dissolves/reverses) | `scripts/topic_outcomes_all.py` + `topic_debut.py` + `topic_firm_margin.py` | `paper/results/topic_outcomes_all*.{json,png}`, `debut_outcome_topic*.{json,png}`, `topic_firm_margin*.json`, `topic_s8_forest.png` |
| Examiner bound + failure modes (4.3% adversarial; unconditional EDGAR test) | `scripts/examiner_confound.py` | `paper/results/examiner_confound.json` |
| Sub-year freshness, class 009 (months-late penalty −9.4pp; no look-ahead) | `scripts/subyear_window_009.py` | `paper/results/subyear_window_009.json` |
| 4-class topic-P pilot (length artifact, concordance) | `scripts/topic_p_scorer.py` | `paper/results/topic_p_validation.json` |
| All-45-class build (vocab + surprise per class) | `scripts/run_full_corpus.py` | `data/processed/surprise_class*.parquet` (not committed) |
| Gross margin +3.2pp/σ (t=5.3); pros/retr decomposition | `scripts/financials_regression.py` | `paper/results/financials_metrics.json` |
| Excess returns +1.5–5.0pp/σ over 1–4y; debut +28pp rejected | `scripts/returns_regression.py` + `wsE_returns_diagnostic.py` | `paper/results/returns_metrics.json`, `wsE_returns_diagnostic.json` |
| Expert lists +0.34σ (n=43, p=0.0003); patents uncorrelated | `scripts/wsD_validity_battery.py` | `paper/results/wsD_validity_battery.json` |
| ΔKL ⟂ patents within firm; −0.048σ, t = −7.9 | `scripts/wsC_within_firm_patents.py` | `paper/results/wsC_within_firm_patents.json` |
| Class-009 diagnostics: 0.97 collinearity, length artifact, sign robustness | `scripts/wsG_class009_analysis.py` | `paper/results/wsG_class009.json` |
| Phrase transit table (cloud, AI, blockchain, …) | `scripts/diff2_theme_transit.py` | `paper/results/diff2_transit.json` |
| LDA T = 50 themes | `scripts/diff3_lda_themes.py` | `paper/results/diff3_themes.json` |
| Bass fits (median q/p ≈ 7) + class-flow asymmetry | `scripts/diff4_phase1_rqs.py` | `paper/results/diff4_phase1.json` |
| Entrant share −2.7pp vs year-matched baseline | `scripts/diff5_rq4_baseline.py` | `paper/results/diff5_rq4_baseline.json` |
| NMF cross-method check; seed-phrase concordance 7/10 | `scripts/diff6_nmf_compare.py` | `paper/results/diff6_nmf.json` |
| Borrowed-novelty margin lift +1.4pp/σ (agenda item) | `scripts/diff8_rq6_borrowed_pays.py` | `paper/results/diff8_rq6.json` |
| New cross-industry vocabulary: 3,653 candidate terms | `scripts/newterms_analysis.py` + `newterms_build_pdf.py` | `paper/newterms_report.pdf` |

## Data not in this repo

The raw USPTO backfile (~12 GB), per-class records/vocab/surprise/outcome parquets (~15 GB after build), and SEC EDGAR FSDS ZIPs (~7 GB) are excluded by `.gitignore`. The reproduction pipeline regenerates them from public sources.

## Citing this work

Working-paper citation (the SSRN ID will be added when posted):

> Silver, E. (2026). *Vocabulary position and trademark lifecycles: An event-dated corpus and a lead/lag text measure for the commercial economy*. SSRN Working Paper.

## Author / contact

Eric Silver — `epsilver@gmail.com`.

Independent researcher; formerly a PhD candidate at Carnegie Mellon University. The author's current employment is unrelated to this work, and the views expressed are the author's alone.

## License

GPL-3.0 (see `LICENSE`).
