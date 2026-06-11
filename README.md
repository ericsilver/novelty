# Vocabulary resonance in US trademark filings

Code, public data panels, and the working paper for an empirical application of the DeDeo prospective/retrospective Kullback–Leibler resonance framework (Murdock, Allen & DeDeo 2017; Barron, Huang, Spang & DeDeo 2018) to US trademark goods/services text.

The framework was developed on cognitive corpora — Darwin's reading notebooks, French Revolution parliamentary debates — where the text is authored by the agent whose novelty is being measured. This work applies the same apparatus to commercially-incentivised legally-drafted text: granted USPTO filings 1990–2024, with the deepest diagnostics on the complete software/electronics class (1.81M filings) and a four-class build (software, tech-services, transport, advertising/retail; 2.32M granted filings) for the cross-industry work.

The construct, ΔKL, is a firm-level measure of commercial novelty that is cheap to compute, available for millions of firms that never patent, and empirically distinct from patent-track invention. Three validation results anchor it:

1. **ΔKL marks commercial risk-taking, with opposite signs at the mark and firm levels.** Forward-leaning marks fare worse in the legal lifecycle: they register slightly less often, and among 2016–2018 registrations facing their first §8 maintenance gate, survival falls from 49.2% to 43.3% across ΔKL quintiles (n=765,154; negative in 31 of 41 industries). The firms that file them fare better: eventual public listing among first-time filers rises ~40% in relative terms across ΔKL quintiles; gross margin on an SEC-matched panel runs +3.2pp per σ (t=5.3); excess stock returns run +1.5 to +5.0pp per σ at 1–4 year horizons, decaying by year five. Prospective surprise alone predicts none of the firm-level outcomes — only the signed resonance does.
2. **ΔKL is empirically distinct from patents.** Within firm, ΔKL on log(1+patents) gives −0.048σ (t=−7.9 on 14,470 matched firms), while firms on BCG/MIT expert "most innovative" lists sit +0.34σ above the panel mean (p<0.001) and patent counts are uncorrelated with ΔKL on the same panel.
3. **The construct supports direct measurement of cross-industry idea diffusion.** Software- and tech-services-origin themes (cloud, AI, as-a-service, blockchain, streaming) arrive in transport and advertising/retail with 1–13 year lags; theme adoption follows Bass curves (median q/p ≈ 6.5 on 118 fits); the cross-class flow is strongly asymmetric (software originates 33 of 72 origin→arrival edges and receives only 10).

A base-rate discipline result is also reported: the raw entrant share among earliest theme carriers (73.7%) sits slightly *below* the corpus debut-rate baseline (76.4%), and the themes that lean incumbent include AI, cloud, and sustainability — the canonically "new" themes are carried by incumbents diversifying in at least as much as by entrants.

The paper defends a bounded reading: ΔKL is a measurement of lexical resonance with interpretable empirical structure on commercial text, not an innovation measure in any strong sense.

## Start here

| File | Pages | Description |
|---|---|---|
| **`paper/ssrn_diffusion_paper.pdf`** | 14 | The working paper. **Read this first.** |
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
| Debut outcomes: registration (inverse-U in ΔKL) + EDGAR-given-registration (+~40% relative) | `scripts/debut_outcome_by_kl_v3.py` | `paper/results/debut_outcome_by_kl.{png,csv}`, `debut_outcome_metrics.json` |
| First §8 gate survival, correct status codes (−5.9pp pooled; negative in 31/41 classes) | `scripts/s8_survival_corrected.py` | `paper/results/s8_corrected_{summary.{csv,json},forest.png,pooled.png}` |
| Registration inverse-U per industry (37/44 positive, 33 significant) | `scripts/registration_and_unconditional.py` | `paper/results/registration_by_industry.{csv,json}`, `registration_inverseU_forest.png` |
| Burn-in optimization (1990 bias ~2.0 nats; ≤0.04 from 1993; standard 1995 cut) | `scripts/burnin_optimization.py` | `paper/results/burnin_by_class.{csv,json,png}`, `burnin_convergence_examples.png` |
| Reference-window decomposition: gate penalty loads on past-rupture (−7.3pp) vs foresight (−3.2pp) | `scripts/run_full_corpus_w37.py` + `window_choice_all.py` (+ class-009 deep dive `window_choice_009.py`) | `paper/results/window_choice_all.{json,png}`, `window_choice_009.{json,png}` |
| AI vs internet era comparison (AI at year 9 ≈ internet at year 5; no turbulence spike) | `scripts/era_turbulence.py` | `paper/results/era_turbulence.{json,png}` |
| Unconditional composite outcome (appendix; inverse-U +1.4pp depth) | `scripts/registration_and_unconditional.py` | `paper/results/appendix_unconditional.{png,json}` |
| Quadrant figure regeneration with verified post-burn-in brand labels | `scripts/quadrant_regen.py` | `paper/results/quadrant.png`, `quadrant_labeled_points.csv` |
| Topic-distribution P robustness (length artifact flattens; concordance 0.46/0.36) | `scripts/topic_p_scorer.py` | `paper/results/topic_p_validation.json` |
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

> Silver, E. (2026). *Vocabulary resonance in US trademark filings: A measurement construct for commercial novelty, with validation and applications*. SSRN Working Paper.

## Author / contact

Eric Silver — `epsilver@gmail.com`.

Independent researcher; formerly a PhD candidate at Carnegie Mellon University. The author's current employment is unrelated to this work, and the views expressed are the author's alone.

## License

GPL-3.0 (see `LICENSE`).
