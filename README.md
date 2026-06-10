# Vocabulary resonance in US trademark filings

Code, public data panels, and the working paper for an empirical application of the DeDeo prospective/retrospective Kullback–Leibler resonance framework (Murdock, Allen & DeDeo 2017; Barron, Huang, Spang & DeDeo 2018) to US trademark goods/services text.

The framework was developed on cognitive corpora — Darwin's reading notebooks, French Revolution parliamentary debates — where the text is authored by the agent whose novelty is being measured. This work applies the same apparatus to a qualitatively different corpus type: commercially-incentivised legally-drafted text from 2.32 million granted USPTO filings (1990–2024) across four NICE classes (software, tech-services, transport, advertising/retail), and asks whether the framework produces interpretable empirical structure on text of this kind.

It does. Three patterns are documented:

1. **Cross-industry vocabulary diffusion is directly traceable** at the phrase level. Software- and tech-services-origin themes (cloud, AI, mobile-app, as-a-service, blockchain, streaming) arrive in transport and advertising/retail with multi-year lags of 1–13 years; the underlying panel structure passes a shuffle null on (class, year) label permutations at z = +46 under LDA topic extraction and replicates at z = +60.7 under independently-fitted NMF on the same vocabulary.
2. **The signal predicts post-registration §8 mark maintenance in a U-shape.** At the filing level, the lift at the tails vs the middle is 5–10 percentage points; at the token level, marks containing distinctive-tail tokens (typesetting, dial-up, blockchain, NFT) survive at 50–80% while marks containing flux-neutral middle tokens (organisational, conformity, freighting) survive at 11–26%.
3. **The signal is empirically distinct from patents.** Within-firm, ΔKL on log(1 + patents) gives a coefficient of −0.048σ (t = −7.9 on 14,463 trademark-and-patent-matched firms). The construct is not a patent proxy.

A worked discipline result is also reported: the raw Schumpeter Mark I read of new themes (73.7% entrant-dominated) collapses on year-matched baseline contrast to a −2.7pp mean excess once the corpus's 76.4% overall debut rate is netted out.

The paper defends a bounded reading: ΔKL is a measurement of lexical resonance with interpretable empirical structure on commercial text, not an innovation measure in any strong sense.

## Start here

| File | Pages | Description |
|---|---|---|
| **`paper/ssrn_diffusion_paper.pdf`** | 12 | The working paper. **Read this first.** |
| `paper/newterms_report.pdf` | 12 | Companion: cross-industry vocabulary identified as introduced after a 1990–1994 burn-in. Top 100 themes tabulated with adoption trajectories. |
| `paper/face_validation.md` | — | Rateable form for the 50 LDA themes (top words, per-class peak, representative marks). |
| `paper/_legacy/` | — | Earlier papers (main.pdf, short.pdf, dynamism, ethnic_clusters_note, construct_validity_note, diffusion_phase0_note, integrated_report). Preserved for reference. See `paper/_legacy/README.md`. |

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
| ΔKL ⟂ patents within firm; −0.048σ, t = −7.9 | `scripts/wsC_within_firm_patents.py` | `paper/results/wsC_within_firm_patents.json` |
| Class-009 diagnostics: 0.97 collinearity, length artifact, sign robustness 6.9–10.4% | `scripts/wsG_class009_analysis.py` | `paper/results/wsG_class009.json` |
| Within-class term provenance: corr(novel_share, ΔKL) = +0.29 | `scripts/wsH_term_provenance009.py` | `paper/results/wsH_provenance009.json` |
| Phrase transit table (cloud, AI, blockchain, …) | `scripts/diff2_theme_transit.py` | `paper/results/diff2_transit.json` |
| LDA T = 50 + kill condition z = +46 | `scripts/diff3_lda_themes.py` | `paper/results/diff3_themes.json` |
| Bass fits (median q/p ≈ 7) + class-flow asymmetry | `scripts/diff4_phase1_rqs.py` | `paper/results/diff4_phase1.json` |
| Schumpeter I collapse: −2.7pp excess vs baseline | `scripts/diff5_rq4_baseline.py` | `paper/results/diff5_rq4_baseline.json` |
| NMF cross-method check: z = +60.7; seed-phrase concordance 7/10 | `scripts/diff6_nmf_compare.py` | `paper/results/diff6_nmf.json` |
| Face-validation form (50 LDA topics) | `scripts/diff7_face_validation_doc.py` | `paper/face_validation.md` |
| New cross-industry vocabulary: 3,653 candidate terms 1995–2021 | `scripts/newterms_analysis.py` + `newterms_build_pdf.py` | `paper/results/newterms_top100.csv` + `paper/newterms_report.pdf` |

The following analyses are run and outputs preserved in `paper/results/` but are not surfaced in the current paper (they relate to scope statements rather than headline claims):

- `wsD_validity_battery.py` — small-n discriminant against BCG/MIT lists (n = 48 matched firms).
- `wsE_returns_diagnostic.py` — identifies the prior +28pp/σ four-year debut excess return as a right-tail artifact.
- `diff8_rq6_borrowed_pays.py` — within-firm borrowed-novelty contemporaneous margin lift (+1.4pp, t = +2.1, transient).
- `diff9_rq5_demand_pull.py` — class-level demand-pull regression on n = 57; associational.

The methodological next step on the methods side is `scripts/blocked/interaction_estimand.py` (a fully implemented but not-yet-run ΔKL ~ logpat × class-churn moderation with firm + year fixed effects and a kernel-independent churn measure) and a cross-paradigm topic-extraction test (sentence-embedding clustering or graph community detection) against the LDA/NMF kill conditions.

## Data not in this repo

The raw USPTO backfile (~12 GB), per-class records/vocab/surprise/outcome parquets (~15 GB after build), and SEC EDGAR FSDS ZIPs (~7 GB) are excluded by `.gitignore`. The reproduction pipeline regenerates them from public sources.

## Citing this work

Working-paper citation (the SSRN ID will be added when posted):

> Silver, E. (2026). *Vocabulary resonance in US trademark filings: cross-industry diffusion structure, the U-shape in registration outcomes, and kill conditions for an applied DeDeo measure*. SSRN Working Paper.

## Author / contact

Eric Silver — `epsilver@gmail.com`.

Independent researcher; formerly a PhD candidate at Carnegie Mellon University. The author's current employment is unrelated to this work, and the views expressed are the author's alone.

## License

GPL-3.0 (see `LICENSE`).
