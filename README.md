# Trademark text resonance and business-model diffusion

Code, public data panels, and working papers for a text-based novelty measure on US trademark filings (USPTO TRTYRAP, 1884 backfile; analysis windows 1990–2024 across 45 NICE classes).

The construct, denoted ΔKL, applies the DeDeo prospective/retrospective Kullback–Leibler resonance framework (Murdock, Allen & DeDeo 2017; Barron, Huang, Spang & DeDeo 2018) to a filing's goods/services text. The work is organized around two related questions:

1. **Construct validity and outcomes** (the original research line). Is ΔKL a useful innovation-related signal at the firm level? Within-firm, ΔKL moves opposite to patenting (–0.048σ, t = –7.9 on 14,463 firms). Cross-sectionally, post-registration §8 mark maintenance is U-shaped in ΔKL: filings whose vocabulary is moving with the field and filings whose vocabulary is moving against the field both outperform the flux-neutral middle. Firm-mean ΔKL aligns weakly but positively with expert "most innovative" lists (BCG 2021–23, MIT TR50 2015; pooled +0.34σ, n = 48).

2. **Cross-industry diffusion of business-model vocabulary** (the diffusion extension). At the theme level, do business-model vocabularies diffuse across industries with measurable structure? Software- and tech-services-origin themes (cloud, AI, blockchain, mobile-app, streaming, AR/VR, as-a-service) arrive in transport and advertising/retail with multi-year lags. The diffusion structure passes two independent kill-condition tests at z = +46 (LDA shuffle null) and z = +60.7 (NMF independent replication). The Schumpeter Mark I (entrant-led innovation) reading collapses on year-matched baseline contrast.

## Start here

| Paper | Pages | Focus |
|---|---|---|
| **`paper/ssrn_diffusion_paper.pdf`** | 13 | The SSRN working paper. Diffusion-lead structure. **Read this first.** |
| `paper/integrated_report.pdf` | 17 | Consolidated working report. Part I construct validity; Part II diffusion Phase-0/1. |
| `paper/main.pdf` | 223 | The original long-form ΔKL paper on the U-shaped maintenance pattern. |
| `paper/short.pdf` | 7 | Short version of the original. |
| `paper/dynamism.pdf` | 13 | USPTO debut coverage of US business formation; crowding tests. |
| `paper/ethnic_clusters_note.pdf` | 21 | US-only ethnic clusters; patent-guild collapse; Black trademark-rise vs patent-fall. |
| `paper/newterms_report.pdf` | 12 | Cross-industry new vocabulary 1995–2021; top 100 themes. |

Design documents:

- `METHOD.md` — the 2026-04 DeDeo prospective/retrospective KL update.
- `PROPOSAL.md` — original Bayesian-surprise proposal.
- `PROPOSAL_diffusion.md` — the business-model-diffusion program proposal.
- `PROJECT.md` — umbrella project framing and target.
- `BLOCKED.md` — remaining workstreams with decision rules stated ex ante.

## What's in this repo

```
.
├── paper/                 papers (.tex + compiled .pdf) and analysis outputs
│   └── results/           JSON metrics, figures, tables, per-class outputs
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

# 4. SEC financial panel (for the RQ5/RQ6 outcome bridges; 2009–2024)
make sec
make crosswalk

# 5. analysis
make analysis
```

`make all` walks the whole pipeline. `make help` lists individual targets. The build is incremental.

The SSRN paper and the working notes are not built by `make` — compile them directly:

```bash
cd paper
pdflatex ssrn_diffusion_paper.tex   # twice for citations
pdflatex ssrn_diffusion_paper.tex
```

The same pattern works for `integrated_report.tex`, `construct_validity_note.tex`, `diffusion_phase0_note.tex`, `newterms_report.tex`.

## Result lookup

The empirical claims in the SSRN paper and their script → output paths:

| Claim | Script | Output |
|---|---|---|
| ΔKL ⟂ patents within firm; −0.048σ, t = −7.9 | `scripts/wsC_within_firm_patents.py` | `paper/results/wsC_within_firm_patents.json` |
| BCG/MIT discriminant: +0.34σ pooled, n = 48 | `scripts/wsD_validity_battery.py` | `paper/results/wsD_validity_battery.json` |
| +28pp return is artifactual; horizon profile non-monotone | `scripts/wsE_returns_diagnostic.py` | `paper/results/wsE_returns_diagnostic.json` |
| Class-009 diagnostics: 0.97 collinearity, length artifact, sign robustness | `scripts/wsG_class009_analysis.py` | `paper/results/wsG_class009.json` |
| Within-class term provenance: corr(novel\_share, ΔKL) = +0.29 | `scripts/wsH_term_provenance009.py` | `paper/results/wsH_provenance009.json` |
| Phrase transit table (cloud, AI, blockchain, …) | `scripts/diff2_theme_transit.py` | `paper/results/diff2_transit.json` |
| LDA T = 50 + kill condition z = +46 | `scripts/diff3_lda_themes.py` | `paper/results/diff3_themes.json` |
| Bass fits (median q/p ≈ 7) + class-flow asymmetry | `scripts/diff4_phase1_rqs.py` | `paper/results/diff4_phase1.json` |
| Schumpeter I collapse: −2.7pp excess vs baseline | `scripts/diff5_rq4_baseline.py` | `paper/results/diff5_rq4_baseline.json` |
| NMF cross-method check: z = +60.7; seed-phrase concordance 7/10 | `scripts/diff6_nmf_compare.py` | `paper/results/diff6_nmf.json` |
| Face-validation form (50 LDA topics) | `scripts/diff7_face_validation_doc.py` | `paper/face_validation.md` |
| RQ6 firm outcome: +1.4pp margin lift, R&D substitution | `scripts/diff8_rq6_borrowed_pays.py` | `paper/results/diff8_rq6.json` |
| RQ5 class level: lagged revenue growth → theme inflow | `scripts/diff9_rq5_demand_pull.py` | `paper/results/diff9_rq5.json` |
| New cross-industry vocabulary: 3,653 candidate terms 1995–2021 | `scripts/newterms_analysis.py` + `newterms_build_pdf.py` | `paper/results/newterms_top100.csv` + `paper/newterms_report.pdf` |

`scripts/blocked/interaction_estimand.py` is the next methodological step: it implements the ΔKL ~ logpat × class-churn moderation with firm + year fixed effects, with the kernel-independent churn measure the critique demanded; it ships ready to run when per-class surprise parquets exist.

## Data not in this repo

The raw USPTO backfile (~12 GB), per-class records/vocab/surprise/outcome parquets (~15 GB after build), and SEC EDGAR FSDS ZIPs (~7 GB) are excluded by `.gitignore`. The reproduction pipeline regenerates them from public sources.

## Citing this work

Working-paper citation (the SSRN ID will be added here when posted):

> Silver, E. (2026). *How business-model vocabulary diffuses across US industries: text-based evidence from trademark filings, with kill-condition tests of the diffusion structure*. SSRN Working Paper.

## Author / contact

Eric Silver — `epsilver@gmail.com`.

Independent researcher; formerly a PhD candidate at Carnegie Mellon University. The author's current employment is unrelated to this work, and the views expressed are the author's alone.

## License

GPL-3.0 (see `LICENSE`).
