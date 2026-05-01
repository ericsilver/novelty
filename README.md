# novelty

KL surprise scores for trademark goods/services descriptions, applied to the
full USPTO backfile and joined to SEC EDGAR financials. The original research
proposal lives in `PROPOSAL.md`; the implemented paper, with all figures and
regressions, lives in `paper/main.pdf` (download
[here](paper/main.pdf)).

## What's here

```
src/novelty/        Library code
  uspto.py            Open Data Portal API client (TRTYRAP backfile)
  parse.py            Streaming lxml parser for USPTO case-file XML
  download.py         CLI: download + filter + write Parquet for one NICE class
  dictionary.py       CLI: build unigram + bigram dictionary
  surprise.py         CLI: compute KL_pros, KL_retr per filing
  firm_year.py        CLI: aggregate per-filing scores to (owner, year)
  survival.py         CLI: derive trademark-survival outcomes
  industries.py       NICE-class to industry-label map
  token_attribution.py  Per-token contribution to a filing's KL

scripts/            Top-level runners
  download_sec_fsds.py    Polite SEC FSDS quarterly ZIP downloader (60 quarters)
  sec_extract.py          Parse SEC FSDS into firm-year financials
  sec_link.py             USPTO owner_name <-> SEC CIK crosswalk
  financials_regression.py  Innovation -> gross margin regression
  analysis_full.py          Build every figure + table the paper consumes

paper/
  main.tex            LaTeX source
  main.pdf            Compiled paper
  results/            Generated figures + LaTeX-table snippets
```

## Reproducing

Requires Python 3.11+, a TeX install (TeX Live), and a free USPTO Open Data
Portal API key from <https://data.uspto.gov> (My ODP > My API Key).

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e .

# Put your API key in .env:
cp .env.example .env  # edit USPTO_ODP_API_KEY=...

# 1) Download + parse one industry (~10 min, peak disk ~300MB during stream)
python -m novelty.download --nice 009    # software & electronics
python -m novelty.download --nice 032    # beer & soft drinks

# 2) Build vocab + surprise + firm-year + survival outcome for each class
for cls in 009 032; do
  python -m novelty.dictionary --input data/processed/tm_class$cls.parquet \
                               --out   data/processed/vocab_class$cls.parquet
  python -m novelty.surprise --records data/processed/tm_class$cls.parquet \
                             --vocab   data/processed/vocab_class$cls.parquet \
                             --out     data/processed/surprise_class$cls.parquet
  python -m novelty.firm_year --surprise data/processed/surprise_class$cls.parquet \
                              --out      data/processed/firm_year_class$cls.parquet
  python -m novelty.survival --records data/processed/tm_class$cls.parquet \
                             --surprise data/processed/surprise_class$cls.parquet \
                             --out      data/processed/outcomes_class$cls.parquet
done

# 3) SEC EDGAR financials (~7 GB download, ~5 min parse)
python scripts/download_sec_fsds.py
python scripts/sec_extract.py
python scripts/sec_link.py

# 4) Run analyses, compile paper
PYTHONPATH=src python scripts/analysis_full.py
PYTHONPATH=src python scripts/financials_regression.py
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Headline finding

In the matched panel of 1,408 publicly-traded firms × 4,110 firm-years
(2009–2024), a one-σ increase in the firm's 3-year trailing mean
$\Delta KL$ raises gross margin by **+2.9 percentage points** (SE 0.7 pp).
At the per-filing level, the same novelty axis lowers the probability of
clearing trademark examination but raises the probability of being a
currently-live registration, recovering the risk/reward asymmetry that the
"pioneers vs. Blue Ocean" debate predicts.

## Data not in this repo

The raw USPTO backfile (~12 GB), per-class records/vocab/surprise/outcome
parquets (~1 GB), and SEC EDGAR FSDS ZIPs (~7 GB) are excluded by
`.gitignore`. The reproduction script regenerates them from public sources.
