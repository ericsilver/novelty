# Reproduce the entire pipeline. Each stage is idempotent on a fresh checkout.

PY      := .venv/bin/python
DATA    := data/processed

.PHONY: all venv setup tm sec crosswalk analysis paper clean help

help:
	@echo "Targets:"
	@echo "  setup       create venv, install deps"
	@echo "  tm          download every NICE class + build vocab/surprise/firm-year/survival"
	@echo "  sec         download SEC FSDS, parse to firm-year financials"
	@echo "  crosswalk   USPTO -> SEC entity match (exact normalize + fuzzy expansion)"
	@echo "  analysis    regenerate every figure + table the paper consumes"
	@echo "  paper       compile paper/main.pdf"
	@echo "  all         setup + tm + sec + crosswalk + analysis + paper"

setup: .venv/bin/python
.venv/bin/python:
	python3.11 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -e .
	$(PY) -m pip install --quiet statsmodels matplotlib rapidfuzz

# Universal sweep: writes data/processed/tm_class<NNN>.parquet for every NICE
# class with any matching filings. Then per-class dictionary/surprise/firm-year/
# survival.
tm: $(DATA)/.tm_done
$(DATA)/.tm_done: $(PY)
	PYTHONPATH=src $(PY) scripts/download_all_classes.py
	PYTHONPATH=src $(PY) scripts/process_all_classes.py
	touch $@

sec: $(DATA)/sec_firm_year.parquet
$(DATA)/sec_firm_year.parquet: $(PY)
	$(PY) scripts/download_sec_fsds.py
	PYTHONPATH=src $(PY) scripts/sec_extract.py

crosswalk: $(DATA)/uspto_sec_crosswalk.parquet
$(DATA)/uspto_sec_crosswalk.parquet: $(DATA)/sec_firm_year.parquet $(DATA)/.tm_done
	PYTHONPATH=src $(PY) scripts/sec_link.py

analysis: paper/results/_metrics.json
paper/results/_metrics.json: $(DATA)/.tm_done $(DATA)/uspto_sec_crosswalk.parquet
	PYTHONPATH=src $(PY) scripts/analysis_full.py
	PYTHONPATH=src $(PY) scripts/financials_regression.py
	PYTHONPATH=src $(PY) scripts/cross_industry.py

paper: paper/main.pdf
paper/main.pdf: paper/main.tex paper/results/_metrics.json
	cd paper && pdflatex -interaction=nonstopmode main.tex
	cd paper && pdflatex -interaction=nonstopmode main.tex

all: setup tm sec crosswalk analysis paper

clean:
	rm -f paper/main.aux paper/main.log paper/main.out paper/main.toc
	rm -f $(DATA)/.tm_done
