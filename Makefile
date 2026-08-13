# End-to-end build for the trademark vocabulary paper.
#
# A fresh checkout can produce paper/ssrn_diffusion_paper.pdf with:
#     make setup tm sec crosswalk analysis paper
# or simply:
#     make all
#
# Each stage is incremental and idempotent: stages 1-3 are heavy data builds
# (multi-hour) and write sentinel files in data/processed/; stage 4 (analysis)
# rebuilds the figures, tables, and per-class summaries the paper consumes;
# stage 5 (paper) compiles ssrn_diffusion_paper.pdf from the .tex sources and
# artifacts.

PY      := .venv/bin/python
DATA    := data/processed
RES     := paper/results
PPYTHON := PYTHONPATH=src $(PY)

.PHONY: all venv setup tm sec crosswalk analysis gates paper figures industry clean clean-results help

help:
	@echo "Targets:"
	@echo "  setup         create venv, install deps"
	@echo "  tm            download every NICE class, build vocab/surprise/firm-year/outcomes"
	@echo "  sec           download SEC FSDS, parse to firm-year financials"
	@echo "  crosswalk     USPTO owner_name <-> SEC CIK match"
	@echo "  analysis      regenerate every figure, table, and per-class summary"
	@echo "  gates         regenerate the gate robustness checks"
	@echo "  paper         compile paper/ssrn_diffusion_paper.pdf"
	@echo "  industry      regenerate the per-industry appendix .tex fragments"
	@echo "  all           setup + tm + sec + crosswalk + analysis + paper"
	@echo "  clean         remove LaTeX intermediates"
	@echo "  clean-results remove every generated figure/table (forces rebuild)"

# ---------------------------------------------------------------------------
# Stage 1: setup
# ---------------------------------------------------------------------------
setup: .venv/bin/python
.venv/bin/python:
	python3.11 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -e .
	$(PY) -m pip install --quiet statsmodels matplotlib rapidfuzz

# ---------------------------------------------------------------------------
# Stage 2: TM corpus (download + per-class vocab/surprise/firm-year/outcomes)
# H=2y exponential-decay reference is production default; flat-window
# baseline kept under src/novelty/surprise.py for the half-life sweep only.
# ---------------------------------------------------------------------------
tm: $(DATA)/.tm_done
$(DATA)/.tm_done: $(PY)
	$(PPYTHON) scripts/download_all_classes.py
	$(PPYTHON) scripts/process_all_classes.py
	$(PPYTHON) scripts/recompute_h2.py
	touch $@

# ---------------------------------------------------------------------------
# Stage 3: SEC EDGAR financials + crosswalk
# ---------------------------------------------------------------------------
sec: $(DATA)/sec_firm_year.parquet
$(DATA)/sec_firm_year.parquet: $(PY)
	$(PY) scripts/download_sec_fsds.py
	$(PPYTHON) scripts/sec_extract.py

crosswalk: $(DATA)/uspto_sec_crosswalk.parquet
$(DATA)/uspto_sec_crosswalk.parquet: $(DATA)/sec_firm_year.parquet $(DATA)/.tm_done
	$(PPYTHON) scripts/sec_link.py

# ---------------------------------------------------------------------------
# Stage 4: analysis - regenerate every figure and table the paper consumes
# ---------------------------------------------------------------------------
ANALYSIS_DEPS := $(DATA)/.tm_done $(DATA)/uspto_sec_crosswalk.parquet

# Catch-all metrics + the core figure/table builders the paper imports.
$(RES)/_metrics.json: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/analysis_full.py
	$(PPYTHON) scripts/financials_regression.py
	$(PPYTHON) scripts/cross_industry.py
	$(PPYTHON) scripts/returns_regression.py
	$(PPYTHON) scripts/r2_decomposition.py
	$(PPYTHON) scripts/per_firm_dynamics.py
	$(PPYTHON) scripts/u_shape_analysis.py
	$(PPYTHON) scripts/industry_vs_finance.py
	$(PPYTHON) scripts/templated_survival.py
	$(PPYTHON) scripts/patent_compare.py
	$(PPYTHON) scripts/vocab_trajectory.py

# Headline-finding figures developed in the most recent rebuild.
$(RES)/survival_by_kl_lines.png: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/survival_by_kl_lines.py
$(RES)/outcome_by_kl_lines.png: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/outcome_by_kl_lines.py
$(RES)/outcome_by_kl_lines_v2.png: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/outcome_by_kl_lines_v2.py
$(RES)/debut_outcome_by_kl.png: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/debut_outcome_by_kl.py
$(RES)/debut_harvest_fit.png: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/debut_harvest_fit.py
$(RES)/u_shape_industry_year_table.tex: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/u_shape_industry_year_matrix.py
$(RES)/halflife_signal_class009.png: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/halflife_signal.py

# Flux-token + vanishing-trend evidence (token-level corroboration of the U).
$(RES)/symmetric_flux_tokens.tex: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/symmetric_flux_tokens.py
$(RES)/vanishing_trend_examples.txt: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/vanishing_trend_examples.py

# Pioneer-ladder + cross-buzzword serial-pioneer analyses.
$(RES)/pioneer_ladder_summary.png: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/pioneer_ladder.py
$(RES)/cross_buzzword_pioneers.txt: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/cross_buzzword_pioneers.py

# Canonical-winner + failed-firm KL profile.
$(RES)/big_firm_kl_v2.tex: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/big_firm_kl_v2.py
	$(PPYTHON) scripts/build_big_firm_kl_v2_tex.py

# Per-industry appendix tex fragments (one row per NICE class).
$(RES)/per_industry_appendix.tex: $(ANALYSIS_DEPS)
	$(PPYTHON) scripts/per_industry_appendix.py

# All paper-consumed figures and tables.
FIGURES := \
  $(RES)/_metrics.json \
  $(RES)/survival_by_kl_lines.png \
  $(RES)/outcome_by_kl_lines.png \
  $(RES)/outcome_by_kl_lines_v2.png \
  $(RES)/debut_outcome_by_kl.png \
  $(RES)/debut_harvest_fit.png \
  $(RES)/u_shape_industry_year_table.tex \
  $(RES)/halflife_signal_class009.png \
  $(RES)/symmetric_flux_tokens.tex \
  $(RES)/vanishing_trend_examples.txt \
  $(RES)/pioneer_ladder_summary.png \
  $(RES)/cross_buzzword_pioneers.txt \
  $(RES)/big_firm_kl_v2.tex \
  $(RES)/per_industry_appendix.tex

# Gate robustness checks. Each reads the scored parquets directly and is
# independent of the others, so they can be built in any order.
GATE_CHECKS := \
  $(RES)/gate_duration.json \
  $(RES)/gate_censoring_check.json \
  $(RES)/gate_prewindow_check.json \
  $(RES)/gate_era_profile.png \
  $(RES)/tie_rule_check.json \
  $(RES)/window_sweep_rolling.json

$(RES)/gate_duration.json: scripts/gate_duration.py
	$(PPYTHON) $<
$(RES)/gate_censoring_check.json: scripts/gate_censoring_check.py
	$(PPYTHON) $<
$(RES)/gate_prewindow_check.json: scripts/gate_prewindow_check.py
	$(PPYTHON) $<
$(RES)/gate_era_profile.png: scripts/gate_era_profile.py
	$(PPYTHON) $<
$(RES)/tie_rule_check.json: scripts/tie_rule_check.py
	$(PPYTHON) $<
$(RES)/window_sweep_rolling.json: scripts/window_sweep_rolling.py
	$(PPYTHON) $<

analysis: $(FIGURES)

gates: $(GATE_CHECKS)

figures: analysis

industry: $(RES)/per_industry_appendix.tex

# ---------------------------------------------------------------------------
# Stage 5: paper
# ---------------------------------------------------------------------------
paper: paper/ssrn_diffusion_paper.pdf
paper/ssrn_diffusion_paper.pdf: paper/ssrn_diffusion_paper.tex \
                                paper/section_construct.tex \
                                $(FIGURES) $(GATE_CHECKS)
	cd paper && pdflatex -interaction=nonstopmode ssrn_diffusion_paper.tex
	cd paper && pdflatex -interaction=nonstopmode ssrn_diffusion_paper.tex

# ---------------------------------------------------------------------------
all: setup tm sec crosswalk analysis gates paper

clean:
	rm -f paper/main.aux paper/main.log paper/main.out paper/main.toc
	rm -f paper/short.aux paper/short.log paper/short.out paper/short.toc

clean-results:
	rm -f $(RES)/*.png $(RES)/*.tex $(RES)/*.csv $(RES)/*.txt $(RES)/*.json
