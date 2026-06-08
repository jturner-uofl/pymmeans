# =====================================================================
# pymmeans — replication entry points
#
# Targets:
#   make reproduce       Re-run the full validation evidence base:
#                        - install the package + dev/tutorial extras
#                          into a fresh virtualenv
#                        - run the pytest suite (1,037 tests)
#                        - rebuild + execute the narrative notebook
#                          (251 contracts)
#                        - render the notebook to HTML
#   make test            Just run pytest.
#   make notebook        Just rebuild + execute the validation notebook.
#   make html            Just render the executed notebook to HTML.
#   make benchmarks      Re-run the §XVIII performance benchmarks
#                        (requires Rscript + R packages emmeans, lme4,
#                        lmerTest, survival).
#   make clean           Delete the build artifacts (does not delete
#                        the validation notebook source).
# =====================================================================

PYTHON       ?= python3
VENV         ?= .venv
PYBIN        := $(VENV)/bin
PIP          := $(PYBIN)/pip
PYTEST       := $(PYBIN)/pytest
JUPYTER      := $(PYBIN)/jupyter

NOTEBOOK_SRC := examples/jss_audit/_build_case_study.py
NOTEBOOK_OUT := examples/jss_audit/jss_case_study.ipynb
NOTEBOOK_HTML := examples/jss_audit/jss_case_study.html

NB_TIMEOUT   ?= 3600

.PHONY: reproduce setup test notebook html benchmarks clean

reproduce: setup test notebook html
	@echo ""
	@echo "============================================================"
	@echo "  Replication complete."
	@echo "    pytest:    pass count above"
	@echo "    notebook:  $(NOTEBOOK_OUT)"
	@echo "    rendered:  $(NOTEBOOK_HTML)"
	@echo "============================================================"

setup:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating fresh virtualenv at $(VENV)..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,plot,tutorial]"

test:
	$(PYTEST) -q

notebook:
	$(PYBIN)/python $(NOTEBOOK_SRC)
	$(JUPYTER) nbconvert --to notebook --execute --inplace \
		$(NOTEBOOK_OUT) \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT)

html:
	$(JUPYTER) nbconvert --to html $(NOTEBOOK_OUT)

benchmarks:
	@command -v Rscript >/dev/null 2>&1 || { \
		echo "ERROR: Rscript not found. Install R + emmeans/lme4/lmerTest/survival."; \
		exit 1; \
	}
	@echo "Running R side..."
	Rscript /tmp/pymmeans_bench/bench_runs.R
	@echo "Running pymmeans side..."
	$(PYBIN)/python /tmp/bench_py.py

clean:
	rm -rf build dist *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	@echo "Build artifacts cleared. Notebook source preserved."
