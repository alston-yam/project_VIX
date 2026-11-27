# VRP S&P 500 - v1 data layer + v2 statistical tests + research section.
# Run `make all` from a fresh clone (after `pip install -r requirements.txt`).

PYTHON ?= python
export PYTHONPATH := src

.PHONY: all data refresh figures stats research deck test lint format clean help

help:  ## Show available targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

all: data figures stats research deck test  ## Build data, figures, tests, research, deck, then run tests

data:  ## Download (cached), clean, build features, sample, events, validate
	$(PYTHON) -m vrp.pipeline --stage data

refresh:  ## Same as `data` but force fresh downloads
	$(PYTHON) -m vrp.pipeline --stage data --refresh

figures:  ## Draw all figures from processed data
	$(PYTHON) -m vrp.pipeline --stage figures

stats:  ## v2 statistical tests: tables, summary, pitch claims, figures 07–11
	$(PYTHON) -m vrp.pipeline --stage stats

research:  ## Research section: tests R-1…R-9, reports, research claims, figures 12–15
	$(PYTHON) -m vrp.pipeline --stage research

deck:  ## Pitch deck from the OAF template → deck/VIX_pitch.pptx
	$(PYTHON) -m vrp.pipeline --stage deck

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

lint:  ## Check formatting and lint
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m ruff check src tests

format:  ## Auto-format code
	$(PYTHON) -m black src tests
	$(PYTHON) -m ruff check --fix src tests

clean:  ## Delete downloaded/intermediate data, logs and caches (keeps committed outputs)
	rm -rf data/raw/* data/interim/* logs/*.log .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	touch data/raw/.gitkeep data/interim/.gitkeep
