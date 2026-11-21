# VRP S&P 500 - v1 data layer.
# Run `make all` from a fresh clone (after `pip install -r requirements.txt`).

PYTHON ?= python
export PYTHONPATH := src

.PHONY: all data refresh figures test lint format clean help

help:  ## Show available targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

all: data figures test  ## Build data, figures, then run tests

data:  ## Download (cached), clean, build features, sample, events, validate
	$(PYTHON) -m vrp.pipeline --stage data

refresh:  ## Same as `data` but force fresh downloads
	$(PYTHON) -m vrp.pipeline --stage data --refresh

figures:  ## Draw all figures from processed data
	$(PYTHON) -m vrp.pipeline --stage figures

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
