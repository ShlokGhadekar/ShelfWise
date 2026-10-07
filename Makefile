# ShelfWise - common commands. Run `make help` for the list.
# Variables you can override:  make run SUBSET=CA   make run LAST_DAY=1913   make run FULL_REFRESH=1

SHELL := /bin/bash
# Python used to create the virtualenv, e.g. make setup PYTHON=python3.12
PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PIPELINE := $(PY) -m shelfwise.pipeline.run

# Load .env (if present) so docker compose and Python see the same settings.
ifneq (,$(wildcard .env))
include .env
export
endif

# SUBSET and LAST_DAY reach Python as environment variables (thanks to `export`),
# whether they come from .env or the command line (`make run SUBSET=CA`).
ARGS :=
ifdef FULL_REFRESH
ARGS += --full-refresh
endif

.PHONY: help setup env db-up db-down db-reset db-shell check-data run run-dev demo-incremental \
        load staging quality features export backtest evaluate forecast benchmark api dashboard \
        test test-integration lint clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

env:  ## create .env from the template (won't overwrite)
	@test -f .env || (cp .env.example .env && echo "created .env - edit POSTGRES_PASSWORD")

setup: env  ## create virtualenv and install dependencies + the package (editable)
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

db-up:  ## start Postgres in Docker and wait until it is ready
	docker compose up -d
	@echo "waiting for postgres..."; \
	for i in $$(seq 1 60); do \
	  docker compose exec -T postgres pg_isready -U $${POSTGRES_USER:-shelfwise} >/dev/null 2>&1 && echo "postgres is ready" && exit 0; \
	  sleep 2; \
	done; echo "postgres did not become ready"; exit 1

db-down:  ## stop Postgres (data is kept in the docker volume)
	docker compose down

db-reset:  ## DANGER: stop Postgres and delete its data volume
	docker compose down -v

db-shell:  ## open psql inside the container
	docker compose exec postgres psql -U $${POSTGRES_USER:-shelfwise} -d $${POSTGRES_DB:-shelfwise}

check-data:  ## verify the three M5 CSV files are in data/raw
	@for f in sales_train_evaluation.csv calendar.csv sell_prices.csv; do \
	  test -f data/raw/$$f && echo "ok      data/raw/$$f" || (echo "MISSING data/raw/$$f (see README: Get the data)"; exit 1); \
	done

run: check-data  ## run the full pipeline end to end (incremental where possible)
	$(PIPELINE) --steps all $(ARGS)

run-dev: check-data  ## full pipeline on a single store (fast development loop)
	$(PIPELINE) --steps all --subset CA_1 $(ARGS)

demo-incremental: check-data  ## build up to d_1913, then load the last 28 days incrementally
	$(PIPELINE) --steps init,load,staging,quality,features --last-day 1913 --full-refresh
	$(PIPELINE) --steps staging,quality,features,export --last-day 1941

load:      ; $(PIPELINE) --steps init,load $(ARGS)
staging:   ; $(PIPELINE) --steps staging $(ARGS)
quality:   ; $(PIPELINE) --steps quality $(ARGS)
features:  ; $(PIPELINE) --steps features $(ARGS)
export:    ; $(PIPELINE) --steps export $(ARGS)
backtest:  ; $(PIPELINE) --steps backtest $(ARGS)

evaluate:  ## metrics, significance tests, error analysis, SHAP, decision simulation (needs backtest)
	$(PIPELINE) --steps evaluate,decide $(ARGS)

forecast:  ## retrain on all data and write the next-28-day forecast + orders to the DB
	$(PIPELINE) --steps forecast $(ARGS)

benchmark:  ## time key queries before/after partitioning + indexing (EXPLAIN ANALYZE)
	$(PY) -m shelfwise.benchmark.run_benchmark

api:  ## serve the FastAPI app on http://localhost:8000 (docs at /docs)
	$(VENV)/bin/uvicorn shelfwise.serving.api:app --reload --port 8000

dashboard:  ## open the Streamlit dashboard on http://localhost:8501
	$(VENV)/bin/streamlit run src/shelfwise/serving/dashboard.py

test:  ## unit tests (no database needed)
	$(PY) -m pytest

test-integration:  ## post-run checks against the real database
	$(PY) -m pytest -m integration

lint:  ## static checks
	$(VENV)/bin/ruff check src tests

clean:  ## delete caches, models and generated reports (keeps raw data and the DB)
	rm -rf data/cache/*.parquet data/cache/*.npz data/cache/*.tmp models/backtest models/final \
	       reports/results/* reports/figures/* reports/benchmarks/* reports/data_quality/* \
	       reports/run_timings.csv .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@for d in reports/results reports/figures reports/benchmarks reports/data_quality; do touch $$d/.gitkeep; done
