.PHONY: setup lint format test smoke
PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -q

# Filled in at the training step (CPU smoke run on ~20 images)
smoke:
	@echo "smoke test not implemented yet"
