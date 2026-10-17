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

# CPU smoke run on ~20 synthetic images (needs: pip install -e ".[train]")
smoke:
	@$(PY) -m src.train.smoke
