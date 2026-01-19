.PHONY: install dev lint test data

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

data:
	python scripts/mgpt.py data build --config configs/data.yaml
