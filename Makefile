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

tok:
	python scripts/mgpt.py tokenizer train --config configs/tokenizer.yaml
	python scripts/mgpt.py tokenizer analyze --config configs/tokenizer.yaml

encode:
	python scripts/mgpt.py tokenizer encode --config configs/tokenizer.yaml
