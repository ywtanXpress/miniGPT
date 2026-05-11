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

train:
	python scripts/mgpt.py train pretrain --config configs/train.yaml

train_resume:
	python scripts/mgpt.py train pretrain --config configs/train.yaml --resume

PROMPT ?= Once upon a time

sample:
	python scripts/mgpt.py train sample --config configs/train.yaml --prompt "$(PROMPT)"

sft_data:
	python scripts/mgpt.py sft build --config configs/sft.yaml

sft_train:
	python scripts/mgpt.py sft train --config configs/sft.yaml

sft_resume:
	python scripts/mgpt.py sft train --config configs/sft.yaml --resume

INSTR ?= Explain what a transformer is.

sft_sample:
	python scripts/mgpt.py sft sample --config configs/sft.yaml --instruction "$(INSTR)"

eval_pretrain:
	python scripts/mgpt.py eval pretrain --config configs/eval.yaml

eval_sft:
	python scripts/mgpt.py eval sft --config configs/eval.yaml

eval_dpo:
	python scripts/mgpt.py eval dpo --config configs/eval.yaml

eval_rag:
	python scripts/mgpt.py eval rag --config configs/eval.yaml

dpo_data:
	python scripts/mgpt.py dpo build --config configs/dpo.yaml

dpo_train:
	python scripts/mgpt.py dpo train --config configs/dpo.yaml

dpo_resume:
	python scripts/mgpt.py dpo train --config configs/dpo.yaml --resume

dpo_sample:
	python scripts/mgpt.py dpo sample --config configs/dpo.yaml --instruction "$(INSTR)"

rag_build:
	python scripts/mgpt.py rag build --config configs/rag.yaml

QUERY ?= What is a transformer?

rag_retrieve:
	python scripts/mgpt.py rag retrieve --config configs/rag.yaml --query "$(QUERY)"

rag_query:
	python scripts/mgpt.py rag query --config configs/rag.yaml --question "$(QUERY)"
