# miniGPT

**miniGPT** is a laptop-scale, end-to-end implementation of a modern large language model (LLM) pipeline.  
The goal of the project is to build and understand the full lifecycle of an LLM — from raw data ingestion to training, alignment, evaluation, and deployment — using clean, modular, and reproducible code.

This repository is designed as a hands-on systems project, focusing on clarity and correctness over production scale or benchmarking performance.


---

## Current Capabilities

At its current stage, the repository implements the following core components:

### Dataset Pipeline
- Streaming ingestion from Hugging Face datasets
- Text cleaning and quality filtering
- Exact and near-duplicate removal (SimHash-based)
- Weighted mixture of multiple data sources
- Sharding into JSONL files for downstream processing
- Summary statistics and reproducible configuration via YAML

### Tokenization
- Byte-level BPE tokenizer training (GPT-style)
- Configurable vocabulary size and special tokens
- Tokenizer analysis (coverage and efficiency metrics)
- Encoding of the dataset into token ID binaries (`.bin`) with document index metadata (`.idx.json`)

These components form a clean and reproducible foundation for subsequent model training.

---

## Repository Structure (Partial)

```text
configs/                 # YAML configuration files
src/minigpt/
  common/                # Shared utilities (logging, paths, text cleaning)
  data/                  # Dataset ingestion and deduplication
  tokenizer/             # Tokenizer training, analysis, encoding
scripts/
  mgpt.py                # CLI entry point
tests/                   # Unit and smoke tests
```

Generated artifacts (datasets, tokenizers, encoded corpora) are written to local output directories and are not committed to version control.

---

## Quickstart

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

Build the processed dataset:

```bash
make data
```

Train and analyze the tokenizer:

```bash
make tok
make encode
```

Run the test suite:

```bash
make test
```

## Roadmap

The repository will be extended to cover the full LLM development lifecycle, including:

- Autoregressive GPT-style model training
- Supervised fine-tuning and preference optimization
- Evaluation and benchmarking
- Retrieval-augmented generation (RAG)
- Tool-using agents
- Inference and deployment infrastructure

The emphasis remains on correctness, reproducibility, and systems-level understanding rather than scale.