# miniGPT

**miniGPT** is a laptop-scale, end-to-end implementation of a modern large language model (LLM) pipeline.  
The goal of the project is to build and understand the full lifecycle of an LLM — from raw data ingestion to training, alignment, evaluation, and deployment — using clean, modular, and reproducible code.

This repository is designed as a hands-on systems project, focusing on clarity and correctness over production scale or benchmarking performance.


---

## Current Capabilities

At its current stage, **miniGPT** implements the following components end-to-end:

### 1. Dataset Pipeline
- Streaming ingestion from Hugging Face datasets
- Text cleaning and quality filtering
- Exact and near-duplicate removal (SimHash-based)
- Weighted mixture of multiple data sources
- Sharding into JSONL files for downstream processing
- Summary statistics and reproducible configuration via YAML

This stage mirrors internet-scale preprocessing logic in a laptop-friendly form.

### 2. Tokenization
- Byte-level BPE tokenizer training (GPT-style)
- Configurable vocabulary size and special tokens
- Tokenizer analysis (coverage and efficiency metrics)
- Encoding of the dataset into token ID binaries (`.bin`) with document index metadata (`.idx.json`)

The tokenizer is trained from scratch and reused consistently across all training stages.

### 3. Autoregressive Pre-training
- GPT-style Transformer decoder model
- Causal language modeling objective (next-token prediction)
- Configurable model depth, width, and attention heads
- Cosine learning rate schedule with warmup and minimum LR
- Gradient accumulation and mixed-precision support
- Periodic evaluation, logging, and checkpointing

This stage trains the model to learn general language structure and fluency from raw text.

### 4. Supervised Fine-Tuning (SFT)
- Instruction–response dataset ingestion (e.g. Alpaca-style)
- Deterministic prompt templating
- Prefix-only supervision (loss applied only to assistant responses)
- Fixed-length block truncation starting from sequence beginning
- Minimum response-length filtering to ensure learnable targets
- Separate SFT training configuration and checkpoints

SFT adapts the pretrained model for basic instruction-following behavior.

---

## Repository Structure (Partial)

```text
configs/                 # YAML configuration files
  train.yaml             # Pretraining configuration
  sft.yaml               # Supervised fine-tuning configuration
src/minigpt/
  common/                # Shared utilities (logging, paths, helpers)
  data/                  # Dataset ingestion, filtering, deduplication
  tokenizer/             # Tokenizer training, analysis, encoding
  model/                 # GPT model definition
  train/                 # Training loops, schedulers, evaluation
  sft/                   # SFT dataset construction and training logic
scripts/
  mgpt.py                # CLI entry point
tests/                   # Unit and smoke tests
```

Generated artifacts (datasets, tokenizers, encoded corpora) are written to local output directories and are not committed to version control.

---

## Quickstart

### Requirements
- Python 3.10+
- CUDA-capable GPU recommended but not required

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

Train a GPT-style autoregressive language model:
```bash
make train
```

Build the supervised fine-tuning dataset:
```bash
make sft_data
```

Fine-tune the pretrained model on instruction–response data:
```bash
make sft_train
```

Run the test suite:

```bash
make test
```

## Roadmap

The repository will be extended to cover the full LLM development lifecycle, including:

- Preference optimization (RLHF, DPO, ORPO)
- Evaluation and benchmarking
- Safety alignment and red-teaming
- Retrieval-augmented generation (RAG)
- Tool-using agents
- Inference and deployment infrastructure

The emphasis remains on correctness, reproducibility, and systems-level understanding rather than scale.