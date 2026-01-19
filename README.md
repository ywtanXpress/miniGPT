# miniGPT

A laptop-scale, end-to-end practice repository that walks through the lifecycle of a modern LLM.
We will implement Steps 1–13 in increasing depth.

## Step 1 (implemented)
Data pipeline:
- stream/gather datasets
- clean + filter
- deduplicate (exact + SimHash near-dup)
- organize + balance mixture across sources
- shard to disk (JSONL)

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[dev]"
python scripts/mgpt.py data build --config configs/data.yaml
