# src/minigpt/tokenizer/corpus.py

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from minigpt.data.io import iter_jsonl


def iter_texts(input_dir: str) -> Iterator[str]:
    p = Path(input_dir)
    files = sorted(p.glob("*.jsonl"))
    for fp in files:
        for rec in iter_jsonl(fp):
            t = rec.get("text", "")
            if t:
                yield t
