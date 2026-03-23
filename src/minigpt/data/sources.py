# src/minigpt/data/sources.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from datasets import load_dataset


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str  # "hf" only for now
    dataset: str
    subset: Optional[str]
    split: str
    text_field: str
    weight: float


def stream_hf_text(spec: SourceSpec) -> Iterator[dict]:
    if spec.kind != "hf":
        raise ValueError(f"Unsupported source kind={spec.kind!r}; only 'hf' is implemented")

    ds = load_dataset(spec.dataset, spec.subset, split=spec.split, streaming=True)
    for ex in ds:
        text = ex.get(spec.text_field, None)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        yield {
            "text": text,
            "source": spec.name,
            "meta": {},
        }
