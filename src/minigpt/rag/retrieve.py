from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import regex as re
import yaml

from minigpt.data.io import iter_jsonl

_TERM_RE = re.compile(r"[\p{L}\p{N}]+", re.UNICODE)


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tokenize_terms(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TERM_RE.finditer(text)]


def _load_index(index_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(index_dir)
    with open(root / "index.json", "r", encoding="utf-8") as f:
        index = json.load(f)
    chunks = list(iter_jsonl(root / "chunks.jsonl"))
    return index, chunks


def _bm25_score(
    query_tf: Counter[str],
    term_freqs: dict[str, int],
    doc_len: int,
    num_chunks: int,
    avg_chunk_len: float,
    doc_freqs: dict[str, int],
    k1: float,
    b: float,
) -> float:
    if doc_len <= 0 or not query_tf:
        return 0.0

    score = 0.0
    norm = k1 * (1.0 - b + b * (doc_len / max(avg_chunk_len, 1e-8)))
    for term, qfreq in query_tf.items():
        tf = int(term_freqs.get(term, 0))
        if tf <= 0:
            continue
        df = int(doc_freqs.get(term, 0))
        idf = math.log(1.0 + ((num_chunks - df + 0.5) / (df + 0.5)))
        score += float(qfreq) * idf * ((tf * (k1 + 1.0)) / (tf + norm))
    return score


def retrieve_chunks(config_path: str, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
    cfg = _load_cfg(config_path)
    retrieval_cfg = cfg.get("retrieval", {})
    k1 = float(retrieval_cfg.get("k1", 1.5))
    b = float(retrieval_cfg.get("b", 0.75))
    top_k = int(top_k if top_k is not None else retrieval_cfg.get("top_k", 3))

    index, chunks = _load_index(cfg["paths"]["index_dir"])
    num_chunks = int(index.get("chunk_count", len(chunks)))
    avg_chunk_len = float(index.get("avg_chunk_len", 1.0))
    doc_freqs = {str(k): int(v) for k, v in index.get("doc_freqs", {}).items()}

    query_terms = Counter(_tokenize_terms(query))
    if not query_terms:
        return []

    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        score = _bm25_score(
            query_tf=query_terms,
            term_freqs={str(k): int(v) for k, v in chunk.get("term_freqs", {}).items()},
            doc_len=int(chunk.get("length", 0)),
            num_chunks=num_chunks,
            avg_chunk_len=avg_chunk_len,
            doc_freqs=doc_freqs,
            k1=k1,
            b=b,
        )
        if score <= 0.0:
            continue
        scored.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "doc_id": str(chunk["doc_id"]),
                "title": str(chunk.get("title", chunk["doc_id"])),
                "source": str(chunk.get("source", chunk["doc_id"])),
                "score": float(score),
                "text": str(chunk["text"]),
                "word_start": int(chunk.get("word_start", 0)),
                "word_end": int(chunk.get("word_end", 0)),
            }
        )

    scored.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    return scored[:top_k]
