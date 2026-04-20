from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import regex as re
import yaml

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir
from minigpt.common.text_cleaning import clean_text_structural, normalize_whitespace_final
from minigpt.data.io import iter_jsonl

log = get_logger("minigpt.rag.build")

_TERM_RE = re.compile(r"[\p{L}\p{N}]+", re.UNICODE)


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _tokenize_terms(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TERM_RE.finditer(text)]


def _iter_local_dir(src_cfg: dict) -> Iterator[dict]:
    root = Path(src_cfg["path"])
    patterns = list(src_cfg.get("patterns", ["*.txt", "*.md"]))
    recursive = bool(src_cfg.get("recursive", True))

    seen: set[Path] = set()
    for pattern in patterns:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        for path in sorted(iterator):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            text = path.read_text(encoding="utf-8")
            rel = str(path.relative_to(root))
            title = path.stem.replace("_", " ").strip() or rel
            yield {
                "doc_id": rel,
                "title": title,
                "text": text,
                "source": rel,
            }


def _iter_jsonl_docs(src_cfg: dict) -> Iterator[dict]:
    text_field = str(src_cfg.get("text_field", "text"))
    id_field = str(src_cfg.get("id_field", "id"))
    title_field = str(src_cfg.get("title_field", "title"))
    source_field = str(src_cfg.get("source_field", "source"))

    for idx, row in enumerate(iter_jsonl(src_cfg["path"])):
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        doc_id = str(row.get(id_field) or f"doc_{idx:06d}")
        title = str(row.get(title_field) or doc_id)
        source = str(row.get(source_field) or doc_id)
        yield {
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "source": source,
        }


def _iter_documents(src_cfg: dict) -> Iterator[dict]:
    kind = str(src_cfg.get("kind", "local_dir"))
    if kind == "local_dir":
        yield from _iter_local_dir(src_cfg)
        return
    if kind == "jsonl":
        yield from _iter_jsonl_docs(src_cfg)
        return
    raise RuntimeError(f"Unsupported RAG source kind={kind!r}; use 'local_dir' or 'jsonl'")


def _chunk_text(text: str, chunk_size_words: int, chunk_overlap_words: int) -> list[tuple[str, int, int]]:
    cleaned = clean_text_structural(text)
    words = cleaned.split()
    if not words:
        return []

    stride = max(1, chunk_size_words - chunk_overlap_words)
    out: list[tuple[str, int, int]] = []
    for start in range(0, len(words), stride):
        piece = words[start : start + chunk_size_words]
        if not piece:
            break
        chunk_text = normalize_whitespace_final(" ".join(piece))
        if chunk_text:
            out.append((chunk_text, start, start + len(piece)))
        if start + chunk_size_words >= len(words):
            break
    return out


def build_rag_index(config_path: str) -> None:
    cfg = _load_cfg(config_path)

    paths_cfg = cfg["paths"]
    index_dir = ensure_dir(paths_cfg["index_dir"])

    chunk_cfg = cfg["data"]["chunking"]
    chunk_size_words = int(chunk_cfg.get("chunk_size_words", 120))
    chunk_overlap_words = int(chunk_cfg.get("chunk_overlap_words", 30))
    if chunk_size_words <= 0:
        raise RuntimeError("data.chunking.chunk_size_words must be > 0")
    if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_size_words:
        raise RuntimeError("data.chunking.chunk_overlap_words must be >= 0 and < chunk_size_words")

    retrieval_cfg = cfg.get("retrieval", {})
    k1 = float(retrieval_cfg.get("k1", 1.5))
    b = float(retrieval_cfg.get("b", 0.75))

    chunks_path = Path(index_dir) / "chunks.jsonl"
    index_path = Path(index_dir) / "index.json"

    doc_count = 0
    chunk_count = 0
    total_terms = 0
    doc_freqs: Counter[str] = Counter()

    with open(chunks_path, "w", encoding="utf-8") as fh:
        for doc in _iter_documents(cfg["data"]["source"]):
            doc_count += 1
            doc_id = str(doc["doc_id"])
            title = str(doc["title"])
            source = str(doc["source"])
            for chunk_text, word_start, word_end in _chunk_text(
                str(doc["text"]),
                chunk_size_words=chunk_size_words,
                chunk_overlap_words=chunk_overlap_words,
            ):
                term_freqs = Counter(_tokenize_terms(chunk_text))
                if not term_freqs:
                    continue

                length = int(sum(term_freqs.values()))
                total_terms += length
                chunk_id = f"chunk_{chunk_count:06d}"
                doc_freqs.update(term_freqs.keys())

                record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "source": source,
                    "text": chunk_text,
                    "word_start": int(word_start),
                    "word_end": int(word_end),
                    "length": length,
                    "term_freqs": dict(sorted(term_freqs.items())),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count += 1

    avg_chunk_len = 0.0 if chunk_count == 0 else total_terms / float(chunk_count)
    index_payload = {
        "version": 1,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "avg_chunk_len": avg_chunk_len,
        "doc_freqs": dict(sorted(doc_freqs.items())),
        "build": {
            "config_path": str(config_path),
            "chunk_size_words": chunk_size_words,
            "chunk_overlap_words": chunk_overlap_words,
            "k1": k1,
            "b": b,
        },
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    log.info(
        "Built RAG index docs=%d chunks=%d avg_chunk_len=%.2f -> %s",
        doc_count,
        chunk_count,
        avg_chunk_len,
        str(index_dir),
    )
