import json
from pathlib import Path

import yaml

from minigpt.rag.build import build_rag_index
from minigpt.rag.query import rag_query
from minigpt.rag.retrieve import retrieve_chunks


def _write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def test_build_rag_index_and_retrieve(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "transformers.txt").write_text(
        "Transformers use self-attention to connect tokens across a sequence. "
        "Attention lets each token weigh other tokens when building representations.",
        encoding="utf-8",
    )
    (docs_dir / "gardening.txt").write_text(
        "Tomatoes need sun, water, and nutrient-rich soil. "
        "Pruning helps airflow in a garden bed.",
        encoding="utf-8",
    )

    cfg = {
        "paths": {"index_dir": str(tmp_path / "rag_index")},
        "data": {
            "source": {
                "kind": "local_dir",
                "path": str(docs_dir),
                "patterns": ["*.txt"],
                "recursive": False,
            },
            "chunking": {
                "chunk_size_words": 12,
                "chunk_overlap_words": 3,
            },
        },
        "retrieval": {
            "top_k": 2,
            "k1": 1.5,
            "b": 0.75,
        },
        "generator": {
            "kind": "sft",
            "config": "configs/sft.yaml",
            "ckpt": None,
        },
        "prompt": {
            "instruction": "Answer using context only.",
            "context_header": "Retrieved context",
        },
        "generation": {
            "max_new_tokens": 32,
            "temperature": 0.3,
            "top_k": 20,
        },
    }
    cfg_path = tmp_path / "rag.yaml"
    _write_yaml(cfg_path, cfg)

    build_rag_index(str(cfg_path))

    index = json.loads((tmp_path / "rag_index" / "index.json").read_text(encoding="utf-8"))
    assert index["doc_count"] == 2
    assert index["chunk_count"] >= 2
    assert "attention" in index["doc_freqs"]

    results = retrieve_chunks(str(cfg_path), query="How does self attention work?", top_k=2)
    assert len(results) >= 1
    assert results[0]["doc_id"] == "transformers.txt"
    assert "self-attention" in results[0]["text"]


def test_rag_query_builds_grounded_prompt_and_saves_report(tmp_path: Path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "moon.txt").write_text(
        "The moon influences tides because its gravity pulls on Earth's oceans.",
        encoding="utf-8",
    )

    cfg = {
        "paths": {"index_dir": str(tmp_path / "rag_index")},
        "data": {
            "source": {
                "kind": "local_dir",
                "path": str(docs_dir),
                "patterns": ["*.txt"],
                "recursive": False,
            },
            "chunking": {
                "chunk_size_words": 20,
                "chunk_overlap_words": 5,
            },
        },
        "retrieval": {
            "top_k": 1,
            "k1": 1.5,
            "b": 0.75,
        },
        "generator": {
            "kind": "sft",
            "config": "configs/sft.yaml",
            "ckpt": None,
        },
        "prompt": {
            "instruction": "Answer using only the retrieved context.",
            "context_header": "Retrieved context",
        },
        "generation": {
            "max_new_tokens": 24,
            "temperature": 0.2,
            "top_k": 10,
        },
    }
    cfg_path = tmp_path / "rag.yaml"
    _write_yaml(cfg_path, cfg)
    build_rag_index(str(cfg_path))

    seen = {}

    def fake_sft_sample(config_path, instruction, inp, ckpt_path, max_new_tokens, temperature, top_k):
        seen["config_path"] = config_path
        seen["instruction"] = instruction
        seen["input"] = inp
        seen["ckpt_path"] = ckpt_path
        seen["max_new_tokens"] = max_new_tokens
        seen["temperature"] = temperature
        seen["top_k"] = top_k
        return "The moon affects tides. [chunk_000000]"

    monkeypatch.setattr("minigpt.rag.query.sft_sample", fake_sft_sample)

    report = rag_query(str(cfg_path), question="Why does the moon affect tides?")

    assert report["text"] == "The moon affects tides. [chunk_000000]"
    assert seen["instruction"] == "Answer using only the retrieved context."
    assert "Question:\nWhy does the moon affect tides?" in seen["input"]
    assert "chunk_id=chunk_000000" in seen["input"]
    assert Path(report["report_path"]).exists()

    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["retrieved"][0]["doc_id"] == "moon.txt"
    assert saved["generator"]["kind"] == "sft"
