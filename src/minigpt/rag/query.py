from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from minigpt.common.paths import ensure_dir
from minigpt.dpo.sample import dpo_sample
from minigpt.rag.retrieve import retrieve_chunks
from minigpt.sft.sample import sft_sample


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_context(chunks: list[dict[str, Any]], header: str) -> str:
    if not chunks:
        return f"{header}:\n(no retrieved chunks)"

    parts = [f"{header}:"]
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            (
                f"[{idx}] chunk_id={chunk['chunk_id']} doc_id={chunk['doc_id']} "
                f"score={chunk['score']:.3f}\n{chunk['text']}"
            )
        )
    return "\n\n".join(parts)


def _generate(
    generator_cfg: dict,
    instruction: str,
    inp: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> str:
    kind = str(generator_cfg.get("kind", "sft"))
    config_path = str(generator_cfg["config"])
    ckpt = generator_cfg.get("ckpt")
    if kind == "sft":
        return sft_sample(
            config_path=config_path,
            instruction=instruction,
            inp=inp,
            ckpt_path=None if ckpt is None else str(ckpt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    if kind == "dpo":
        return dpo_sample(
            config_path=config_path,
            instruction=instruction,
            inp=inp,
            ckpt_path=None if ckpt is None else str(ckpt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    raise RuntimeError(f"Unsupported generator kind={kind!r}; use 'sft' or 'dpo'")


def rag_query(
    config_path: str,
    question: str,
    retrieval_top_k: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    prompt_cfg = cfg.get("prompt", {})
    generation_cfg = cfg.get("generation", {})

    retrieved = retrieve_chunks(config_path=config_path, query=question, top_k=retrieval_top_k)

    instruction = str(
        prompt_cfg.get(
            "instruction",
            "Answer the question using only the retrieved context. If the context is insufficient, say you do not know.",
        )
    )
    context_header = str(prompt_cfg.get("context_header", "Retrieved context"))
    inp = f"Question:\n{question}\n\n{_format_context(retrieved, context_header)}"

    out_text = _generate(
        generator_cfg=cfg["generator"],
        instruction=instruction,
        inp=inp,
        max_new_tokens=int(max_new_tokens if max_new_tokens is not None else generation_cfg.get("max_new_tokens", 160)),
        temperature=float(temperature if temperature is not None else generation_cfg.get("temperature", 0.3)),
        top_k=top_k if top_k is not None else generation_cfg.get("top_k", 20),
    )

    report = {
        "question": question,
        "instruction": instruction,
        "input": inp,
        "retrieved": retrieved,
        "generator": {
            "kind": str(cfg["generator"]["kind"]),
            "config": str(cfg["generator"]["config"]),
            "ckpt": cfg["generator"].get("ckpt"),
        },
        "text": out_text,
    }

    out_dir = ensure_dir(cfg["paths"]["index_dir"])
    report_path = Path(out_dir) / "last_query.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    report["report_path"] = str(report_path)
    return report
