# src/minigpt/tokenizer/analyze.py

from __future__ import annotations

from pathlib import Path

import yaml
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.tokenizer.corpus import iter_texts

log = get_logger("minigpt.tokenizer.analyze")


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def analyze_tokenizer(config_path: str) -> None:
    cfg = _load_cfg(config_path)

    input_dir = cfg["input_dir"]
    tok_dir = cfg["output_dir"]

    tok_path = Path(tok_dir) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))

    n_docs = 0
    n_chars = 0
    n_tokens = 0

    max_docs = int(cfg.get("analyze_max_docs", 2000))

    for t in iter_texts(input_dir):
        enc = tok.encode(t)
        n_docs += 1
        n_chars += len(t)
        n_tokens += len(enc.ids)
        if n_docs >= max_docs:
            break

    avg_tok_per_char = (float(n_tokens) / float(n_chars)) if n_chars else 0.0
    avg_chars_per_tok = (float(n_chars) / float(n_tokens)) if n_tokens else 0.0

    info = {
        "docs_analyzed": n_docs,
        "total_chars": n_chars,
        "total_tokens": n_tokens,
        "avg_tokens_per_char": avg_tok_per_char,
        "avg_chars_per_token": avg_chars_per_tok,
        "vocab_size": tok.get_vocab_size(),
    }

    out_path = Path(tok_dir) / "analysis.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, sort_keys=False, allow_unicode=True)

    log.info("Tokenizer vocab size: %d", tok.get_vocab_size())
    log.info("Avg chars/token: %.4f", avg_chars_per_tok)
    log.info("Avg tokens/char: %.6f", avg_tok_per_char)
    log.info("Wrote analysis to %s", str(out_path))
