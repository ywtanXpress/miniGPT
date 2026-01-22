# src/minigpt/tokenizer/encode.py

from __future__ import annotations

import json
from array import array
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir
from minigpt.data.io import iter_jsonl

log = get_logger("minigpt.tokenizer.encode")


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_corpus(config_path: str) -> None:
    cfg = _load_cfg(config_path)

    input_dir = cfg["input_dir"]
    tok_dir = cfg["output_dir"]
    enc_cfg = cfg.get("encode", {})

    if not enc_cfg.get("enabled", False):
        log.info("encode.enabled is false; skipping.")
        return

    out_dir = ensure_dir(enc_cfg.get("output_dir", "data/processed/step2_tokens"))
    shard_prefix = enc_cfg.get("shard_prefix", "tok")

    tok_path = Path(tok_dir) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))

    in_path = Path(input_dir)
    files = sorted(in_path.glob("*.jsonl"))

    ensure_dir(out_dir)

    for si, fp in enumerate(files):
        bin_path = Path(out_dir) / f"{shard_prefix}_{si:05d}.bin"
        idx_path = Path(out_dir) / f"{shard_prefix}_{si:05d}.idx.json"

        token_ids = array("I")
        doc_starts = []
        n_docs = 0

        for rec in iter_jsonl(fp):
            text = rec.get("text", "")
            if not text:
                continue

            doc_starts.append(len(token_ids))
            ids = tok.encode(text).ids
            token_ids.extend(ids)
            n_docs += 1

        with open(bin_path, "wb") as f:
            token_ids.tofile(f)

        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_shard": fp.name,
                    "num_docs": n_docs,
                    "num_tokens": len(token_ids),
                    "doc_starts": doc_starts,
                },
                f,
                ensure_ascii=False,
            )

        log.info("Encoded %s -> %s (%d docs, %d tokens)", fp.name, bin_path.name, n_docs, len(token_ids))
