# src/minigpt/tokenizer/train.py

from __future__ import annotations

from pathlib import Path

import yaml
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir
from minigpt.tokenizer.corpus import iter_texts

log = get_logger("minigpt.tokenizer.train")


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_tokenizer(config_path: str) -> None:
    cfg = _load_cfg(config_path)

    seed = int(cfg.get("seed", 1337))
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]

    vocab_size = int(cfg.get("vocab_size", 32000))
    min_frequency = int(cfg.get("min_frequency", 2))
    limit_alphabet = int(cfg.get("limit_alphabet", 1000))
    special_tokens = list(cfg.get("special_tokens", ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]))

    out = ensure_dir(output_dir)

    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        limit_alphabet=limit_alphabet,
    )

    log.info("Training tokenizer on corpus at %s", input_dir)
    tok.train_from_iterator(iter_texts(input_dir), trainer=trainer, length=None)

    tok_json = Path(out) / "tokenizer.json"
    tok.save(str(tok_json))

    meta = {
        "seed": seed,
        "input_dir": input_dir,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "limit_alphabet": limit_alphabet,
        "special_tokens": special_tokens,
    }
    meta_path = Path(out) / "tokenizer_meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    log.info("Saved tokenizer to %s", str(tok_json))
    log.info("Saved metadata to %s", str(meta_path))
