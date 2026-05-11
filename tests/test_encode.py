import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from minigpt.tokenizer.encode import encode_corpus


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _train_test_tokenizer(out_dir: Path) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<|unk|>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=128,
        min_frequency=1,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    )
    tok.train_from_iterator(["hello world", "goodbye world"], trainer=trainer)
    tok.save(str(out_dir / "tokenizer.json"))
    return tok


def test_encode_corpus_appends_eos(tmp_path: Path):
    input_dir = tmp_path / "corpus"
    tok_dir = tmp_path / "tok"
    out_dir = tmp_path / "tokens"
    input_dir.mkdir()
    tok_dir.mkdir()

    _write_jsonl(input_dir / "step1_00000.jsonl", [{"text": "hello world"}])
    tok = _train_test_tokenizer(tok_dir)
    eos_id = tok.token_to_id("<|eos|>")
    assert eos_id is not None

    cfg_path = tmp_path / "tokenizer.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'seed: 1337',
                f'input_dir: "{input_dir}"',
                f'output_dir: "{tok_dir}"',
                "encode:",
                "  enabled: true",
                f'  output_dir: "{out_dir}"',
                '  shard_prefix: "tok"',
            ]
        ),
        encoding="utf-8",
    )

    encode_corpus(str(cfg_path))

    arr = np.fromfile(out_dir / "tok_00000.bin", dtype=np.uint32)
    assert int(arr[-1]) == int(eos_id)
