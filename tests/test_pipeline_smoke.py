import json
from pathlib import Path

import torch
import yaml

from minigpt.data.pipeline import build_corpus
from minigpt.sft.build import build_sft_tokens
from minigpt.sft.data import SFTTokenDataset
from minigpt.tokenizer.encode import encode_corpus
from minigpt.tokenizer.train import train_tokenizer
from minigpt.train.data import TokenShardDataset


class FakeMapDataset:
    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def _write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _count_docs(tokens_dir: Path, pattern: str) -> int:
    total = 0
    for idx_path in sorted(tokens_dir.glob(pattern)):
        with open(idx_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        total += len(info["doc_starts"])
    return total


def test_end_to_end_data_and_sft_smoke(tmp_path: Path, monkeypatch):
    corpus_texts = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        "one two three four five six seven eight nine ten eleven twelve",
        "red blue green yellow orange purple cyan magenta black white silver gold",
        "cat dog bird fish horse zebra whale dolphin otter tiger lion bear",
    ]
    sft_rows = [
        {"instruction": "Say hi", "input": "", "output": "Hello there."},
        {"instruction": "Count", "input": "to three", "output": "One two three."},
        {"instruction": "Answer", "input": "", "output": "This is a complete answer."},
        {"instruction": "Explain", "input": "gravity", "output": "Gravity attracts mass."},
        {"instruction": "Summarize", "input": "short text", "output": "A short summary."},
    ]

    def fake_streaming_load_dataset(dataset, subset, split, streaming=False):
        if dataset == "demo-corpus":
            assert streaming is True
            return iter({"text": text} for text in corpus_texts)
        if dataset == "demo-sft":
            assert streaming is False
            return FakeMapDataset(sft_rows)
        raise AssertionError(f"unexpected dataset: {dataset}")

    monkeypatch.setattr("minigpt.data.sources.load_dataset", fake_streaming_load_dataset)
    monkeypatch.setattr("minigpt.sft.build.load_dataset", fake_streaming_load_dataset)

    corpus_dir = tmp_path / "step1_corpus"
    tok_dir = tmp_path / "tokenizer"
    tokens_dir = tmp_path / "step2_tokens"
    sft_tokens_dir = tmp_path / "step4_sft_tokens"

    data_cfg = {
        "output_dir": str(corpus_dir),
        "seed": 1337,
        "max_examples_total": 4,
        "shard_size": 2,
        "filters": {
            "min_chars": 10,
            "max_chars": 1000,
            "max_nonalpha_ratio": 0.9,
            "max_repeated_line_ratio": 1.0,
        },
        "dedupe": {"exact": True, "simhash": {"enabled": False}},
        "sources": [
            {
                "name": "demo",
                "kind": "hf",
                "dataset": "demo-corpus",
                "subset": None,
                "split": "train",
                "text_field": "text",
                "weight": 1.0,
            }
        ],
    }
    tok_cfg = {
        "seed": 1337,
        "input_dir": str(corpus_dir),
        "output_dir": str(tok_dir),
        "vocab_size": 128,
        "min_frequency": 1,
        "limit_alphabet": 256,
        "special_tokens": ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
        "encode": {
            "enabled": True,
            "output_dir": str(tokens_dir),
            "shard_prefix": "tok",
        },
    }
    sft_cfg = {
        "seed": 1337,
        "paths": {
            "tokenizer_dir": str(tok_dir),
            "out_dir": str(tmp_path / "unused_sft_ckpts"),
            "sft_tokens_dir": str(sft_tokens_dir),
            "init_ckpt": str(tmp_path / "unused_pretrain_ckpts"),
        },
        "data": {
            "shard_prefix": "sft",
            "shard_size": 4,
            "max_examples_total": 4,
            "val_ratio": 0.25,
            "min_resp_tokens": 2,
            "template": {
                "system": "",
                "user_prefix": "### Instruction:\n",
                "assistant_prefix": "\n\n### Response:\n",
                "end": "\n",
            },
            "source": {
                "kind": "hf",
                "dataset": "demo-sft",
                "subset": None,
                "split": "train",
                "instruction_field": "instruction",
                "input_field": "input",
                "output_field": "output",
            },
        },
        "tokenization": {"block_size": 64},
        "model": {"n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.0},
        "train": {
            "max_steps": 1,
            "grad_accum": 1,
            "batch_size": 2,
            "lr": 1.0e-4,
            "min_lr": 1.0e-5,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "eval_interval": 1,
            "eval_batches": 1,
            "save_interval": 1,
            "log_interval": 1,
            "device": "cpu",
            "dtype": "fp32",
        },
    }

    data_cfg_path = tmp_path / "data.yaml"
    tok_cfg_path = tmp_path / "tokenizer.yaml"
    sft_cfg_path = tmp_path / "sft.yaml"
    _write_yaml(data_cfg_path, data_cfg)
    _write_yaml(tok_cfg_path, tok_cfg)
    _write_yaml(sft_cfg_path, sft_cfg)

    build_corpus(str(data_cfg_path))
    train_tokenizer(str(tok_cfg_path))
    encode_corpus(str(tok_cfg_path))

    pretrain_ds = TokenShardDataset(
        tokens_dir=str(tokens_dir),
        shard_prefix="tok",
        block_size=8,
        val_num_shards=1,
    )
    x, y = pretrain_ds.get_batch("train", batch_size=2, device=torch.device("cpu"))
    assert x.shape == (2, 8)
    assert y.shape == (2, 8)

    build_sft_tokens(str(sft_cfg_path))

    total_sft_docs = _count_docs(sft_tokens_dir, "*_sft_*.idx.json")
    assert total_sft_docs == 4

    sft_ds = SFTTokenDataset(
        tokens_dir=str(sft_tokens_dir),
        shard_prefix="sft",
        block_size=64,
        min_resp_tokens=2,
    )
    sx, sy, sm = sft_ds.get_batch("train", batch_size=1, device=torch.device("cpu"))
    assert sx.shape == (1, 64)
    assert sy.shape == (1, 64)
    assert sm.shape == (1, 64)
    assert bool(sm.any().item()) is True
