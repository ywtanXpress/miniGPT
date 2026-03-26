import json
from pathlib import Path

import torch
import yaml

from minigpt.dpo.build import build_dpo_pairs
from minigpt.dpo.data import DPOPairDataset
from minigpt.dpo.train import _resolve_ckpt
from minigpt.dpo.train import _dpo_loss
from minigpt.tokenizer.train import train_tokenizer
from minigpt.data.pipeline import build_corpus


def _write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_dpo_loss_prefers_positive_margin():
    better = _dpo_loss(
        policy_chosen=torch.tensor([3.0]),
        policy_rejected=torch.tensor([1.0]),
        ref_chosen=torch.tensor([2.0]),
        ref_rejected=torch.tensor([1.5]),
        beta=0.1,
    )
    worse = _dpo_loss(
        policy_chosen=torch.tensor([1.0]),
        policy_rejected=torch.tensor([3.0]),
        ref_chosen=torch.tensor([2.0]),
        ref_rejected=torch.tensor([1.5]),
        beta=0.1,
    )
    assert better.item() < worse.item()


def test_resolve_reference_ckpt_defaults_to_init_when_null():
    init_ckpt = "/tmp/init_ckpt.pt"
    reference_ckpt_cfg = None
    resolved = init_ckpt if reference_ckpt_cfg is None else _resolve_ckpt(str(reference_ckpt_cfg))
    assert resolved == init_ckpt


def test_build_dpo_pairs_and_dataset_smoke(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    tok_dir = tmp_path / "tok"
    pref_path = tmp_path / "prefs.jsonl"
    out_dir = tmp_path / "dpo_pairs"

    corpus_rows = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        "one two three four five six seven eight nine ten eleven twelve",
    ]

    def fake_load_dataset(dataset, subset, split, streaming=False):
        assert dataset == "demo"
        assert streaming is True
        return iter({"text": text} for text in corpus_rows)

    monkeypatch.setattr("minigpt.data.sources.load_dataset", fake_load_dataset)

    data_cfg = {
        "output_dir": str(corpus_dir),
        "seed": 1337,
        "max_examples_total": 2,
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
                "dataset": "demo",
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
    }

    prefs = [
        {
            "instruction": "Explain transformers",
            "input": "",
            "chosen": "Transformers use self-attention to model token relationships.",
            "rejected": "Transformers are a type of toaster used in kitchens.",
        },
        {
            "instruction": "Explain batching",
            "input": "",
            "chosen": "Batching groups examples so training uses vectorized computation.",
            "rejected": "Batching means deleting all data before training.",
        },
    ]

    data_cfg_path = tmp_path / "data.yaml"
    tok_cfg_path = tmp_path / "tok.yaml"
    dpo_cfg_path = tmp_path / "dpo.yaml"
    _write_yaml(data_cfg_path, data_cfg)
    _write_yaml(tok_cfg_path, tok_cfg)
    _write_jsonl(pref_path, prefs)

    dpo_cfg = {
        "seed": 1337,
        "paths": {
            "tokenizer_dir": str(tok_dir),
            "preference_dir": str(out_dir),
            "out_dir": str(tmp_path / "unused"),
            "init_ckpt": str(tmp_path / "unused"),
            "reference_ckpt": None,
        },
        "data": {
            "shard_prefix": "dpo",
            "shard_size": 2,
            "max_examples_total": 2,
            "val_ratio": 0.5,
            "min_resp_tokens": 2,
            "template": {
                "system": "",
                "user_prefix": "### Instruction:\n",
                "assistant_prefix": "\n\n### Response:\n",
                "end": "\n",
            },
            "source": {
                "kind": "jsonl",
                "path": str(pref_path),
                "instruction_field": "instruction",
                "input_field": "input",
                "chosen_field": "chosen",
                "rejected_field": "rejected",
            },
        },
        "tokenization": {"block_size": 64},
        "model": {"n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.0},
        "train": {
            "max_steps": 1,
            "grad_accum": 1,
            "batch_size": 1,
            "lr": 1.0e-6,
            "min_lr": 1.0e-7,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "max_grad_norm": 1.0,
            "beta": 0.1,
            "eval_interval": 1,
            "eval_batches": 1,
            "save_interval": 1,
            "log_interval": 1,
            "device": "cpu",
            "dtype": "fp32",
        },
    }
    _write_yaml(dpo_cfg_path, dpo_cfg)

    build_corpus(str(data_cfg_path))
    train_tokenizer(str(tok_cfg_path))
    build_dpo_pairs(str(dpo_cfg_path))

    ds = DPOPairDataset(preference_dir=str(out_dir), shard_prefix="dpo", block_size=64, min_resp_tokens=2)
    batch = ds.get_batch("train", batch_size=1, device=torch.device("cpu"))
    assert len(batch) == 6
    assert batch[0].shape == (1, 64)
    assert batch[2].shape == (1, 64)
