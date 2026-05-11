import json
from pathlib import Path

import yaml

from minigpt.eval.run import run_dpo_eval, run_pretrain_eval, run_rag_eval, run_sft_eval


def _write_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def test_run_pretrain_eval_writes_report(tmp_path: Path, monkeypatch):
    cfg = {
        "paths": {"out_dir": str(tmp_path / "eval")},
        "pretrain": {
            "train_config": "configs/train.yaml",
            "ckpt": None,
            "eval_batches": 3,
            "max_new_tokens": 16,
            "temperature": 0.3,
            "top_k": 20,
            "prompts": ["Prompt A", "Prompt B"],
        },
    }
    cfg_path = tmp_path / "eval.yaml"
    _write_yaml(cfg_path, cfg)

    monkeypatch.setattr(
        "minigpt.eval.run._estimate_pretrain_loss",
        lambda train_config_path, ckpt_path, eval_batches: (1.25, 3.490342957, "/tmp/ckpt.pt"),
    )
    monkeypatch.setattr(
        "minigpt.eval.run.sample_text",
        lambda config_path, prompt, ckpt_path, max_new_tokens, temperature, top_k: f"sample::{prompt}",
    )

    out_path = run_pretrain_eval(str(cfg_path))
    report = json.loads(Path(out_path).read_text(encoding="utf-8"))

    assert report["kind"] == "pretrain_eval"
    assert report["val_loss"] == 1.25
    assert len(report["samples"]) == 2
    assert report["samples"][0]["text"] == "sample::Prompt A"


def test_run_sft_eval_writes_report(tmp_path: Path, monkeypatch):
    cfg = {
        "paths": {"out_dir": str(tmp_path / "eval")},
        "sft": {
            "sft_config": "configs/sft.yaml",
            "ckpt": None,
            "eval_batches": 2,
            "max_new_tokens": 16,
            "temperature": 0.3,
            "top_k": 20,
            "instructions": [
                {"instruction": "Prompt A"},
                {"instruction": "Prompt B", "input": "extra"},
            ],
        },
    }
    cfg_path = tmp_path / "eval.yaml"
    _write_yaml(cfg_path, cfg)

    monkeypatch.setattr(
        "minigpt.eval.run._estimate_sft_loss",
        lambda sft_config_path, ckpt_path, eval_batches: (0.75, "/tmp/sft.pt"),
    )
    monkeypatch.setattr(
        "minigpt.eval.run.sft_sample",
        lambda config_path, instruction, inp, ckpt_path, max_new_tokens, temperature, top_k: f"sft::{instruction}::{inp}",
    )

    out_path = run_sft_eval(str(cfg_path))
    report = json.loads(Path(out_path).read_text(encoding="utf-8"))

    assert report["kind"] == "sft_eval"
    assert report["val_loss"] == 0.75
    assert len(report["samples"]) == 2
    assert report["samples"][1]["text"] == "sft::Prompt B::extra"


def test_run_dpo_eval_writes_report(tmp_path: Path, monkeypatch):
    cfg = {
        "paths": {"out_dir": str(tmp_path / "eval")},
        "dpo": {
            "dpo_config": "configs/dpo.yaml",
            "ckpt": None,
            "eval_batches": 2,
            "max_new_tokens": 16,
            "temperature": 0.3,
            "top_k": 20,
            "instructions": [
                {"instruction": "Prompt A"},
                {"instruction": "Prompt B", "input": "extra"},
            ],
        },
    }
    cfg_path = tmp_path / "eval.yaml"
    _write_yaml(cfg_path, cfg)

    monkeypatch.setattr(
        "minigpt.eval.run._estimate_dpo_loss",
        lambda dpo_config_path, ckpt_path, eval_batches: (0.65, "/tmp/dpo.pt", "/tmp/ref.pt"),
    )
    monkeypatch.setattr(
        "minigpt.eval.run.dpo_sample",
        lambda config_path, instruction, inp, ckpt_path, max_new_tokens, temperature, top_k: f"dpo::{instruction}::{inp}",
    )

    out_path = run_dpo_eval(str(cfg_path))
    report = json.loads(Path(out_path).read_text(encoding="utf-8"))

    assert report["kind"] == "dpo_eval"
    assert report["val_dpo_loss"] == 0.65
    assert report["reference_checkpoint"] == "/tmp/ref.pt"
    assert len(report["samples"]) == 2
    assert report["samples"][1]["text"] == "dpo::Prompt B::extra"


def test_run_rag_eval_writes_report(tmp_path: Path, monkeypatch):
    cfg = {
        "paths": {"out_dir": str(tmp_path / "eval")},
        "rag": {
            "rag_config": "configs/rag.yaml",
            "retrieval_top_k": 1,
            "max_new_tokens": 16,
            "temperature": 0.3,
            "top_k": 20,
            "queries": [
                {"question": "What is self-attention?"},
                "What is DPO?",
            ],
        },
    }
    cfg_path = tmp_path / "eval.yaml"
    _write_yaml(cfg_path, cfg)

    def fake_rag_query(
        config_path,
        question,
        retrieval_top_k,
        max_new_tokens,
        temperature,
        top_k,
    ):
        return {
            "text": f"rag::{question}",
            "retrieved": [{"chunk_id": "chunk_000000", "score": 1.0}],
            "generator": {"kind": "sft", "config": config_path, "ckpt": None},
            "report_path": str(tmp_path / "last_query.json"),
        }

    monkeypatch.setattr("minigpt.eval.run.rag_query", fake_rag_query)

    out_path = run_rag_eval(str(cfg_path))
    report = json.loads(Path(out_path).read_text(encoding="utf-8"))

    assert report["kind"] == "rag_eval"
    assert len(report["queries"]) == 2
    assert report["queries"][0]["text"] == "rag::What is self-attention?"
    assert report["queries"][1]["retrieved"][0]["chunk_id"] == "chunk_000000"
