from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir
from minigpt.dpo.data import DPOPairDataset
from minigpt.dpo.sample import dpo_sample
from minigpt.dpo.train import _dpo_loss, _masked_logprob_sum
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.rag.query import rag_query
from minigpt.sft.data import SFTTokenDataset
from minigpt.sft.sample import sft_sample
from minigpt.train.ckpt import find_latest_checkpoint, load_checkpoint
from minigpt.train.data import TokenShardDataset
from minigpt.train.sample import sample_text

log = get_logger("minigpt.eval.run")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return str(path)


def _resolve_ckpt(out_dir: str, ckpt_path: Optional[str]) -> str:
    if ckpt_path:
        return str(ckpt_path)
    return find_latest_checkpoint(out_dir)


def _resolve_file_or_latest(path_or_dir: str) -> str:
    p = Path(path_or_dir)
    if p.is_dir():
        return find_latest_checkpoint(str(p))
    return str(p)


def _select_device(device_cfg: str) -> torch.device:
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _build_model(config_path: str) -> tuple[dict, GPT, torch.device]:
    cfg = _load_yaml(config_path)
    train_cfg = cfg.get("train", {})
    device = _select_device(str(train_cfg.get("device", "auto")))

    tok_path = Path(cfg["paths"]["tokenizer_dir"]) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    vocab_size = int(tok.get_vocab_size())

    block_size = int(cfg.get("data", {}).get("block_size", cfg.get("tokenization", {}).get("block_size")))
    model_cfg = cfg["model"]
    gcfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=int(model_cfg["n_layer"]),
        n_head=int(model_cfg["n_head"]),
        n_embd=int(model_cfg["n_embd"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model = GPT(gcfg).to(device)
    return cfg, model, device


@torch.no_grad()
def _estimate_pretrain_loss(train_config_path: str, ckpt_path: Optional[str], eval_batches: int) -> tuple[float, float, str]:
    cfg, model, device = _build_model(train_config_path)
    resolved_ckpt = _resolve_ckpt(cfg["paths"]["out_dir"], ckpt_path)
    load_checkpoint(resolved_ckpt, model, opt=None)

    data_cfg = cfg["data"]
    ds = TokenShardDataset(
        tokens_dir=cfg["paths"]["tokens_dir"],
        shard_prefix=str(data_cfg.get("shard_prefix", "tok")),
        block_size=int(data_cfg["block_size"]),
        val_num_shards=int(data_cfg.get("val_num_shards", 1)),
    )
    batch_size = int(data_cfg["batch_size"])

    model.eval()
    losses: list[float] = []
    for _ in range(int(eval_batches)):
        x, y = ds.get_batch("val", batch_size=batch_size, device=device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))

    val_loss = sum(losses) / float(len(losses))
    perplexity = float(math.exp(val_loss))
    return val_loss, perplexity, resolved_ckpt


def _masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, vocab = logits.shape
    logits2 = logits.reshape(bsz * seq_len, vocab)
    targets2 = targets.reshape(bsz * seq_len)
    mask2 = loss_mask.reshape(bsz * seq_len)
    ignore = torch.full_like(targets2, -100)
    targets_m = torch.where(mask2, targets2, ignore)
    return F.cross_entropy(logits2, targets_m, ignore_index=-100)


@torch.no_grad()
def _estimate_sft_loss(sft_config_path: str, ckpt_path: Optional[str], eval_batches: int) -> tuple[float, str]:
    cfg, model, device = _build_model(sft_config_path)
    resolved_ckpt = _resolve_ckpt(cfg["paths"]["out_dir"], ckpt_path)
    load_checkpoint(resolved_ckpt, model, opt=None)

    ds = SFTTokenDataset(
        tokens_dir=cfg["paths"]["sft_tokens_dir"],
        shard_prefix=cfg["data"]["shard_prefix"],
        block_size=int(cfg["tokenization"]["block_size"]),
        min_resp_tokens=int(cfg["data"].get("min_resp_tokens", 16)),
    )
    batch_size = int(cfg["train"]["batch_size"])

    model.eval()
    losses: list[float] = []
    for _ in range(int(eval_batches)):
        x, y, m = ds.get_batch("val", batch_size=batch_size, device=device)
        logits, _ = model(x, targets=None)
        loss = _masked_ce_loss(logits, y, m)
        losses.append(float(loss.item()))

    val_loss = sum(losses) / float(len(losses))
    return val_loss, resolved_ckpt


@torch.no_grad()
def _estimate_dpo_loss(dpo_config_path: str, ckpt_path: Optional[str], eval_batches: int) -> tuple[float, str, str]:
    cfg, policy, device = _build_model(dpo_config_path)
    _, reference, _ = _build_model(dpo_config_path)

    paths = cfg["paths"]
    policy_ckpt = _resolve_ckpt(paths["out_dir"], ckpt_path)
    ref_cfg = paths.get("reference_ckpt")
    reference_ckpt = _resolve_file_or_latest(str(paths["init_ckpt"] if ref_cfg is None else ref_cfg))

    load_checkpoint(policy_ckpt, policy, opt=None)
    load_checkpoint(reference_ckpt, reference, opt=None)

    ds = DPOPairDataset(
        preference_dir=paths["preference_dir"],
        shard_prefix=cfg["data"]["shard_prefix"],
        block_size=int(cfg["tokenization"]["block_size"]),
        min_resp_tokens=int(cfg["data"].get("min_resp_tokens", 16)),
    )
    batch_size = int(cfg["train"]["batch_size"])
    beta = float(cfg["train"].get("beta", 0.1))

    policy.eval()
    reference.eval()
    losses: list[float] = []
    for _ in range(int(eval_batches)):
        cx, cy, cm, rx, ry, rm = ds.get_batch("val", batch_size=batch_size, device=device)
        policy_c, _ = policy(cx, targets=None)
        policy_r, _ = policy(rx, targets=None)
        ref_c, _ = reference(cx, targets=None)
        ref_r, _ = reference(rx, targets=None)
        loss = _dpo_loss(
            _masked_logprob_sum(policy_c, cy, cm),
            _masked_logprob_sum(policy_r, ry, rm),
            _masked_logprob_sum(ref_c, cy, cm),
            _masked_logprob_sum(ref_r, ry, rm),
            beta=beta,
        )
        losses.append(float(loss.item()))

    val_loss = sum(losses) / float(len(losses))
    return val_loss, policy_ckpt, reference_ckpt


def run_pretrain_eval(config_path: str) -> str:
    cfg = _load_yaml(config_path)
    out_dir = ensure_dir(cfg["paths"]["out_dir"])
    pre_cfg = cfg["pretrain"]

    val_loss, perplexity, ckpt_path = _estimate_pretrain_loss(
        train_config_path=str(pre_cfg["train_config"]),
        ckpt_path=pre_cfg.get("ckpt"),
        eval_batches=int(pre_cfg.get("eval_batches", 20)),
    )

    prompts = list(pre_cfg.get("prompts", []))
    if not prompts:
        raise RuntimeError("eval.pretrain.prompts must contain at least one prompt")

    samples = []
    for prompt in prompts:
        text = sample_text(
            config_path=str(pre_cfg["train_config"]),
            prompt=str(prompt),
            ckpt_path=pre_cfg.get("ckpt"),
            max_new_tokens=int(pre_cfg.get("max_new_tokens", 120)),
            temperature=float(pre_cfg.get("temperature", 0.3)),
            top_k=int(pre_cfg.get("top_k", 20)),
        )
        samples.append({"prompt": str(prompt), "text": text})

    report = {
        "kind": "pretrain_eval",
        "checkpoint": ckpt_path,
        "train_config": str(pre_cfg["train_config"]),
        "val_loss": val_loss,
        "perplexity": perplexity,
        "num_prompts": len(samples),
        "samples": samples,
    }
    out_path = Path(out_dir) / "pretrain_eval.json"
    saved = _write_json(out_path, report)
    log.info("Wrote pretrain eval report to %s", saved)
    return saved


def run_sft_eval(config_path: str) -> str:
    cfg = _load_yaml(config_path)
    out_dir = ensure_dir(cfg["paths"]["out_dir"])
    sft_cfg = cfg["sft"]

    val_loss, ckpt_path = _estimate_sft_loss(
        sft_config_path=str(sft_cfg["sft_config"]),
        ckpt_path=sft_cfg.get("ckpt"),
        eval_batches=int(sft_cfg.get("eval_batches", 20)),
    )

    instructions = list(sft_cfg.get("instructions", []))
    if not instructions:
        raise RuntimeError("eval.sft.instructions must contain at least one instruction item")

    samples = []
    for item in instructions:
        instruction = str(item["instruction"])
        inp = item.get("input")
        text = sft_sample(
            config_path=str(sft_cfg["sft_config"]),
            instruction=instruction,
            inp=None if inp is None else str(inp),
            ckpt_path=sft_cfg.get("ckpt"),
            max_new_tokens=int(sft_cfg.get("max_new_tokens", 120)),
            temperature=float(sft_cfg.get("temperature", 0.3)),
            top_k=int(sft_cfg.get("top_k", 20)),
        )
        samples.append(
            {
                "instruction": instruction,
                "input": None if inp is None else str(inp),
                "text": text,
            }
        )

    report = {
        "kind": "sft_eval",
        "checkpoint": ckpt_path,
        "sft_config": str(sft_cfg["sft_config"]),
        "val_loss": val_loss,
        "num_instructions": len(samples),
        "samples": samples,
    }
    out_path = Path(out_dir) / "sft_eval.json"
    saved = _write_json(out_path, report)
    log.info("Wrote SFT eval report to %s", saved)
    return saved


def run_dpo_eval(config_path: str) -> str:
    cfg = _load_yaml(config_path)
    out_dir = ensure_dir(cfg["paths"]["out_dir"])
    dpo_cfg = cfg["dpo"]

    val_loss, ckpt_path, reference_ckpt = _estimate_dpo_loss(
        dpo_config_path=str(dpo_cfg["dpo_config"]),
        ckpt_path=dpo_cfg.get("ckpt"),
        eval_batches=int(dpo_cfg.get("eval_batches", 20)),
    )

    instructions = list(dpo_cfg.get("instructions", []))
    if not instructions:
        raise RuntimeError("eval.dpo.instructions must contain at least one instruction item")

    samples = []
    for item in instructions:
        instruction = str(item["instruction"])
        inp = item.get("input")
        text = dpo_sample(
            config_path=str(dpo_cfg["dpo_config"]),
            instruction=instruction,
            inp=None if inp is None else str(inp),
            ckpt_path=dpo_cfg.get("ckpt"),
            max_new_tokens=int(dpo_cfg.get("max_new_tokens", 120)),
            temperature=float(dpo_cfg.get("temperature", 0.3)),
            top_k=int(dpo_cfg.get("top_k", 20)),
        )
        samples.append(
            {
                "instruction": instruction,
                "input": None if inp is None else str(inp),
                "text": text,
            }
        )

    report = {
        "kind": "dpo_eval",
        "checkpoint": ckpt_path,
        "reference_checkpoint": reference_ckpt,
        "dpo_config": str(dpo_cfg["dpo_config"]),
        "val_dpo_loss": val_loss,
        "num_instructions": len(samples),
        "samples": samples,
    }
    out_path = Path(out_dir) / "dpo_eval.json"
    saved = _write_json(out_path, report)
    log.info("Wrote DPO eval report to %s", saved)
    return saved


def run_rag_eval(config_path: str) -> str:
    cfg = _load_yaml(config_path)
    out_dir = ensure_dir(cfg["paths"]["out_dir"])
    rag_cfg = cfg["rag"]

    queries = list(rag_cfg.get("queries", []))
    if not queries:
        raise RuntimeError("eval.rag.queries must contain at least one question item")

    reports = []
    for item in queries:
        if isinstance(item, str):
            question = item
            overrides: dict[str, Any] = {}
        else:
            question = str(item["question"])
            overrides = dict(item)
            overrides.pop("question", None)

        report = rag_query(
            config_path=str(rag_cfg["rag_config"]),
            question=question,
            retrieval_top_k=overrides.get("retrieval_top_k", rag_cfg.get("retrieval_top_k")),
            max_new_tokens=overrides.get("max_new_tokens", rag_cfg.get("max_new_tokens")),
            temperature=overrides.get("temperature", rag_cfg.get("temperature")),
            top_k=overrides.get("top_k", rag_cfg.get("top_k")),
        )
        reports.append(
            {
                "question": question,
                "text": report["text"],
                "retrieved": report["retrieved"],
                "generator": report["generator"],
                "query_report_path": report["report_path"],
            }
        )

    report = {
        "kind": "rag_eval",
        "rag_config": str(rag_cfg["rag_config"]),
        "num_queries": len(reports),
        "queries": reports,
    }
    out_path = Path(out_dir) / "rag_eval.json"
    saved = _write_json(out_path, report)
    log.info("Wrote RAG eval report to %s", saved)
    return saved
