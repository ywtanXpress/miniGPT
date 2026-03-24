from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
import torch.nn.functional as F
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir
from minigpt.model.gpt import GPT, GPTConfig
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
