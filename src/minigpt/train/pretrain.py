# src/minigpt/train/pretrain.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import torch
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.train.data import TokenShardDataset
from minigpt.train.optim import get_lr, set_optimizer_lr
from minigpt.train.ckpt import save_checkpoint, load_checkpoint, find_latest_checkpoint

log = get_logger("minigpt.train.pretrain")


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def _estimate_loss(model: GPT, ds: TokenShardDataset, device: torch.device, batch_size: int, eval_batches: int):
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = ds.get_batch("val", batch_size=batch_size, device=device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / float(len(losses))


def pretrain(config_path: str, resume: bool = False, ckpt_path: Optional[str] = None) -> None:
    cfg = _load_cfg(config_path)

    seed = int(cfg.get("seed", 1337))
    torch.manual_seed(seed)
    np.random.seed(seed)

    paths = cfg["paths"]
    tok_dir = paths["tokenizer_dir"]
    tokens_dir = paths["tokens_dir"]
    out_dir = ensure_dir(paths["out_dir"])

    data_cfg = cfg["data"]
    shard_prefix = data_cfg.get("shard_prefix", "tok")
    block_size = int(data_cfg["block_size"])
    batch_size = int(data_cfg["batch_size"])
    val_num_shards = int(data_cfg.get("val_num_shards", 1))

    train_cfg = cfg["train"]
    max_steps = int(train_cfg["max_steps"])
    grad_accum = int(train_cfg.get("grad_accum", 1))
    base_lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", 0.0))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    eval_interval = int(train_cfg.get("eval_interval", 200))
    eval_batches = int(train_cfg.get("eval_batches", 50))
    save_interval = int(train_cfg.get("save_interval", 500))
    log_interval = int(train_cfg.get("log_interval", 20))

    device_cfg = str(train_cfg.get("device", "auto"))
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")

    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    dtype_cfg = str(train_cfg.get("dtype", "auto"))
    if dtype_cfg == "auto":
        use_amp = device.type == "cuda"
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    elif dtype_cfg == "bf16":
        use_amp = True
        amp_dtype = torch.bfloat16
    else:
        use_amp = False
        amp_dtype = torch.float32

    tok_path = Path(tok_dir) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    vocab_size = int(tok.get_vocab_size())

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
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    ds = TokenShardDataset(
        tokens_dir=tokens_dir,
        shard_prefix=shard_prefix,
        block_size=block_size,
        val_num_shards=val_num_shards,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    start_step = 0

    if resume or ckpt_path is not None:
        if ckpt_path is None:
            ckpt_path = find_latest_checkpoint(out_dir)

        log.info("Resuming from checkpoint: %s", ckpt_path)
        ckpt = load_checkpoint(ckpt_path, model, opt=opt)
        start_step = int(ckpt.get("step", 0))

        log.info("Checkpoint step=%d (will continue from step %d)", start_step, start_step + 1)

    log.info("Device=%s use_amp=%s amp_dtype=%s", str(device), str(use_amp), str(amp_dtype))
    log.info("Model: vocab=%d block=%d layers=%d heads=%d embd=%d", vocab_size, block_size, gcfg.n_layer, gcfg.n_head, gcfg.n_embd)

    model.train()
    running = 0.0

    for step in range(start_step, max_steps):
        lr = get_lr(step, base_lr, warmup_steps, max_steps, min_lr)
        set_optimizer_lr(opt, lr)

        opt.zero_grad(set_to_none=True)

        for _ in range(grad_accum):
            x, y = ds.get_batch("train", batch_size=batch_size, device=device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    _, loss = model(x, y)
                loss = loss / float(grad_accum)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                _, loss = model(x, y)
                loss = loss / float(grad_accum)
                loss.backward()

            running += float(loss.item())

        if scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

        if (step + 1) % log_interval == 0:
            avg = running / float(log_interval)
            running = 0.0
            log.info("step=%d lr=%.6g loss=%.4f", step + 1, lr, avg)

        if (step + 1) % eval_interval == 0:
            val_loss = _estimate_loss(model, ds, device, batch_size=batch_size, eval_batches=eval_batches)
            log.info("eval step=%d val_loss=%.4f", step + 1, val_loss)

        if (step + 1) % save_interval == 0:
            meta = {"config_path": config_path, "vocab_size": vocab_size}
            p = save_checkpoint(out_dir, step + 1, model, opt, meta)
            log.info("saved checkpoint: %s", p)

    meta = {"config_path": config_path, "vocab_size": vocab_size}
    p = save_checkpoint(out_dir, max_steps, model, opt, meta)
    log.info("done. saved final checkpoint: %s", p)
