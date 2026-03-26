from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir
from minigpt.dpo.data import DPOPairDataset
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.train.ckpt import find_latest_checkpoint, load_checkpoint, save_checkpoint
from minigpt.train.optim import get_lr, set_optimizer_lr

log = get_logger("minigpt.dpo.train")


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_ckpt(path_or_dir: str) -> str:
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


def _masked_logprob_sum(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (token_lp * loss_mask.to(dtype=token_lp.dtype)).sum(dim=1)


def _dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(float(beta) * margin).mean()


@torch.no_grad()
def _estimate_loss(
    policy: GPT,
    reference: GPT,
    ds: DPOPairDataset,
    device: torch.device,
    batch_size: int,
    eval_batches: int,
    beta: float,
) -> float:
    policy.eval()
    reference.eval()
    losses = []
    for _ in range(eval_batches):
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
    policy.train()
    return sum(losses) / float(len(losses))


def dpo_train(config_path: str, resume: bool = False, ckpt_path: Optional[str] = None) -> None:
    cfg = _load_cfg(config_path)
    seed = int(cfg.get("seed", 1337))
    torch.manual_seed(seed)
    np.random.seed(seed)

    paths = cfg["paths"]
    out_dir = ensure_dir(paths["out_dir"])
    preference_dir = paths["preference_dir"]
    init_ckpt = _resolve_ckpt(str(paths["init_ckpt"]))
    reference_ckpt_cfg = paths.get("reference_ckpt", None)
    ref_ckpt = init_ckpt if reference_ckpt_cfg is None else _resolve_ckpt(str(reference_ckpt_cfg))

    train_cfg = cfg["train"]
    device = _select_device(str(train_cfg.get("device", "auto")))
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

    tok = Tokenizer.from_file(str(Path(paths["tokenizer_dir"]) / "tokenizer.json"))
    vocab_size = int(tok.get_vocab_size())
    block_size = int(cfg["tokenization"]["block_size"])
    model_cfg = cfg["model"]
    gcfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=int(model_cfg["n_layer"]),
        n_head=int(model_cfg["n_head"]),
        n_embd=int(model_cfg["n_embd"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )

    policy = GPT(gcfg).to(device)
    reference = GPT(gcfg).to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    load_checkpoint(ref_ckpt, reference, opt=None)

    opt = torch.optim.AdamW(
        policy.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        betas=(0.9, 0.95),
    )

    start_step = 0
    if ckpt_path is not None:
        resolved = _resolve_ckpt(ckpt_path)
        log.info("Resuming DPO from checkpoint: %s", resolved)
        ckpt = load_checkpoint(resolved, policy, opt=opt)
        start_step = int(ckpt.get("step", 0))
    elif resume:
        latest = find_latest_checkpoint(out_dir)
        log.info("Resuming DPO from latest checkpoint: %s", latest)
        ckpt = load_checkpoint(latest, policy, opt=opt)
        start_step = int(ckpt.get("step", 0))
    else:
        log.info("Initializing DPO policy from SFT checkpoint: %s", init_ckpt)
        load_checkpoint(init_ckpt, policy, opt=None)

    ds = DPOPairDataset(
        preference_dir=preference_dir,
        shard_prefix=cfg["data"]["shard_prefix"],
        block_size=block_size,
        min_resp_tokens=int(cfg["data"].get("min_resp_tokens", 16)),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    max_steps = int(train_cfg["max_steps"])
    grad_accum = int(train_cfg.get("grad_accum", 1))
    batch_size = int(train_cfg["batch_size"])
    base_lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", 0.0))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    eval_interval = int(train_cfg.get("eval_interval", 200))
    eval_batches = int(train_cfg.get("eval_batches", 20))
    save_interval = int(train_cfg.get("save_interval", 500))
    log_interval = int(train_cfg.get("log_interval", 20))
    beta = float(train_cfg.get("beta", 0.1))

    running = 0.0
    for step in range(start_step, max_steps):
        lr = get_lr(step, base_lr, warmup_steps, max_steps, min_lr)
        set_optimizer_lr(opt, lr)
        opt.zero_grad(set_to_none=True)

        for _ in range(grad_accum):
            cx, cy, cm, rx, ry, rm = ds.get_batch("train", batch_size=batch_size, device=device)
            with torch.no_grad():
                ref_c, _ = reference(cx, targets=None)
                ref_r, _ = reference(rx, targets=None)
                ref_chosen = _masked_logprob_sum(ref_c, cy, cm)
                ref_rejected = _masked_logprob_sum(ref_r, ry, rm)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pol_c, _ = policy(cx, targets=None)
                    pol_r, _ = policy(rx, targets=None)
                    loss = _dpo_loss(
                        _masked_logprob_sum(pol_c, cy, cm),
                        _masked_logprob_sum(pol_r, ry, rm),
                        ref_chosen,
                        ref_rejected,
                        beta=beta,
                    )
                loss = loss / float(grad_accum)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                pol_c, _ = policy(cx, targets=None)
                pol_r, _ = policy(rx, targets=None)
                loss = _dpo_loss(
                    _masked_logprob_sum(pol_c, cy, cm),
                    _masked_logprob_sum(pol_r, ry, rm),
                    ref_chosen,
                    ref_rejected,
                    beta=beta,
                )
                loss = loss / float(grad_accum)
                loss.backward()

            running += float(loss.item())

        if scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()

        if (step + 1) % log_interval == 0:
            avg = running / float(log_interval)
            running = 0.0
            log.info("step=%d lr=%.6g dpo_loss=%.4f", step + 1, lr, avg)

        if (step + 1) % eval_interval == 0:
            val_loss = _estimate_loss(policy, reference, ds, device, batch_size, eval_batches, beta)
            log.info("eval step=%d val_dpo_loss=%.4f", step + 1, val_loss)

        if (step + 1) % save_interval == 0:
            p = save_checkpoint(out_dir, step + 1, policy, opt, {"config_path": config_path, "vocab_size": vocab_size})
            log.info("saved checkpoint: %s", p)

    p = save_checkpoint(out_dir, max_steps, policy, opt, {"config_path": config_path, "vocab_size": vocab_size})
    log.info("done. saved final checkpoint: %s", p)
