# src/minigpt/train/optim.py

from __future__ import annotations

import math

import torch


def get_lr(step: int, lr: float, warmup_steps: int, max_steps: int, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return lr * (float(step + 1) / float(max(1, warmup_steps)))
    if step >= max_steps:
        return min_lr
    t = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))  # 1 -> 0
    return min_lr + (lr - min_lr) * cosine


def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr
