# src/minigpt/train/optim.py

from __future__ import annotations

import math

import torch


def get_lr(step: int, lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return lr * (float(step + 1) / float(max(1, warmup_steps)))
    if step >= max_steps:
        return 0.0
    t = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.5 * lr * (1.0 + math.cos(math.pi * t))


def set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = lr
