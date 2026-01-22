# src/minigpt/train/ckpt.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from minigpt.common.paths import ensure_dir


def save_checkpoint(out_dir: str, step: int, model: torch.nn.Module, opt: torch.optim.Optimizer, meta: Dict[str, Any]) -> str:
    ensure_dir(out_dir)
    path = Path(out_dir) / f"ckpt_{step:06d}.pt"
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "meta": meta,
    }
    torch.save(payload, str(path))
    return str(path)


def load_checkpoint(path: str, model: torch.nn.Module, opt: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    return ckpt


def find_latest_checkpoint(out_dir: str) -> str:
    p = Path(out_dir)
    ckpts = sorted(p.glob("ckpt_*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {out_dir}")
    return str(ckpts[-1])
