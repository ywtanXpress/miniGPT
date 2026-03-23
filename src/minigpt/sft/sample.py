# src/minigpt/sft/sample.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
import torch
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.train.ckpt import load_checkpoint, find_latest_checkpoint

log = get_logger("minigpt.sft.sample")


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_device(device_cfg: str) -> torch.device:
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _select_amp_dtype(dtype_cfg: str, device: torch.device):
    if dtype_cfg == "auto":
        if device.type != "cuda":
            return False, torch.float32
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        return True, torch.float16
    if dtype_cfg == "bf16":
        return True, torch.bfloat16
    return False, torch.float32


def _format_prompt(template: dict, instruction: str, inp: Optional[str]) -> str:
    user = instruction.strip()
    if inp is not None:
        inp = inp.strip()
        if inp:
            user = user + "\n\n" + inp

    sys = template.get("system", "") or ""
    up = template.get("user_prefix", "### Instruction:\n")
    ap = template.get("assistant_prefix", "\n\n### Response:\n")

    prompt = ""
    if sys.strip():
        prompt += sys.strip() + "\n\n"
    prompt += up + user + ap
    return prompt


@torch.no_grad()
def sft_sample(
    config_path: str,
    instruction: str,
    inp: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
) -> str:
    cfg = _load_cfg(config_path)

    paths = cfg["paths"]
    tok_dir = paths["tokenizer_dir"]
    out_dir = paths["out_dir"]

    device_cfg = str(cfg["train"].get("device", "auto"))
    dtype_cfg = str(cfg["train"].get("dtype", "auto"))

    device = _select_device(device_cfg)
    use_amp, amp_dtype = _select_amp_dtype(dtype_cfg, device)

    tok_path = Path(tok_dir) / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    vocab_size = int(tok.get_vocab_size())
    eos_id = tok.token_to_id("<|eos|>")

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

    model = GPT(gcfg).to(device)
    model.eval()

    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(out_dir)

    log.info("Loading SFT checkpoint: %s", ckpt_path)
    load_checkpoint(ckpt_path, model, opt=None)

    prompt = _format_prompt(cfg["data"]["template"], instruction, inp)
    ids = tok.encode(prompt).ids
    if len(ids) > block_size:
        ids = ids[-block_size:]

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_id,
            )
    else:
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
        )

    out_ids = out[0].tolist()
    return tok.decode(out_ids)
