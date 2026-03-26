from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir

log = get_logger("minigpt.dpo.build")


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


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


def _build_sequence(
    prompt_ids: List[int],
    resp_ids: List[int],
    seq_len: int,
    min_resp_tokens: int,
    eos_id: int,
    pad_id: int,
) -> Optional[Tuple[List[int], int, bool]]:
    if not prompt_ids or not resp_ids:
        return None

    available = seq_len - len(prompt_ids)
    if available < min_resp_tokens:
        return None

    full_resp_len = len(resp_ids) + 1
    if full_resp_len <= available:
        seq = prompt_ids + resp_ids + [int(eos_id)]
        added_eos = True
    else:
        resp_keep = min(len(resp_ids), available)
        if resp_keep < min_resp_tokens:
            return None
        seq = prompt_ids + resp_ids[:resp_keep]
        added_eos = False

    real_len = len(seq)
    if real_len < seq_len:
        seq = seq + [int(pad_id)] * (seq_len - real_len)

    return seq, real_len, added_eos


def _iter_preference_rows(cfg: dict) -> Iterator[Tuple[str, Optional[str], str, str]]:
    src = cfg["data"]["source"]
    kind = str(src["kind"])

    if kind == "jsonl":
        rows = _iter_jsonl(src["path"])
    elif kind == "hf":
        rows = load_dataset(src["dataset"], src.get("subset"), split=src.get("split", "train"))
    else:
        raise RuntimeError(f"Unsupported DPO source kind={kind!r}")

    inst_f = src["instruction_field"]
    inp_f = src.get("input_field")
    chosen_f = src["chosen_field"]
    rejected_f = src["rejected_field"]

    for row in rows:
        instruction = str(row.get(inst_f, "") or "")
        inp = None
        if inp_f is not None:
            val = row.get(inp_f)
            inp = None if val is None else str(val)
        chosen = str(row.get(chosen_f, "") or "")
        rejected = str(row.get(rejected_f, "") or "")
        if not instruction.strip() or not chosen.strip() or not rejected.strip():
            continue
        yield instruction, inp, chosen, rejected


def _write_records(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_dpo_pairs(config_path: str) -> None:
    cfg = _load_cfg(config_path)

    seed = int(cfg.get("seed", 1337))
    random.seed(seed)

    paths = cfg["paths"]
    out_dir = ensure_dir(paths["preference_dir"])
    tok_dir = Path(paths["tokenizer_dir"])

    data_cfg = cfg["data"]
    shard_prefix = str(data_cfg.get("shard_prefix", "dpo"))
    shard_size = int(data_cfg.get("shard_size", 5000))
    max_total = int(data_cfg.get("max_examples_total", 20000))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    min_resp_tokens = int(data_cfg.get("min_resp_tokens", 16))
    template = data_cfg.get("template", {})

    block_size = int(cfg["tokenization"]["block_size"])
    seq_len = block_size + 1

    tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    pad_id = tok.token_to_id("<|pad|>")
    eos_id = tok.token_to_id("<|eos|>")
    if pad_id is None or eos_id is None:
        raise RuntimeError("Tokenizer must include <|pad|> and <|eos|> for DPO")

    rows = list(_iter_preference_rows(cfg))
    random.shuffle(rows)

    total_budget = min(max_total, len(rows))
    val_target = min(len(rows), max(1, int(total_budget * val_ratio)))
    train_target = max(0, total_budget - val_target)

    val_rows = rows[:val_target]
    train_rows = rows[val_target : val_target + train_target]

    def _build_split(split_name: str, split_rows: List[Tuple[str, Optional[str], str, str]]) -> int:
        shard_idx = 0
        records: List[dict] = []
        written = 0
        skipped = 0
        chosen_with_eos = 0
        rejected_with_eos = 0

        def _flush() -> None:
            nonlocal shard_idx, records
            if not records:
                return
            out_path = Path(out_dir) / f"{split_name}_{shard_prefix}_{shard_idx:05d}.jsonl"
            _write_records(out_path, records)
            log.info("%s shard=%d wrote records=%d -> %s", split_name, shard_idx, len(records), str(out_path))
            shard_idx += 1
            records = []

        for instruction, inp, chosen, rejected in split_rows:
            prompt = _format_prompt(template, instruction, inp)
            prompt_ids = tok.encode(prompt).ids
            chosen_ids = tok.encode(chosen.strip() + (template.get("end", "\n"))).ids
            rejected_ids = tok.encode(rejected.strip() + (template.get("end", "\n"))).ids

            chosen_seq = _build_sequence(prompt_ids, chosen_ids, seq_len, min_resp_tokens, int(eos_id), int(pad_id))
            rejected_seq = _build_sequence(prompt_ids, rejected_ids, seq_len, min_resp_tokens, int(eos_id), int(pad_id))
            if chosen_seq is None or rejected_seq is None:
                skipped += 1
                continue

            chosen_tokens, chosen_len, chosen_eos = chosen_seq
            rejected_tokens, rejected_len, rejected_eos = rejected_seq
            response_start = len(prompt_ids)

            records.append(
                {
                    "chosen_ids": chosen_tokens,
                    "chosen_len": int(chosen_len),
                    "rejected_ids": rejected_tokens,
                    "rejected_len": int(rejected_len),
                    "response_start": int(response_start),
                }
            )
            written += 1
            if chosen_eos:
                chosen_with_eos += 1
            if rejected_eos:
                rejected_with_eos += 1

            if len(records) >= shard_size:
                _flush()

        _flush()
        log.info(
            "%s done: wrote=%d skipped=%d chosen_with_eos=%d rejected_with_eos=%d",
            split_name,
            written,
            skipped,
            chosen_with_eos,
            rejected_with_eos,
        )
        return written

    n_train = _build_split("train", train_rows)
    n_val = _build_split("val", val_rows)
    summary = {
        "train_records": n_train,
        "val_records": n_val,
        "total_budget": total_budget,
        "tokenization_block_size": block_size,
    }
    with open(Path(out_dir) / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    log.info("Done. train_records=%d val_records=%d out_dir=%s", n_train, n_val, str(out_dir))
