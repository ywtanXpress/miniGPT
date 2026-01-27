# src/minigpt/sft/build.py

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir

log = get_logger("minigpt.sft.build")


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_idx(
    path: Path,
    num_tokens: int,
    doc_starts: List[int],
    response_starts: List[int],
    doc_lens: List[int],
) -> None:
    obj = {
        "num_tokens": int(num_tokens),
        "doc_starts": doc_starts,
        "response_starts": response_starts,
        "doc_lens": doc_lens,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _format_example(
    template: dict,
    instruction: str,
    inp: Optional[str],
    output: str,
) -> Tuple[str, str]:
    user = instruction.strip()
    if inp is not None:
        inp = inp.strip()
        if inp:
            user = user + "\n\n" + inp

    sys = template.get("system", "") or ""
    up = template.get("user_prefix", "### Instruction:\n")
    ap = template.get("assistant_prefix", "\n\n### Response:\n")
    end = template.get("end", "\n")

    prompt = ""
    if sys.strip():
        prompt += sys.strip() + "\n\n"
    prompt += up + user + ap

    response = output.strip() + end
    return prompt, response


def build_sft_tokens(config_path: str) -> None:
    cfg = _load_cfg(config_path)

    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)

    paths = cfg["paths"]
    tok_dir = Path(paths["tokenizer_dir"])
    out_dir = ensure_dir(paths["sft_tokens_dir"])

    data_cfg = cfg["data"]
    shard_prefix = str(data_cfg.get("shard_prefix", "sft"))
    shard_size = int(data_cfg.get("shard_size", 5000))
    max_total = int(data_cfg.get("max_examples_total", 50000))
    val_ratio = float(data_cfg.get("val_ratio", 0.02))
    min_resp_tokens = int(data_cfg.get("min_resp_tokens", 16))
    template = data_cfg.get("template", {})

    block_size = int(cfg["tokenization"]["block_size"])
    seq_len = block_size + 1  # we store (block_size + 1) to form x/y

    src = data_cfg["source"]
    if src["kind"] != "hf":
        raise RuntimeError("Only kind=hf supported for Step 4 builder")

    ds_name = src["dataset"]
    subset = src.get("subset", None)
    split = src.get("split", "train")

    inst_f = src["instruction_field"]
    inp_f = src.get("input_field", None)
    out_f = src["output_field"]

    tok_path = tok_dir / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))

    pad_id = tok.token_to_id("<|pad|>")
    if pad_id is None:
        raise RuntimeError("Tokenizer missing <|pad|> token; cannot pad SFT sequences")

    log.info("Loading dataset: %s split=%s", ds_name, split)
    ds = load_dataset(ds_name, subset, split=split)

    indices = list(range(len(ds)))
    random.shuffle(indices)

    n_val = max(1, int(len(indices) * val_ratio))
    val_ids = indices[:n_val]
    train_ids = indices[n_val:]

    def _iter_rows(idxs):
        n = 0
        for i in idxs:
            r = ds[int(i)]
            instruction = str(r.get(inst_f, "") or "")
            inp = None
            if inp_f is not None:
                v = r.get(inp_f, None)
                inp = None if v is None else str(v)
            output = str(r.get(out_f, "") or "")
            if not instruction.strip() or not output.strip():
                continue
            yield instruction, inp, output
            n += 1
            if n >= max_total:
                break

    def _build_split(split_name: str, rows_iter):
        tokens_all: List[int] = []
        doc_starts: List[int] = []
        response_starts: List[int] = []
        doc_lens: List[int] = []

        shard_idx = 0
        docs_in_shard = 0
        num_written = 0
        num_skipped = 0

        def _flush():
            nonlocal shard_idx, docs_in_shard, tokens_all, doc_starts, response_starts, doc_lens
            if docs_in_shard == 0:
                return

            base = Path(out_dir) / f"{split_name}_{shard_prefix}_{shard_idx:05d}"
            bin_path = base.with_suffix(".bin")
            idx_path = base.with_suffix(".idx.json")

            arr = np.asarray(tokens_all, dtype=np.uint32)
            arr.tofile(str(bin_path))

            _write_idx(
                idx_path,
                num_tokens=len(tokens_all),
                doc_starts=doc_starts,
                response_starts=response_starts,
                doc_lens=doc_lens,
            )

            log.info(
                "%s shard=%d wrote docs=%d tokens=%d -> %s",
                split_name,
                shard_idx,
                docs_in_shard,
                len(tokens_all),
                str(bin_path),
            )
            shard_idx += 1
            docs_in_shard = 0
            tokens_all = []
            doc_starts = []
            response_starts = []
            doc_lens = []

        for instruction, inp, output in rows_iter:
            prompt, response = _format_example(template, instruction, inp, output)

            prompt_ids = tok.encode(prompt).ids
            resp_ids = tok.encode(response).ids

            if not prompt_ids or not resp_ids:
                num_skipped += 1
                continue

            # We require the prompt to leave room for at least min_resp_tokens in-window.
            max_resp_in_window = seq_len - len(prompt_ids)
            if max_resp_in_window < min_resp_tokens:
                num_skipped += 1
                continue

            resp_keep = min(len(resp_ids), max_resp_in_window)
            if resp_keep < min_resp_tokens:
                num_skipped += 1
                continue

            seq = prompt_ids + resp_ids[:resp_keep]
            real_len = len(seq)

            # Pad to fixed length (seq_len) so training always uses start-of-doc windows.
            if real_len < seq_len:
                seq = seq + [int(pad_id)] * (seq_len - real_len)
            else:
                # Should never exceed seq_len due to resp_keep, but keep it safe.
                seq = seq[:seq_len]
                real_len = min(real_len, seq_len)

            doc_start = len(tokens_all)
            response_start = doc_start + len(prompt_ids)

            doc_starts.append(doc_start)
            response_starts.append(response_start)
            doc_lens.append(int(real_len))

            tokens_all.extend(seq)

            docs_in_shard += 1
            num_written += 1

            if docs_in_shard >= shard_size:
                _flush()

        _flush()
        log.info(
            "%s done: wrote=%d skipped=%d (prompt too long / too few resp tokens / empty)",
            split_name,
            num_written,
            num_skipped,
        )
        return num_written

    n_train = _build_split("train", _iter_rows(train_ids))
    n_val_written = _build_split("val", _iter_rows(val_ids))

    log.info(
        "Done. train_docs=%d val_docs=%d out_dir=%s",
        n_train,
        n_val_written,
        str(out_dir),
    )
