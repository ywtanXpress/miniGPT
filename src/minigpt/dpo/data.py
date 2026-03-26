from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


class DPOPairDataset:
    def __init__(self, preference_dir: str, shard_prefix: str, block_size: int, min_resp_tokens: int = 16):
        self.preference_dir = Path(preference_dir)
        self.shard_prefix = shard_prefix
        self.block_size = int(block_size)
        self.min_resp_tokens = int(min_resp_tokens)
        self.train_records = self._load_split("train")
        self.val_records = self._load_split("val")

    def _load_split(self, split: str) -> List[dict]:
        files = sorted(self.preference_dir.glob(f"{split}_{self.shard_prefix}_*.jsonl"))
        if not files:
            raise RuntimeError(f"Missing {split} DPO shards in {self.preference_dir}")
        records: List[dict] = []
        for path in files:
            records.extend(list(_iter_jsonl(path)))
        if not records:
            raise RuntimeError(f"No records found in {split} DPO shards")
        return records

    def _mask_for(self, response_start: int, real_len: int) -> torch.Tensor:
        idxs = torch.arange(0, self.block_size, dtype=torch.long) + 1
        valid = idxs < int(real_len)
        return (idxs >= int(response_start)) & valid

    def get_batch(self, split: str, batch_size: int, device: torch.device):
        records = self.train_records if split == "train" else self.val_records if split == "val" else None
        if records is None:
            raise ValueError(f"Unknown split: {split}")

        chosen_xs = []
        chosen_ys = []
        chosen_masks = []
        rejected_xs = []
        rejected_ys = []
        rejected_masks = []

        for _ in range(batch_size):
            rec = records[np.random.randint(0, len(records))]
            chosen = np.asarray(rec["chosen_ids"], dtype=np.int64)
            rejected = np.asarray(rec["rejected_ids"], dtype=np.int64)
            response_start = int(rec["response_start"])
            chosen_len = int(rec["chosen_len"])
            rejected_len = int(rec["rejected_len"])

            chosen_xs.append(torch.from_numpy(chosen[:-1]))
            chosen_ys.append(torch.from_numpy(chosen[1:]))
            chosen_masks.append(self._mask_for(response_start, chosen_len))

            rejected_xs.append(torch.from_numpy(rejected[:-1]))
            rejected_ys.append(torch.from_numpy(rejected[1:]))
            rejected_masks.append(self._mask_for(response_start, rejected_len))

        return (
            torch.stack(chosen_xs, dim=0).to(device),
            torch.stack(chosen_ys, dim=0).to(device),
            torch.stack(chosen_masks, dim=0).to(device),
            torch.stack(rejected_xs, dim=0).to(device),
            torch.stack(rejected_ys, dim=0).to(device),
            torch.stack(rejected_masks, dim=0).to(device),
        )
