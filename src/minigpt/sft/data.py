# sec/minigpt/sft/data.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Shard:
    bin_path: Path
    idx_path: Path
    num_tokens: int
    doc_starts: List[int]
    response_starts: List[int]
    mm: np.memmap
    eligible_docs: List[int]


def _load_idx(idx_path: Path) -> dict:
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_eligible_docs(doc_starts: List[int], num_tokens: int, block_size: int) -> List[int]:
    out = []
    n = len(doc_starts)
    for i in range(n):
        s = doc_starts[i]
        e = doc_starts[i + 1] if i + 1 < n else num_tokens
        if (e - s) >= (block_size + 1):
            out.append(i)
    return out


class SFTTokenDataset:
    def __init__(self, tokens_dir: str, shard_prefix: str, block_size: int):
        self.tokens_dir = Path(tokens_dir)
        self.shard_prefix = shard_prefix
        self.block_size = int(block_size)

        self.train_shards: List[Shard] = []
        self.val_shards: List[Shard] = []

        self._load_shards()

    def _load_shards(self) -> None:
        train_bins = sorted(self.tokens_dir.glob(f"train_{self.shard_prefix}_*.bin"))
        val_bins = sorted(self.tokens_dir.glob(f"val_{self.shard_prefix}_*.bin"))
        if not train_bins or not val_bins:
            raise RuntimeError(f"Missing train/val shards in {self.tokens_dir}")

        self.train_shards = [self._load_one(p) for p in train_bins]
        self.val_shards = [self._load_one(p) for p in val_bins]

    def _load_one(self, bin_path: Path) -> Shard:
        idx_path = bin_path.with_suffix(".idx.json")
        if not idx_path.exists():
            raise RuntimeError(f"Missing idx file for {bin_path}")

        info = _load_idx(idx_path)
        num_tokens = int(info["num_tokens"])
        doc_starts = list(info["doc_starts"])
        response_starts = list(info["response_starts"])

        if len(doc_starts) != len(response_starts):
            raise RuntimeError(f"doc_starts/response_starts length mismatch: {idx_path}")

        mm = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(num_tokens,))
        eligible = _build_eligible_docs(doc_starts, num_tokens, self.block_size)
        if not eligible:
            raise RuntimeError(f"No eligible docs in shard {bin_path.name}")

        return Shard(
            bin_path=bin_path,
            idx_path=idx_path,
            num_tokens=num_tokens,
            doc_starts=doc_starts,
            response_starts=response_starts,
            mm=mm,
            eligible_docs=eligible,
        )

    def _sample_from_shards(self, shards: List[Shard], batch_size: int, device: torch.device):
        xs = []
        ys = []
        loss_masks = []

        for _ in range(batch_size):
            shard = shards[np.random.randint(0, len(shards))]
            di = shard.eligible_docs[np.random.randint(0, len(shard.eligible_docs))]

            s = shard.doc_starts[di]
            e = shard.doc_starts[di + 1] if (di + 1) < len(shard.doc_starts) else shard.num_tokens
            doc_len = e - s

            resp_start = shard.response_starts[di]

            off = np.random.randint(0, doc_len - (self.block_size + 1) + 1)
            start = s + int(off)

            seq = np.array(shard.mm[start : start + self.block_size + 1], dtype=np.int64)
            x = torch.from_numpy(seq[:-1])
            y = torch.from_numpy(seq[1:])

            # loss mask applies to y positions
            # y[t] corresponds to original token index (start + t + 1)
            base = start + 1
            idxs = torch.arange(0, self.block_size, dtype=torch.long) + base
            lm = (idxs >= int(resp_start)).to(torch.bool)

            xs.append(x)
            ys.append(y)
            loss_masks.append(lm)

        x = torch.stack(xs, dim=0).to(device)
        y = torch.stack(ys, dim=0).to(device)
        loss_mask = torch.stack(loss_masks, dim=0).to(device)

        return x, y, loss_mask

    def get_batch(self, split: str, batch_size: int, device: torch.device):
        if split == "train":
            return self._sample_from_shards(self.train_shards, batch_size, device)
        if split == "val":
            return self._sample_from_shards(self.val_shards, batch_size, device)
        raise ValueError(f"Unknown split: {split}")
