# src/minigpt/train/data.py

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


class TokenShardDataset:
    def __init__(self, tokens_dir: str, shard_prefix: str, block_size: int, val_num_shards: int = 1):
        self.tokens_dir = Path(tokens_dir)
        self.shard_prefix = shard_prefix
        self.block_size = int(block_size)
        self.val_num_shards = int(val_num_shards)

        self.train_shards: List[Shard] = []
        self.val_shards: List[Shard] = []

        self._load_shards()

    def _load_shards(self) -> None:
        bins = sorted(self.tokens_dir.glob(f"{self.shard_prefix}_*.bin"))
        if not bins:
            raise RuntimeError(f"No token shards found in {self.tokens_dir}")

        pairs: List[Tuple[Path, Path]] = []
        for b in bins:
            idx = b.with_suffix(".idx.json")
            if not idx.exists():
                raise RuntimeError(f"Missing idx file for {b}")
            pairs.append((b, idx))

        if self.val_num_shards >= len(pairs):
            raise RuntimeError("val_num_shards must be < number of shards")

        train_pairs = pairs[:-self.val_num_shards]
        val_pairs = pairs[-self.val_num_shards:]

        self.train_shards = [self._load_one(bp, ip) for (bp, ip) in train_pairs]
        self.val_shards = [self._load_one(bp, ip) for (bp, ip) in val_pairs]

    def _load_one(self, bin_path: Path, idx_path: Path) -> Shard:
        info = _load_idx(idx_path)
        num_tokens = int(info["num_tokens"])
        doc_starts = list(info["doc_starts"])

        mm = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(num_tokens,))
        eligible = _build_eligible_docs(doc_starts, num_tokens, self.block_size)
        if not eligible:
            raise RuntimeError(f"No eligible docs in shard {bin_path.name}")

        return Shard(
            bin_path=bin_path,
            idx_path=idx_path,
            num_tokens=num_tokens,
            doc_starts=doc_starts,
            mm=mm,
            eligible_docs=eligible,
        )

    def _sample_from_shards(self, shards: List[Shard], batch_size: int, device: torch.device):
        xs = []
        ys = []

        for _ in range(batch_size):
            shard = shards[np.random.randint(0, len(shards))]
            di = shard.eligible_docs[np.random.randint(0, len(shard.eligible_docs))]

            s = shard.doc_starts[di]
            e = shard.doc_starts[di + 1] if (di + 1) < len(shard.doc_starts) else shard.num_tokens
            doc_len = e - s

            off = np.random.randint(0, doc_len - (self.block_size + 1) + 1)
            start = s + int(off)

            seq = np.array(shard.mm[start : start + self.block_size + 1], dtype=np.int64)
            x = torch.from_numpy(seq[:-1])
            y = torch.from_numpy(seq[1:])

            xs.append(x)
            ys.append(y)

        x = torch.stack(xs, dim=0).to(device)
        y = torch.stack(ys, dim=0).to(device)
        return x, y

    def get_batch(self, split: str, batch_size: int, device: torch.device):
        if split == "train":
            return self._sample_from_shards(self.train_shards, batch_size, device)
        if split == "val":
            return self._sample_from_shards(self.val_shards, batch_size, device)
        raise ValueError(f"Unknown split: {split}")
