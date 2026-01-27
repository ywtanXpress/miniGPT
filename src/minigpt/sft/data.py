# src/minigpt/sft/data.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch


@dataclass
class Shard:
    bin_path: Path
    idx_path: Path
    num_tokens: int
    doc_starts: List[int]
    response_starts: List[int]
    doc_lens: List[int]
    mm: np.memmap
    eligible_docs: List[int]


def _load_idx(idx_path: Path) -> dict:
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_eligible_docs(
    doc_starts: List[int],
    response_starts: List[int],
    doc_lens: List[int],
    block_size: int,
    min_resp_tokens: int,
) -> List[int]:
    """
    Each stored doc is (block_size+1) tokens on disk (padded/truncated), but we also
    store doc_lens = real length before padding (<= block_size+1).

    We consider a doc eligible if, within the window, there are at least min_resp_tokens
    response targets that are also within the real (unpadded) region.
    """
    out: List[int] = []
    for i in range(len(doc_starts)):
        s = int(doc_starts[i])
        resp = int(response_starts[i])
        real_len = int(doc_lens[i])  # number of real tokens in this doc, incl prompt+resp_prefix
        # y positions correspond to token indices [s+1, s+block_size] inclusive range length block_size
        # valid y indices must be < s+real_len
        y_first = s + 1
        y_last_excl = s + min(block_size + 1, real_len)  # exclusive upper bound for y indices

        # response-supervised y indices are [max(resp, y_first), y_last_excl)
        supervised = max(resp, y_first)
        n_supervised = max(0, y_last_excl - supervised)
        if n_supervised >= int(min_resp_tokens):
            out.append(i)
    return out


class SFTTokenDataset:
    def __init__(self, tokens_dir: str, shard_prefix: str, block_size: int, min_resp_tokens: int = 16):
        self.tokens_dir = Path(tokens_dir)
        self.shard_prefix = shard_prefix
        self.block_size = int(block_size)
        self.min_resp_tokens = int(min_resp_tokens)

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
        doc_lens = list(info.get("doc_lens", []))

        if len(doc_starts) != len(response_starts):
            raise RuntimeError(f"doc_starts/response_starts length mismatch: {idx_path}")
        if len(doc_lens) != len(doc_starts):
            raise RuntimeError(f"doc_lens missing or length mismatch: {idx_path}")

        mm = np.memmap(bin_path, dtype=np.uint32, mode="r", shape=(num_tokens,))

        eligible = _build_eligible_docs(
            doc_starts=doc_starts,
            response_starts=response_starts,
            doc_lens=doc_lens,
            block_size=self.block_size,
            min_resp_tokens=self.min_resp_tokens,
        )
        if not eligible:
            raise RuntimeError(f"No eligible docs in shard {bin_path.name}")

        return Shard(
            bin_path=bin_path,
            idx_path=idx_path,
            num_tokens=num_tokens,
            doc_starts=doc_starts,
            response_starts=response_starts,
            doc_lens=doc_lens,
            mm=mm,
            eligible_docs=eligible,
        )

    def _sample_from_shards(self, shards: List[Shard], batch_size: int, device: torch.device):
        xs = []
        ys = []
        loss_masks = []

        # IMPORTANT: Always sample from the start of the doc (doc_start).
        for _ in range(batch_size):
            shard = shards[np.random.randint(0, len(shards))]
            di = shard.eligible_docs[np.random.randint(0, len(shard.eligible_docs))]

            s = int(shard.doc_starts[di])
            resp_start = int(shard.response_starts[di])
            real_len = int(shard.doc_lens[di])

            # We stored docs as fixed length (block_size+1). Always take the prefix window.
            seq = np.array(shard.mm[s : s + self.block_size + 1], dtype=np.int64)
            x = torch.from_numpy(seq[:-1])
            y = torch.from_numpy(seq[1:])

            # loss mask applies to y positions
            # y[t] corresponds to original token index (s + t + 1)
            base = s + 1
            idxs = torch.arange(0, self.block_size, dtype=torch.long) + base

            # mask: (a) only response tokens (>= resp_start) AND (b) only real (unpadded) region
            valid = idxs < (s + real_len)
            lm = (idxs >= resp_start) & valid

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
