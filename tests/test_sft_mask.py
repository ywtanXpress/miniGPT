# tests/test_sft_mask.py

import json
from pathlib import Path

import numpy as np
import torch

from minigpt.sft.data import SFTTokenDataset


def _write_sft_shard(
    tokens_dir: Path,
    split: str,
    shard_prefix: str,
    shard_id: int,
    tokens: np.ndarray,
    response_start: int,
    doc_len: int,
) -> None:
    tokens_dir.mkdir(parents=True, exist_ok=True)

    # Must match SFTTokenDataset._load_shards():
    bin_path = tokens_dir / ("{0}_{1}_{2:05d}.bin".format(split, shard_prefix, shard_id))
    idx_path = tokens_dir / ("{0}_{1}_{2:05d}.idx.json".format(split, shard_prefix, shard_id))

    # IMPORTANT: write as int32 so file is not smaller than loader's expected dtype
    # (prevents mmap length > file size if loader uses int32/uint32).
    tokens = np.asarray(tokens, dtype=np.int32)
    tokens.tofile(str(bin_path))

    idx = {
        "num_tokens": int(tokens.shape[0]),
        "doc_starts": [0],
        "response_starts": [int(response_start)],
        "doc_lens": [int(doc_len)],
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx, f)


def test_loss_mask_boundary(tmp_path: Path):
    block_size = 8

    # 9 stored tokens: prompt prefix, response region, EOS, then PAD.
    tokens = np.arange(10, 10 + (block_size + 1), dtype=np.int32)
    response_start = 5  # global token index in doc
    doc_len = 8  # final token is padding and must be masked out

    tokens_dir = tmp_path / "sft_tokens"
    shard_prefix = "sft"
    shard_id = 0

    _write_sft_shard(tokens_dir, "train", shard_prefix, shard_id, tokens, response_start, doc_len)
    _write_sft_shard(tokens_dir, "val", shard_prefix, shard_id, tokens, response_start, doc_len)

    # Sanity check: files exist (helps debugging naming/glob mismatches)
    train_bins = sorted(tokens_dir.glob("train_{0}_*.bin".format(shard_prefix)))
    val_bins = sorted(tokens_dir.glob("val_{0}_*.bin".format(shard_prefix)))
    assert len(train_bins) == 1, "Expected 1 train shard, found: {0}".format([p.name for p in tokens_dir.glob("*")])
    assert len(val_bins) == 1, "Expected 1 val shard, found: {0}".format([p.name for p in tokens_dir.glob("*")])

    ds = SFTTokenDataset(
        tokens_dir=str(tokens_dir),
        shard_prefix=shard_prefix,
        block_size=block_size,
        min_resp_tokens=3,
    )

    x, y, m = ds.get_batch("train", batch_size=1, device=torch.device("cpu"))

    assert x.shape == (1, block_size)
    assert y.shape == (1, block_size)
    assert m.shape == (1, block_size)

    # For start=0, y corresponds to original doc indices 1..8.
    # real_len=8 => valid y indices are 1..7.
    # resp_start=5 => supervise indices 5,6,7 only; the padded final position stays masked out.
    expected = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0]], dtype=torch.bool)
    assert torch.equal(m, expected)
