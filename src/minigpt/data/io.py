# src/minigpt/data/io.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from minigpt.common.paths import ensure_dir


@dataclass
class ShardWriter:
    output_dir: Path
    shard_size: int
    prefix: str = "shard"
    shard_index: int = 0
    count_in_shard: int = 0
    total_written: int = 0

    def __post_init__(self):
        ensure_dir(self.output_dir)
        self._fh = None

    def _open_new(self):
        if self._fh:
            self._fh.close()
        fname = f"{self.prefix}_{self.shard_index:05d}.jsonl"
        self._fh = open(self.output_dir / fname, "w", encoding="utf-8")
        self.count_in_shard = 0
        self.shard_index += 1

    def write(self, record: dict) -> None:
        if self._fh is None or self.count_in_shard >= self.shard_size:
            self._open_new()
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count_in_shard += 1
        self.total_written += 1

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
