# src/minigpt/data/mix.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from minigpt.data.sources import SourceSpec, stream_hf_text


@dataclass
class MixedStream:
    specs: list[SourceSpec]
    seed: int = 1337

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed)

        # Build iterators per source
        iters = [(spec, iter(stream_hf_text(spec))) for spec in self.specs]

        # Normalize weights
        total_w = sum(max(0.0, s.weight) for s, _ in iters)
        probs = [max(0.0, s.weight) / total_w for s, _ in iters]

        # Weighted sampling with replacement among sources;
        # if a source iterator exhausts (rare in streaming), we drop it.
        alive = iters
        alive_probs = probs

        while alive:
            idx = rng.choices(range(len(alive)), weights=alive_probs, k=1)[0]
            spec, it_ = alive[idx]
            try:
                yield next(it_)
            except StopIteration:
                alive.pop(idx)
                alive_probs.pop(idx)
                if alive_probs:
                    s = sum(alive_probs)
                    alive_probs = [p / s for p in alive_probs]
