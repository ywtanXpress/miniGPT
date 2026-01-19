# src/minigpt/data/dedupe.py

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from minigpt.common.hashing import SimHash


def exact_fingerprint(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class DedupeConfig:
    exact: bool = True
    simhash_enabled: bool = True
    hamming_threshold: int = 4


@dataclass
class DedupeState:
    exact_seen: set[str] = field(default_factory=set)
    simhashes: list[int] = field(default_factory=list)

    def is_exact_dup(self, text: str) -> bool:
        fp = exact_fingerprint(text)
        if fp in self.exact_seen:
            return True
        self.exact_seen.add(fp)
        return False

    def is_near_dup(self, text: str, threshold: int) -> bool:
        sh = SimHash.from_text(text).value
        # naive scan is OK for laptop-scale; later we’ll replace with LSH buckets
        for prev in self.simhashes:
            if (sh ^ prev).bit_count() <= threshold:
                return True
        self.simhashes.append(sh)
        return False
