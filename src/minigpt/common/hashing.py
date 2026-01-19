# src/minigpt/common/hashing.py

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import regex as re


_WORD_RE = re.compile(r"\p{L}[\p{L}\p{N}_'-]{1,}", re.UNICODE)


def stable_hash64(s: str) -> int:
    # 64-bit hash derived from sha1 for stability across runs/platforms
    h = hashlib.sha1(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def tokenize_for_simhash(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class SimHash:
    value: int

    @staticmethod
    def from_text(text: str, bits: int = 64) -> SimHash:
        tokens = tokenize_for_simhash(text)
        if not tokens:
            return SimHash(0)

        v = [0] * bits
        for t in tokens:
            h = stable_hash64(t)
            for i in range(bits):
                bit = (h >> i) & 1
                v[i] += 1 if bit else -1

        out = 0
        for i in range(bits):
            if v[i] > 0:
                out |= (1 << i)

        return SimHash(out)

    def hamming_distance(self, other: SimHash) -> int:
        x = self.value ^ other.value
        # popcount
        return x.bit_count()
