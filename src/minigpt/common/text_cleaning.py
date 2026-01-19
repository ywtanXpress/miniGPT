# src/minigpt/common/text_cleaning.py

from __future__ import annotations

from dataclasses import dataclass

import regex as re


_CONTROL_RE = re.compile(r"[\p{C}&&[^\n\t]]", re.UNICODE)
_SPACES_TABS_RE = re.compile(r"[ \t]+", re.UNICODE)
_ANY_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)
_MANY_PUNCT_RE = re.compile(r"[\p{P}\p{S}]{8,}", re.UNICODE)


def normalize_whitespace_preserve_newlines(text: str) -> str:
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES_TABS_RE.sub(" ", text)
    return text.strip()


def normalize_whitespace_final(text: str) -> str:
    return _ANY_WHITESPACE_RE.sub(" ", text).strip()


def strip_weird_artifacts(text: str) -> str:
    # Remove extremely long punctuation runs (common in scraped spam)
    text = _MANY_PUNCT_RE.sub(" ", text)
    return text


def clean_text_structural(text: str) -> str:
    text = strip_weird_artifacts(text)
    text = normalize_whitespace_preserve_newlines(text)
    return text


def clean_text_final(text: str) -> str:
    return normalize_whitespace_final(text)


def nonalpha_ratio(text: str) -> float:
    if not text:
        return 1.0
    alpha = sum(ch.isalpha() for ch in text)
    return 1.0 - (alpha / max(len(text), 1))


def repeated_line_ratio(text: str) -> float:
    # Many scraped pages repeat nav/footer lines. Approximate by line uniqueness.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 5:
        return 0.0
    unique = len(set(lines))
    return 1.0 - unique / len(lines)


@dataclass
class FilterConfig:
    min_chars: int = 200
    max_chars: int = 20000
    max_nonalpha_ratio: float = 0.45
    max_repeated_line_ratio: float = 0.35


def passes_filters(text: str, cfg: FilterConfig) -> bool:
    n = len(text)
    if n < cfg.min_chars or n > cfg.max_chars:
        return False
    if nonalpha_ratio(text) > cfg.max_nonalpha_ratio:
        return False
    if repeated_line_ratio(text) > cfg.max_repeated_line_ratio:
        return False
    return True