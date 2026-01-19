# src/minigpt/data/pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from minigpt.common.logging import get_logger
from minigpt.common.paths import ensure_dir
from minigpt.common.text_cleaning import FilterConfig, clean_text_structural, clean_text_final, passes_filters
from minigpt.data.dedupe import DedupeConfig, DedupeState
from minigpt.data.io import ShardWriter
from minigpt.data.mix import MixedStream
from minigpt.data.sources import SourceSpec

log = get_logger("minigpt.data.pipeline")


@dataclass
class PipelineConfig:
    output_dir: str
    seed: int
    max_examples_total: int
    shard_size: int
    filters: FilterConfig
    dedupe: DedupeConfig
    sources: list[SourceSpec]


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    filters = cfg.get("filters", {})
    dedupe = cfg.get("dedupe", {})
    simhash = dedupe.get("simhash", {})

    sources_cfg = cfg["sources"]
    sources: list[SourceSpec] = []
    for s in sources_cfg:
        sources.append(
            SourceSpec(
                name=s["name"],
                kind=s["kind"],
                dataset=s["dataset"],
                subset=s.get("subset", None),
                split=s["split"],
                text_field=s["text_field"],
                weight=float(s["weight"]),
            )
        )

    return PipelineConfig(
        output_dir=str(cfg["output_dir"]),
        seed=int(cfg.get("seed", 1337)),
        max_examples_total=int(cfg.get("max_examples_total", 50000)),
        shard_size=int(cfg.get("shard_size", 5000)),
        filters=FilterConfig(
            min_chars=int(filters.get("min_chars", 200)),
            max_chars=int(filters.get("max_chars", 20000)),
            max_nonalpha_ratio=float(filters.get("max_nonalpha_ratio", 0.45)),
            max_repeated_line_ratio=float(filters.get("max_repeated_line_ratio", 0.35)),
        ),
        dedupe=DedupeConfig(
            exact=bool(dedupe.get("exact", True)),
            simhash_enabled=bool(simhash.get("enabled", True)),
            hamming_threshold=int(simhash.get("hamming_threshold", 4)),
        ),
        sources=sources,
    )


def build_corpus(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = ensure_dir(cfg.output_dir)

    writer = ShardWriter(output_dir=Path(out_dir), shard_size=cfg.shard_size, prefix="step1")
    dedupe_state = DedupeState()

    mixed = MixedStream(cfg.sources, seed=cfg.seed)

    stats: dict[str, Any] = {
        "seen": 0,
        "written": 0,
        "filtered": 0,
        "exact_dups": 0,
        "near_dups": 0,
        "by_source_written": {s.name: 0 for s in cfg.sources},
    }

    pbar = tqdm(total=cfg.max_examples_total, desc="Writing examples")
    try:
        for ex in mixed:
            if stats["written"] >= cfg.max_examples_total:
                break

            stats["seen"] += 1
            raw = ex["text"]
            src = ex["source"]

            text = clean_text_structural(raw)
            if not passes_filters(text, cfg.filters):
                stats["filtered"] += 1
                continue

            text = clean_text_final(text)

            if cfg.dedupe.exact and dedupe_state.is_exact_dup(text):
                stats["exact_dups"] += 1
                continue

            if cfg.dedupe.simhash_enabled and dedupe_state.is_near_dup(
                text, threshold=cfg.dedupe.hamming_threshold
            ):
                stats["near_dups"] += 1
                continue

            record = {
                "text": text,
                "source": src,
                "meta": ex.get("meta", {}),
            }
            writer.write(record)
            stats["written"] += 1
            stats["by_source_written"][src] += 1
            pbar.update(1)

        log.info("Done. Wrote %d examples to %s", stats["written"], str(out_dir))
        log.info("Stats: %s", stats)

        # Write a small summary file
        summary_path = Path(out_dir) / "summary.yaml"
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(stats, f, sort_keys=False, allow_unicode=True)
        log.info("Wrote summary to %s", str(summary_path))

    finally:
        pbar.close()
        writer.close()
